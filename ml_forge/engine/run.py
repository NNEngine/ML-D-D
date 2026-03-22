"""
run.py
Real PyTorch training execution in a background thread.

The thread communicates with the UI via a queue:
  _result_queue  — epoch results posted by the thread, drained each frame

PyTorch is required. The caller (training.py) is responsible for checking
that torch is installed before calling start_real_training().

Public API:
    start_training(cfg)   start the background thread
    stop_training()       signal the thread to stop
    pause_training()      toggle pause
    drain_result_queue()       called each frame; updates UI with results
"""

from __future__ import annotations

import pathlib
import queue
import threading
import time

import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.ui.console import log
from ml_forge.engine.graph import build_graph, topological_sort, get_tab_by_role


# Resolving Device

def _resolve_device(setting: str) -> "torch.device":
    import torch
    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(setting)


# Model builder, instantiates from graph

def _build_torch_model(device):
    """
    Walk the Model tab graph and return a live nn.Module.
    Raises ValueError if the graph cannot be instantiated.
    """
    import torch.nn as nn
    from ml_forge.engine.generator import _LAYER_MAP, _p, _fill

    tab = get_tab_by_role("model")
    if tab is None:
        raise ValueError("No Model tab found.")

    try:
        nodes = topological_sort(tab)
    except Exception as e:
        raise ValueError(f"Model graph error: {e}")

    layer_nodes = [n for n in nodes if n.block_label not in ("Input", "Output")]
    if not layer_nodes:
        raise ValueError("Model tab has no layer nodes.")

    layers = []
    for node in layer_nodes:
        label = node.block_label
        if label not in _LAYER_MAP:
            log(f"Skipping unsupported layer: {label}", "warning")
            continue
        module_name, template = _LAYER_MAP[label]
        args_str = _fill(template, node)

        # Check for unfilled required params
        if "..." in args_str:
            raise ValueError(
                f"Node '{label}' has unfilled parameters. "
                f"Fill all required fields before training."
            )

        # Dynamically evaluate the module expression
        import torch.nn as nn  # noqa: F811
        try:
            module = eval(f"nn.{module_name.replace('nn.', '')}({args_str})")
            layers.append(module)
        except Exception as e:
            raise ValueError(f"Could not instantiate {label}({args_str}): {e}")

    model = nn.Sequential(*layers).to(device)
    return model


#  DataLoader builder

def _build_dataloaders(device, val_split: float, seed: int, shuffle: bool):
    """
    Walk the Data Prep graph and build train and val DataLoaders.

    Supports two modes:
      A) Dual chain — user places two Dataset nodes (train=True, train=False)
         each with their own transform chain wired into DataLoader (train)
         and DataLoader (val). The graph is walked separately for each loader.
      B) Single chain — one Dataset node wired into DataLoader (train) only.
         Falls back to val_split from Training Config to split the dataset.
    """
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from torchvision.datasets import ImageFolder
    from ml_forge.engine.graph import (topological_sort, get_tab_by_role,
                               _DATASET_BLOCKS, _AUG_BLOCKS, build_graph)

    tab = get_tab_by_role("data_prep")
    if tab is None:
        raise ValueError("No Data Prep tab found.")

    try:
        ordered = topological_sort(tab)
    except Exception as e:
        raise ValueError(f"Data Prep graph error: {e}")

    graph  = build_graph(tab)
    nodes  = list(graph.values())

    TORCHVISION_DATASETS = {
        "MNIST":        datasets.MNIST,
        "CIFAR10":      datasets.CIFAR10,
        "CIFAR100":     datasets.CIFAR100,
        "FashionMNIST": datasets.FashionMNIST,
    }

    # Detect mode
    train_loader_node = next(
        (n for n in nodes if n.block_label in ("DataLoader (train)", "DataLoader")), None)
    val_loader_node   = next(
        (n for n in nodes if n.block_label == "DataLoader (val)"), None)

    dual_chain = val_loader_node is not None

    # Helper: build transform list from a subchain
    def _build_transform(chain_nodes):
        tlist = []
        for n in chain_nodes:
            label = n.block_label
            if label not in _AUG_BLOCKS:
                continue
            if label == "ToTensor":
                tlist.append(transforms.ToTensor())
            elif label == "Resize":
                tlist.append(transforms.Resize(int(n.params.get("size","224") or "224")))
            elif label == "CenterCrop":
                tlist.append(transforms.CenterCrop(int(n.params.get("size","224") or "224")))
            elif label == "RandomCrop":
                sz  = int(n.params.get("size","32") or "32")
                pad = int(n.params.get("padding","0") or "0")
                tlist.append(transforms.RandomCrop(sz, padding=pad))
            elif label == "RandomHFlip":
                tlist.append(transforms.RandomHorizontalFlip(float(n.params.get("p","0.5") or "0.5")))
            elif label == "RandomVFlip":
                tlist.append(transforms.RandomVerticalFlip(float(n.params.get("p","0.5") or "0.5")))
            elif label == "RandomRotation":
                tlist.append(transforms.RandomRotation(float(n.params.get("degrees","15") or "15")))
            elif label == "Normalize":
                mean = n.params.get("mean","0.5").strip() or "0.5"
                std  = n.params.get("std", "0.5").strip() or "0.5"
                try:
                    mean = eval(mean) if "[" in mean or "(" in mean else [float(mean)]*3
                    std  = eval(std)  if "[" in std  or "(" in std  else [float(std)] *3
                except Exception:
                    mean, std = [0.5,0.5,0.5], [0.5,0.5,0.5]
                tlist.append(transforms.Normalize(mean=mean, std=std))
            elif label == "ColorJitter":
                tlist.append(transforms.ColorJitter(
                    float(n.params.get("brightness","0") or "0"),
                    float(n.params.get("contrast",  "0") or "0"),
                    float(n.params.get("saturation","0") or "0"),
                    float(n.params.get("hue",       "0") or "0"),
                ))
            elif label == "GaussianBlur":
                ks = int(n.params.get("kernel_size","3") or "3")
                if ks % 2 == 0: ks += 1
                tlist.append(transforms.GaussianBlur(ks, sigma=float(n.params.get("sigma","1.0") or "1.0")))
            elif label == "RandomErasing":
                tlist.append(transforms.RandomErasing(p=float(n.params.get("p","0.5") or "0.5")))
            elif label == "Grayscale":
                tlist.append(transforms.Grayscale(int(n.params.get("num_output_channels","1") or "1")))
        if not tlist:
            tlist = [transforms.ToTensor()]
        return transforms.Compose(tlist)

    # Helper: instantiate a dataset node
    def _make_dataset(ds_node, transform):
        label    = ds_node.block_label
        root     = ds_node.params.get("root","./data").strip() or "./data"
        download = ds_node.params.get("download","True").strip()
        train_f  = ds_node.params.get("train","True").strip()
        is_train = train_f.lower() != "false"
        if label in TORCHVISION_DATASETS:
            return TORCHVISION_DATASETS[label](
                root=root, train=is_train,
                download=(download.lower() != "false"),
                transform=transform,
            )
        elif label == "ImageFolder":
            if not pathlib.Path(root).exists():
                raise ValueError(f"ImageFolder root '{root}' does not exist.")
            return ImageFolder(root=root, transform=transform)
        else:
            raise ValueError(
                f"'{label}' is not yet supported for training in this version.\n"
                f"Supported datasets: MNIST, CIFAR10, CIFAR100, FashionMNIST, ImageFolder."
            )

    # find the chain feeding into a loader node
    def _chain_for_loader(loader_node):
        """Return ordered nodes that are ancestors of loader_node."""
        loader_ntag = loader_node.ntag
        ancestors   = set()
        # Walk backwards via links — any node whose output connects to loader
        # or to another ancestor is an ancestor
        changed = True
        targets = {loader_ntag}
        while changed:
            changed = False
            for link_tag, (a1, a2) in tab["links"].items():
                import dearpygui.dearpygui as dpg
                if isinstance(a1, int): a1 = dpg.get_item_alias(a1) or str(a1)
                if isinstance(a2, int): a2 = dpg.get_item_alias(a2) or str(a2)
                dst_parts = a2.split("_")
                src_parts = a1.split("_")
                if len(dst_parts) >= 3 and len(src_parts) >= 3:
                    dst_ntag = f"node_{dst_parts[1]}_{dst_parts[2]}"
                    src_ntag = f"node_{src_parts[1]}_{src_parts[2]}"
                    if dst_ntag in targets and src_ntag not in ancestors:
                        ancestors.add(src_ntag)
                        targets.add(src_ntag)
                        changed = True
        return [n for n in ordered if n.ntag in ancestors]

    # Mode A: dual chain
    if dual_chain:
        train_chain = _chain_for_loader(train_loader_node)
        val_chain   = _chain_for_loader(val_loader_node)

        train_ds_node = next((n for n in train_chain if n.block_label in _DATASET_BLOCKS), None)
        val_ds_node   = next((n for n in val_chain   if n.block_label in _DATASET_BLOCKS), None)

        if train_ds_node is None:
            raise ValueError("DataLoader (train) chain has no Dataset node.")
        if val_ds_node is None:
            raise ValueError("DataLoader (val) chain has no Dataset node.")

        train_transform = _build_transform(train_chain)
        val_transform   = _build_transform(val_chain)

        _result_queue.put({"type":"log","level":"info",
            "msg": f"Train transforms: {[type(t).__name__ for t in train_transform.transforms]}"})
        _result_queue.put({"type":"log","level":"info",
            "msg": f"Val transforms:   {[type(t).__name__ for t in val_transform.transforms]}"})

        train_ds = _make_dataset(train_ds_node, train_transform)
        val_ds   = _make_dataset(val_ds_node,   val_transform)

        def _loader_params(ln):
            bs = int(ln.params.get("batch_size", "32") or "32")
            nw = int(ln.params.get("num_workers","2")  or "2")
            pm = ln.params.get("pin_memory","True").lower() != "false"
            return bs, nw, pm

        tbs, tnw, tpm = _loader_params(train_loader_node)
        vbs, vnw, vpm = _loader_params(val_loader_node)

        train_loader = DataLoader(train_ds, batch_size=tbs, shuffle=shuffle,
                                  num_workers=tnw, pin_memory=tpm and device.type=="cuda")
        val_loader   = DataLoader(val_ds,   batch_size=vbs, shuffle=False,
                                  num_workers=vnw, pin_memory=vpm and device.type=="cuda")
        return train_loader, val_loader

    # Mode B: single chain with val_split fallback
    train_chain = _chain_for_loader(train_loader_node) if train_loader_node else ordered
    ds_node     = next((n for n in train_chain if n.block_label in _DATASET_BLOCKS), None)
    if ds_node is None:
        raise ValueError("No Dataset node found in Data Prep tab.")

    transform = _build_transform(train_chain)
    _result_queue.put({"type":"log","level":"info",
        "msg": f"Transforms: {[type(t).__name__ for t in transform.transforms]}"})

    dataset = _make_dataset(ds_node, transform)

    generator = torch.Generator().manual_seed(seed)
    if val_split > 0:
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    else:
        train_ds, val_ds = dataset, None

    loader_node = train_loader_node
    bs = int(loader_node.params.get("batch_size", "32") or "32") if loader_node else 32
    nw = int(loader_node.params.get("num_workers","2")  or "2")  if loader_node else 2
    pm = (loader_node.params.get("pin_memory","True").lower() != "false") if loader_node else True

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle,
                              num_workers=nw, pin_memory=pm and device.type=="cuda")
    val_loader   = (DataLoader(val_ds, batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=pm and device.type=="cuda")
                    if val_ds else None)

    return train_loader, val_loader


#  Training setup builder

def _build_criterion_and_optimizer(model, device):
    import torch.nn as nn
    import torch.optim as optim
    from ml_forge.engine.generator import _LOSS_MAP, _OPTIM_MAP, _fill

    tab = get_tab_by_role("training")
    if tab is None:
        raise ValueError("No Training tab found.")

    graph = build_graph(tab)
    nodes = list(graph.values())

    loss_node = next((n for n in nodes if n.block_label in _LOSS_MAP), None)
    if loss_node:
        module_name, template = _LOSS_MAP[loss_node.block_label]
        args_str = _fill(template, loss_node).replace("...", "")
        try:
            criterion = eval(f"nn.{module_name.replace('nn.', '')}({args_str})").to(device)
        except Exception:
            criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optim_node = next((n for n in nodes if n.block_label in _OPTIM_MAP), None)
    if optim_node:
        module_name, template = _OPTIM_MAP[optim_node.block_label]
        args_str = _fill(template, optim_node).replace("...", "")
        try:
            optimizer = eval(
                f"optim.{module_name.replace('optim.', '')}(model.parameters(), {args_str})"
            )
        except Exception:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    return criterion, optimizer

#  Thread queues and control events

_result_queue: queue.Queue = queue.Queue()
_stop_event:  threading.Event = threading.Event()
_pause_event: threading.Event = threading.Event()
_train_thread: threading.Thread | None = None


#  Training thread

def _training_thread(cfg: dict) -> None:
    """
    Runs in a background thread. Builds model/data/optimiser, trains,
    and posts results to _result_queue.
    """
    import torch

    try:
        device       = _resolve_device(cfg["device"])
        epochs       = cfg["epochs"]
        val_split    = cfg["val_split"]
        seed         = cfg["seed"]
        shuffle      = cfg["shuffle"]
        use_amp      = cfg["amp"] and device.type == "cuda"
        ckpt_dir     = pathlib.Path(cfg["ckpt_dir"])
        ckpt_every   = cfg["ckpt_every"]
        ckpt_best    = cfg["ckpt_best"]
        ckpt_monitor = cfg["ckpt_monitor"]
        es_enable    = cfg["es_enable"]
        es_patience  = cfg["es_patience"]
        es_min_delta = cfg["es_min_delta"]

        _result_queue.put({"type": "log", "msg": f"Device: {device}", "level": "info"})

        # Validate checkpoint dir early
        if ckpt_dir:
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Verify write permission by touching a temp file
                test_file = ckpt_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                raise ValueError(
                    f"Cannot write to checkpoint directory: {ckpt_dir}\n"
                    f"Check folder permissions or choose a different Save Dir."
                )
            except Exception as e:
                raise ValueError(f"Checkpoint directory error: {e}")

        # Build
        model = _build_torch_model(device)
        _result_queue.put({"type": "log",
                           "msg": f"Model: {sum(p.numel() for p in model.parameters()):,} params",
                           "level": "info"})

        train_loader, val_loader = _build_dataloaders(device, val_split, seed, shuffle)
        _result_queue.put({"type": "log",
                           "msg": f"Dataset: {len(train_loader.dataset):,} train samples",
                           "level": "info"})

        criterion, optimizer = _build_criterion_and_optimizer(model, device)
        scaler = torch.amp.GradScaler(device.type) if use_amp else None

        best_metric   = float("inf") if "loss" in ckpt_monitor else 0.0
        es_counter    = 0
        start_time    = time.time()

        # Epoch loop
        for epoch in range(1, epochs + 1):

            # Pause check
            while _pause_event.is_set():
                if _stop_event.is_set():
                    break
                time.sleep(0.1)

            if _stop_event.is_set():
                _result_queue.put({"type": "stopped"})
                return

            # Train
            model.train()
            train_loss   = 0.0
            batch_losses = []
            for batch_idx, (X, y) in enumerate(train_loader):
                if _stop_event.is_set():
                    break
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(X)
                    loss   = criterion(logits, y)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                batch_loss  = loss.item()
                train_loss += batch_loss
                batch_losses.append(batch_loss)

                # Post smoothed batch loss every 10 steps
                if (batch_idx + 1) % 10 == 0:
                    smoothed = sum(batch_losses[-10:]) / len(batch_losses[-10:])
                    _result_queue.put({
                        "type":       "batch",
                        "epoch":      epoch,
                        "batch":      batch_idx + 1,
                        "batch_loss": smoothed,
                        "total_batches": len(train_loader),
                    })

            train_loss /= max(len(train_loader), 1)

            # Validate
            val_loss = val_acc = None
            if val_loader:
                model.eval()
                v_loss = 0.0
                correct = total = 0
                with torch.inference_mode():
                    for X, y in val_loader:
                        X, y   = X.to(device), y.to(device)
                        logits = model(X)
                        v_loss += criterion(logits, y).item()
                        preds   = logits.argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total   += y.size(0)
                val_loss = v_loss / max(len(val_loader), 1)
                val_acc  = correct / total if total > 0 else 0.0

            # Checkpoint
            current_metric = (
                val_loss if ckpt_monitor == "val_loss" and val_loss is not None
                else val_acc if ckpt_monitor == "val_acc" and val_acc is not None
                else train_loss
            )
            is_best = (
                current_metric < best_metric if "loss" in ckpt_monitor
                else current_metric > best_metric
            )
            if is_best:
                best_metric = current_metric
                if ckpt_best:
                    torch.save(model.state_dict(), ckpt_dir / "best.pth")
                    _result_queue.put({"type": "log",
                                      "msg": f"Best checkpoint saved ({ckpt_monitor}={best_metric:.4f})",
                                      "level": "success"})
            if not ckpt_best and epoch % ckpt_every == 0:
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pth")

            # Post epoch result first, before any early-stop break
            _result_queue.put({
                "type":       "epoch",
                "epoch":      epoch,
                "total":      epochs,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            })

            # Early stopping
            if es_enable and val_loss is not None:
                improved = (
                    (best_metric - current_metric) > es_min_delta
                    if "loss" in ckpt_monitor
                    else (current_metric - best_metric) > es_min_delta
                )
                if not improved:
                    es_counter += 1
                    if es_counter >= es_patience:
                        _result_queue.put({"type": "log",
                                          "msg": f"Early stopping at epoch {epoch}.",
                                          "level": "warning"})
                        break
                else:
                    es_counter = 0

        elapsed = time.time() - start_time
        torch.save(model.state_dict(), ckpt_dir / "final.pth")
        _result_queue.put({
            "type":    "done",
            "elapsed": elapsed,
            "msg":     f"Training complete in {elapsed:.1f}s. Saved final.pth",
        })

    except Exception as e:
        _result_queue.put({"type": "error", "msg": str(e)})


#  Public

def start_training(cfg: dict) -> None:
    global _train_thread
    _stop_event.clear()
    _pause_event.clear()
    _train_thread = threading.Thread(
        target=_training_thread, args=(cfg,), daemon=True
    )
    _train_thread.start()


def stop_training() -> None:
    _stop_event.set()
    _pause_event.clear()


def pause_training() -> None:
    if _pause_event.is_set():
        _pause_event.clear()
    else:
        _pause_event.set()


def is_paused() -> bool:
    return _pause_event.is_set()


#  Result queue drain  (called every frame from render loop)

def drain_result_queue() -> None:
    """
    Drain all pending results from the training thread and update the UI.
    Must be called from the main thread every frame.
    """
    from ml_forge.ui.training import apply_train_btn_style, update_status_indicator

    try:
        while True:
            item = _result_queue.get_nowait()
            _handle_result(item)
    except queue.Empty:
        pass


def _handle_result(item: dict) -> None:
    from ml_forge.ui.training import apply_train_btn_style, update_status_indicator

    t = item["type"]

    if t == "log":
        log(item["msg"], item.get("level", "info"))

    elif t == "batch":
        # Fine-grained progress within the current epoch
        epoch        = item["epoch"]
        batch        = item["batch"]
        total_b      = item["total_batches"]
        batch_loss   = item["batch_loss"]
        total_epochs = state.train_state.get("total_epochs", 1)

        # Only update progress if we have a valid total_epochs already set
        # (avoids jitter before the first epoch result arrives)
        if total_epochs > 1 or state.train_state.get("epoch", 0) > 0:
            coarse = (epoch - 1) / total_epochs
            fine   = (batch / total_b) / total_epochs
            prog   = min(coarse + fine, epoch / total_epochs)
            dpg.set_value("train_progress", prog)

        if dpg.does_item_exist("train_progress"):
            dpg.configure_item("train_progress",
                               overlay=f"Epoch {epoch}/{total_epochs}  "
                                       f"batch {batch}/{total_b}  "
                                       f"loss={batch_loss:.4f}")

        # Append to batch loss series
        ts = state.train_state
        if "plot_batch_x" not in ts:
            ts["plot_batch_x"] = []
            ts["plot_batch_y"] = []
        global_step = (epoch - 1) * total_b + batch
        ts["plot_batch_x"].append(global_step / total_b)
        ts["plot_batch_y"].append(batch_loss)

        if dpg.does_item_exist("series_batch_loss"):
            # Apply smoothing window from the slider
            window = 10
            if dpg.does_item_exist("cfg_batch_smooth"):
                window = max(1, int(dpg.get_value("cfg_batch_smooth")))
            raw_y = ts["plot_batch_y"]
            if window > 1 and len(raw_y) >= window:
                # Exponential moving average for a smooth line
                alpha = 2.0 / (window + 1)
                smoothed = [raw_y[0]]
                for v in raw_y[1:]:
                    smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
                display_y = smoothed
            else:
                display_y = raw_y
            dpg.set_value("series_batch_loss",
                          [ts["plot_batch_x"], display_y])
            dpg.fit_axis_data("loss_y")

    elif t == "epoch":
        e     = item["epoch"]
        total = item["total"]
        tl    = item["train_loss"]
        vl    = item.get("val_loss")
        va    = item.get("val_acc")

        # Guard against duplicate epoch results (can occur if queue drains mid-epoch)
        last_logged = state.train_state.get("_last_logged_epoch", 0)
        if e <= last_logged:
            return
        state.train_state["_last_logged_epoch"] = e

        # Set total_epochs first so batch handler has it before next epoch starts
        state.train_state["total_epochs"] = total
        state.train_state["epoch"]        = e

        # Clean epoch-level progress
        dpg.set_value("train_progress", e / total)
        if dpg.does_item_exist("train_progress"):
            dpg.configure_item("train_progress",
                               overlay=f"Epoch {e}/{total}")

        if vl is not None and va is not None:
            log(f"Epoch {e:>3}/{total}  loss={tl:.4f}  "
                f"val_loss={vl:.4f}  val_acc={va:.4f}", "info")
        else:
            log(f"Epoch {e:>3}/{total}  loss={tl:.4f}", "info")

        # Update plots
        ts = state.train_state
        if "plot_epochs" not in ts:
            ts["plot_epochs"] = []
            ts["plot_tl"]     = []
            ts["plot_vl"]     = []
            ts["plot_ta"]     = []
            ts["plot_va"]     = []

        ts["plot_epochs"].append(e)
        ts["plot_tl"].append(tl)
        ts["plot_vl"].append(vl if vl is not None else tl)
        ts["plot_ta"].append(va if va is not None else 0.0)
        ts["plot_va"].append(va if va is not None else 0.0)

        if dpg.does_item_exist("series_train_loss"):
            xs = ts["plot_epochs"]
            dpg.set_value("series_train_loss", [xs, ts["plot_tl"]])
            dpg.set_value("series_val_loss",   [xs, ts["plot_vl"]])
            dpg.set_value("series_train_acc",  [xs, ts["plot_ta"]])
            dpg.set_value("series_val_acc",    [xs, ts["plot_va"]])
            dpg.fit_axis_data("loss_y")
            dpg.fit_axis_data("acc_y")

    elif t in ("done", "stopped"):
        state.train_state["status"] = "idle"
        state.train_state["epoch"]  = 0
        if "plot_epochs" in state.train_state:
            for key in ("plot_epochs", "plot_tl", "plot_vl", "plot_ta", "plot_va",
                        "plot_batch_x", "plot_batch_y"):
                state.train_state.pop(key, None)
        if dpg.does_item_exist("series_batch_loss"):
            dpg.set_value("series_batch_loss", [[], []])
        dpg.set_value("train_progress", 1.0 if t == "done" else 0.0)
        if "msg" in item:
            log(item["msg"], "success" if t == "done" else "warning")
        from ml_forge.engine.training_setup import reset_block_labels
        reset_block_labels()
        apply_train_btn_style()
        update_status_indicator()

    elif t == "error":
        state.train_state["status"] = "idle"
        log(f"Training error: {item['msg']}", "error")
        apply_train_btn_style()
        update_status_indicator()