"""
blocks.py
Block (node type) definitions and lookup helpers.
"""

from typing import Optional

# Block registry
# Structure:
#   SECTIONS[section_name][category_name] = [block_def, ...]
#
# block_def keys:
#   label   : str           - display name and unique identifier
#   color   : (r, g, b)     - title bar / text accent colour
#   params  : [str]         - editable parameter field names
#   inputs  : [str]         - input pin names
#   outputs : [str]         - output pin names

SECTIONS: dict = {
    "Model Creation": {
        "Layers": [
            {"label": "Linear",          "color": (100, 180, 255), "params": ["in_features", "out_features"],                                     "inputs": ["x"],           "outputs": ["out"]},
            {"label": "Conv2D",          "color": (120, 220, 140), "params": ["in_channels", "out_channels", "kernel_size", "stride", "padding"], "inputs": ["x"],           "outputs": ["out"]},
            {"label": "ConvTranspose2D", "color": (120, 220, 140), "params": ["in_channels", "out_channels", "kernel_size", "stride"],            "inputs": ["x"],           "outputs": ["out"]},
            {"label": "Flatten",         "color": (100, 180, 255), "params": ["start_dim", "end_dim"],                                            "inputs": ["x"],           "outputs": ["out"]},
        ],
        "Activations": [
            {"label": "ReLU",      "color": (255, 180, 80), "params": [],                  "inputs": ["x"], "outputs": ["out"]},
            {"label": "Sigmoid",   "color": (255, 180, 80), "params": [],                  "inputs": ["x"], "outputs": ["out"]},
            {"label": "Tanh",      "color": (255, 180, 80), "params": [],                  "inputs": ["x"], "outputs": ["out"]},
            {"label": "Softmax",   "color": (255, 180, 80), "params": ["dim"],             "inputs": ["x"], "outputs": ["out"]},
            {"label": "GELU",      "color": (255, 180, 80), "params": [],                  "inputs": ["x"], "outputs": ["out"]},
            {"label": "LeakyReLU", "color": (255, 180, 80), "params": ["negative_slope"],  "inputs": ["x"], "outputs": ["out"]},
        ],
        "Normalization": [
            {"label": "BatchNorm2D", "color": (200, 130, 255), "params": ["num_features", "eps", "momentum"], "inputs": ["x"], "outputs": ["out"]},
            {"label": "LayerNorm",   "color": (200, 130, 255), "params": ["normalized_shape", "eps"],         "inputs": ["x"], "outputs": ["out"]},
            {"label": "GroupNorm",   "color": (200, 130, 255), "params": ["num_groups", "num_channels"],      "inputs": ["x"], "outputs": ["out"]},
            {"label": "Dropout",     "color": (200, 130, 255), "params": ["p"],                               "inputs": ["x"], "outputs": ["out"]},
        ],
        "Pooling": [
            {"label": "MaxPool2D",         "color": (255, 120, 120), "params": ["kernel_size", "stride", "padding"], "inputs": ["x"], "outputs": ["out"]},
            {"label": "AvgPool2D",         "color": (255, 120, 120), "params": ["kernel_size", "stride", "padding"], "inputs": ["x"], "outputs": ["out"]},
            {"label": "AdaptiveAvgPool2D", "color": (255, 120, 120), "params": ["output_size"],                      "inputs": ["x"], "outputs": ["out"]},
        ],
        "I/O": [
            {"label": "Input",  "color": (80, 220, 200), "params": ["shape"], "inputs": [],    "outputs": ["out"]},
            {"label": "Output", "color": (80, 220, 200), "params": ["shape"], "inputs": ["x"], "outputs": []},
        ],
    },
    "Training": {
        "Pipeline Inputs": [
            {"label": "ModelBlock",      "color": (80,  180, 255), "params": [],       "inputs": ["images"],  "outputs": ["predictions"]},
            {"label": "DataLoaderBlock", "color": (180, 100, 255), "params": [],       "inputs": [],          "outputs": ["images", "labels"]},
        ],
        "Loss Functions": [
            {"label": "CrossEntropyLoss", "color": (255, 160, 100), "params": ["weight", "ignore_index", "reduction"], "inputs": ["pred", "target"], "outputs": ["loss"]},
            {"label": "MSELoss",          "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"]},
            {"label": "BCELoss",          "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"]},
            {"label": "BCEWithLogits",    "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"]},
            {"label": "NLLLoss",          "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"]},
            {"label": "HuberLoss",        "color": (255, 160, 100), "params": ["delta", "reduction"],                  "inputs": ["pred", "target"], "outputs": ["loss"]},
            {"label": "KLDivLoss",        "color": (255, 160, 100), "params": ["reduction"],                           "inputs": ["pred", "target"], "outputs": ["loss"]},
        ],
        "Optimizers": [
            {"label": "Adam",    "color": (100, 220, 180), "params": ["lr", "betas", "eps", "weight_decay"], "inputs": ["params"], "outputs": []},
            {"label": "AdamW",   "color": (100, 220, 180), "params": ["lr", "betas", "eps", "weight_decay"], "inputs": ["params"], "outputs": []},
            {"label": "SGD",     "color": (100, 220, 180), "params": ["lr", "momentum", "weight_decay"],     "inputs": ["params"], "outputs": []},
            {"label": "RMSprop", "color": (100, 220, 180), "params": ["lr", "alpha",   "eps", "weight_decay"],"inputs": ["params"], "outputs": []},
            {"label": "Adagrad", "color": (100, 220, 180), "params": ["lr", "lr_decay","weight_decay"],      "inputs": ["params"], "outputs": []},
            {"label": "LBFGS",   "color": (100, 220, 180), "params": ["lr", "max_iter","history_size"],      "inputs": ["params"], "outputs": []},
        ],
    },
    "Data Prep": {
        "Datasets": [
            {"label": "MNIST",        "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [],      "outputs": ["img"]},
            {"label": "CIFAR10",      "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [],      "outputs": ["img"]},
            {"label": "CIFAR100",     "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [],      "outputs": ["img"]},
            {"label": "FashionMNIST", "color": (220, 180, 255), "params": ["root", "train", "download"], "inputs": [],      "outputs": ["img"]},
            {"label": "ImageFolder",  "color": (220, 180, 255), "params": ["root"],                      "inputs": [],      "outputs": ["img"]},
        ],
        "Augmentation": [
            {"label": "Resize",         "color": (255, 200, 120), "params": ["size"],                                     "inputs": ["img"], "outputs": ["img"]},
            {"label": "CenterCrop",     "color": (255, 200, 120), "params": ["size"],                                     "inputs": ["img"], "outputs": ["img"]},
            {"label": "RandomCrop",     "color": (255, 200, 120), "params": ["size", "padding"],                          "inputs": ["img"], "outputs": ["img"]},
            {"label": "RandomHFlip",    "color": (255, 200, 120), "params": ["p"],                                        "inputs": ["img"], "outputs": ["img"]},
            {"label": "RandomVFlip",    "color": (255, 200, 120), "params": ["p"],                                        "inputs": ["img"], "outputs": ["img"]},
            {"label": "ColorJitter",    "color": (255, 200, 120), "params": ["brightness","contrast","saturation","hue"], "inputs": ["img"], "outputs": ["img"]},
            {"label": "RandomRotation", "color": (255, 200, 120), "params": ["degrees"],                                  "inputs": ["img"], "outputs": ["img"]},
            {"label": "GaussianBlur",   "color": (255, 200, 120), "params": ["kernel_size", "sigma"],                     "inputs": ["img"], "outputs": ["img"]},
            {"label": "RandomErasing",  "color": (255, 200, 120), "params": ["p", "scale", "ratio"],                      "inputs": ["img"], "outputs": ["img"]},
            {"label": "Normalize",      "color": (255, 200, 120), "params": ["mean", "std"],                              "inputs": ["img"], "outputs": ["img"]},
            {"label": "ToTensor",       "color": (255, 200, 120), "params": [],                                           "inputs": ["img"], "outputs": ["img"]},
            {"label": "Grayscale",      "color": (255, 200, 120), "params": ["num_output_channels"],                      "inputs": ["img"], "outputs": ["img"]},
        ],
        "DataLoader": [
            {"label": "DataLoader (train)", "color": (200, 160, 255), "params": ["batch_size", "shuffle", "num_workers", "pin_memory"], "inputs": ["img"], "outputs": []},
            {"label": "DataLoader (val)",   "color": (180, 140, 235), "params": ["batch_size", "num_workers", "pin_memory"],            "inputs": ["img"], "outputs": []},
        ],
    },
}


def get_block_def(label: str) -> Optional[dict]:
    """Return the block definition dict for a given label, or None."""
    for section in SECTIONS.values():
        for block_list in section.values():
            for block in block_list:
                if block["label"] == label:
                    return block
    return None


def all_block_labels() -> list[str]:
    """Return a flat list of every block label across all sections."""
    labels = []
    for section in SECTIONS.values():
        for block_list in section.values():
            for block in block_list:
                labels.append(block["label"])
    return labels
