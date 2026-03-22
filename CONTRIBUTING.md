# Contributing to MLForge

Thanks for your interest in contributing! MLForge is a solo project that's grown faster than expected, and outside contributions are very welcome.

---

## Ways to contribute
- Reporting bugs with clear reproduction steps
- Suggesting new block types or features
- Improving the README or documentation
- Sharing the project with others

---

## Reporting bugs

Open an issue on GitHub with:

1. What you were doing when it broke
2. What you expected to happen
3. What actually happened
4. Your OS, Python version, and PyTorch version
5. The full error message or traceback if there is one

The more detail the better. 

---

## Suggesting features

Open an issue tagged `enhancement`. Describe:

- What you want to do that you currently can't
- Why it would be useful
- Any ideas on how it could work

---

## Contributing code

### Setup

```
git clone https://github.com/zaina-ml/ml_forge
cd ml_forge
python -m ml_forge
```

### Before you start

- Open an issue first for anything significant
- Small bug fixes and typo fixes don't need an issue first

### Adding a new block

Blocks are defined in `ml_forge/blocks.py`. Each block is a dict with:

```
{
    "label":   "MyBlock",
    "color":   (R, G, B),
    "params":  ["param1", "param2"],   # input text fields
    "inputs":  ["x"],                  # input pins
    "outputs": ["out"],                # output pins
}
```

Add it to the appropriate section in `SECTIONS`. If the block needs to generate PyTorch code on export, add a corresponding entry in `ml_forge/engine/generator.py` in the relevant map (`_LAYER_MAP`, `_LOSS_MAP`, `_OPTIM_MAP`, or `_DATASET_MAP`).


### Pull requests

1. Fork the repo
2. Create a branch: `git checkout -b my-feature`
3. Make your changes
4. Test that the app launches and the affected feature works
5. Submit a pull request with a clear description of what changed and why

Keep pull requests focused.

---

## What's on the roadmap

Stuff that needs to be added:

1. Support for non-computer-vision datasets and custom datasets.
2. Ability for users to import PyTorch definitions for custom blocks
3. Skip connections or multi-input models 

---

## Questions

Open an issue or start a GitHub Discussion. Happy to help you get oriented in the codebase.
