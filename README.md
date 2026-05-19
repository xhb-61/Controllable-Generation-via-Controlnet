# Controllable Generation via ControlNet

This repository contains my ControlNet project for controllable image generation with
Stable Diffusion. It is based on the official
[lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) implementation and
keeps the runnable Gradio demos for multiple control conditions, including Canny
edges, HED boundaries, human pose, depth, normal maps, semantic segmentation,
scribbles, and M-LSD lines.

## Project Notes

- Main demos are the `gradio_*2image.py` scripts.
- `gradio_canny2image.py` and `gradio_hed2image.py` were adapted for the server
  environment used in this project.
- Model checkpoints and downloaded CLIP/OpenAI weights are intentionally excluded
  from git because they are very large. Put ControlNet checkpoints under `models/`
  and detector checkpoints under `annotator/ckpts/` before running the demos.
- On the `lf2` server, the temporary Gradio directory can be configured with:

```bash
export GRADIO_TEMP_DIR=~/local/ControlNet-main/gradio_tmp
```

## Quick Start

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate control
```

Download the required weights from the
[ControlNet Hugging Face page](https://huggingface.co/lllyasviel/ControlNet), then
run a demo, for example:

```bash
python gradio_canny2image.py
python gradio_hed2image.py
```

The application launches a Gradio interface and listens on `0.0.0.0`, which makes
it usable through a forwarded server port.

## Mini Fine-Tuning Task

This repo now includes a small Canny-edge fine-tuning task for demonstrating a
complete ControlNet training loop:

```bash
python scripts/prepare_mini_canny_dataset.py --overwrite
python train_mini_canny.py --max-steps 20
python scripts/infer_mini_canny.py --checkpoint training/mini_canny_runs/checkpoints/last.ckpt
```

See [docs/mini_canny_training.md](docs/mini_canny_training.md) for the full
dataset, training, checkpointing, and sampling workflow.

## Upstream ControlNet README

Original ControlNet repository: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
