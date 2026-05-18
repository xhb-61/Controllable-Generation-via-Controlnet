# Mini Canny Fine-Tuning Task

This is a small but complete ControlNet training task for interview discussion:
learn a Canny-edge-conditioned image generator from a curated subset of the
repository's sample images.

The goal is not to beat the official ControlNet checkpoint. The goal is to show
the full training loop: data construction, dataset loading, fine-tuning,
checkpointing, qualitative sampling, and failure analysis.

## Task Definition

- Input: a 512 x 512 Canny edge map.
- Text condition: a short prompt describing the target image.
- Target: the corresponding RGB image.
- Model: Stable Diffusion 1.5 + ControlNet.
- Training mode: lock the original SD weights and train the ControlNet branch.

## 1. Prepare Starting Weights

The fastest path is to fine-tune from the pretrained SD1.5 Canny ControlNet
checkpoint:

```text
models/control_sd15_canny.pth
```

This checkpoint is already ignored by git because it is several GB. Download it
from the official ControlNet Hugging Face page and keep it in `models/` on the
training machine.

If you want to train a new control branch from SD1.5 initialization instead,
build the initialization checkpoint first:

```bash
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
```

## 2. Build The Mini Dataset

The manifest is tracked at `configs/mini_canny_prompts.jsonl`. It points to
existing images in `test_imgs/` and pairs each image with a prompt.

```bash
python scripts/prepare_mini_canny_dataset.py \
  --manifest configs/mini_canny_prompts.jsonl \
  --output-root training/mini_canny \
  --resolution 512 \
  --low-threshold 100 \
  --high-threshold 200 \
  --overwrite
```

Expected output:

```text
training/mini_canny/
  prompt.json
  preview.png
  source/
  target/
```

`source/` stores Canny maps, `target/` stores RGB targets, and `prompt.json`
uses the original ControlNet JSONL format.

## 3. Train

A short smoke run:

```bash
python train_mini_canny.py \
  --data-root training/mini_canny \
  --resume-path models/control_sd15_canny.pth \
  --output-dir training/mini_canny_runs \
  --batch-size 1 \
  --max-steps 20 \
  --logger-freq 10 \
  --save-every-n-steps 10
```

A more useful overfit run for qualitative results:

```bash
python train_mini_canny.py \
  --data-root training/mini_canny \
  --resume-path models/control_sd15_canny.pth \
  --output-dir training/mini_canny_runs \
  --batch-size 1 \
  --max-steps 1000 \
  --logger-freq 100 \
  --save-every-n-steps 250 \
  --accumulate-grad-batches 4
```

Important outputs:

```text
training/mini_canny_runs/
  checkpoints/
    last.ckpt
    mini-canny-*.ckpt
  lightning_logs/
  image_log/
```

## 4. Sample From A Checkpoint

```bash
python scripts/infer_mini_canny.py \
  --checkpoint training/mini_canny_runs/checkpoints/last.ckpt \
  --data-root training/mini_canny \
  --output-dir training/mini_canny_eval \
  --limit 8 \
  --steps 20 \
  --scale 9.0
```

The script saves each control map, target image, and generated sample into
`training/mini_canny_eval/`.

## 5. What To Show In An Interview

- `training/mini_canny/preview.png`: proves the dataset construction is correct.
- `training/mini_canny_runs/image_log/train/*.png`: shows training progress.
- `training/mini_canny_eval/*sample*.png`: shows final qualitative behavior.
- A short ablation table with `control_strength`, `guidance_scale`, and Canny
  thresholds.

Good discussion points:

- This is intentionally a tiny overfit task, so the success criterion is control
  alignment and pipeline correctness, not generalization.
- Locking SD protects the base generator while the ControlNet branch learns the
  conditioning signal.
- Re-running Canny on generated images gives a simple control-consistency metric
  for the next evaluation step.
- For video generation, the same idea can be extended from one Canny map to a
  sequence of edge maps, with an additional temporal consistency metric.
