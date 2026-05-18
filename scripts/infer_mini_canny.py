import argparse
import json
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference with a mini Canny checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./training/mini_canny")
    parser.add_argument("--model-config", default="./models/cldm_v15.yaml")
    parser.add_argument("--output-dir", default="./training/mini_canny_eval")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--negative-prompt", default="lowres, bad anatomy, worst quality, low quality")
    return parser.parse_args()


def load_records(data_root):
    records = []
    with (data_root / "prompt.json").open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    seed_everything(args.seed)
    model = create_model(args.model_config).cpu()
    missing, unexpected = model.load_state_dict(
        load_state_dict(args.checkpoint, location="cpu"),
        strict=False,
    )
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    records = load_records(data_root)
    if args.limit > 0:
        records = records[:args.limit]

    for index, item in enumerate(records):
        control_bgr = cv2.imread(str(data_root / item["source"]), cv2.IMREAD_COLOR)
        target_bgr = cv2.imread(str(data_root / item["target"]), cv2.IMREAD_COLOR)
        if control_bgr is None:
            raise FileNotFoundError(data_root / item["source"])
        if target_bgr is None:
            raise FileNotFoundError(data_root / item["target"])

        control_rgb = cv2.cvtColor(control_bgr, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = control_rgb.shape

        control = torch.from_numpy(control_rgb.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(args.num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([item["prompt"]] * args.num_samples)],
        }
        un_cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([args.negative_prompt] * args.num_samples)],
        }
        shape = (4, height // 8, width // 8)
        model.control_scales = [args.strength] * 13

        with torch.no_grad():
            samples, _ = sampler.sample(
                args.steps,
                args.num_samples,
                shape,
                cond,
                verbose=False,
                eta=args.eta,
                unconditional_guidance_scale=args.scale,
                unconditional_conditioning=un_cond,
            )
            decoded = model.decode_first_stage(samples)
            decoded = (
                einops.rearrange(decoded, "b c h w -> b h w c") * 127.5 + 127.5
            ).cpu().numpy().clip(0, 255).astype(np.uint8)

        stem = Path(item["target"]).stem
        save_rgb(output_dir / f"{index:03d}_{stem}_control.png", control_rgb)
        save_rgb(output_dir / f"{index:03d}_{stem}_target.png", target_rgb)
        for sample_index, image in enumerate(decoded):
            save_rgb(output_dir / f"{index:03d}_{stem}_sample_{sample_index:02d}.png", image)

    print(f"Wrote inference results to {output_dir}")


if __name__ == "__main__":
    main()
