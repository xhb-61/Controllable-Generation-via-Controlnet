import argparse
import json
import re
import shutil
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a tiny ControlNet Canny-to-image dataset from local images."
    )
    parser.add_argument("--manifest", default="configs/mini_canny_prompts.jsonl")
    parser.add_argument("--output-root", default="training/mini_canny")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--low-threshold", type=int, default=100)
    parser.add_argument("--high-threshold", type=int, default=200)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path):
    records = []
    with Path(path).open("rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "image" not in record or "prompt" not in record:
                raise ValueError(f"{path}:{line_number} must contain image and prompt fields.")
            records.append(record)
    return records


def sanitize_id(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "sample"


def resize_center_crop(image, resolution):
    height, width = image.shape[:2]
    scale = resolution / min(height, width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    x0 = max((new_width - resolution) // 2, 0)
    y0 = max((new_height - resolution) // 2, 0)
    return resized[y0:y0 + resolution, x0:x0 + resolution]


def write_preview(output_root, entries, preview_count):
    if preview_count <= 0:
        return

    rows = []
    for item in entries[:preview_count]:
        source = cv2.imread(str(output_root / item["source"]), cv2.IMREAD_COLOR)
        target = cv2.imread(str(output_root / item["target"]), cv2.IMREAD_COLOR)
        if source is not None and target is not None:
            rows.append(cv2.hconcat([source, target]))

    if rows:
        preview = cv2.vconcat(rows)
        cv2.imwrite(str(output_root / "preview.png"), preview)


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)

    if args.resolution % 64 != 0:
        raise ValueError("--resolution should be a multiple of 64 for Stable Diffusion.")

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(output_root)

    (output_root / "source").mkdir(parents=True, exist_ok=True)
    (output_root / "target").mkdir(parents=True, exist_ok=True)

    records = read_manifest(manifest_path)
    entries = []
    for index, record in enumerate(records):
        image_path = Path(record["image"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image from manifest: {image_path}")

        sample_id = sanitize_id(record.get("id") or image_path.stem)
        sample_id = f"{index:03d}_{sample_id}"
        target = resize_center_crop(image, args.resolution)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, args.low_threshold, args.high_threshold)
        source = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        source_rel = f"source/{sample_id}.png"
        target_rel = f"target/{sample_id}.png"
        cv2.imwrite(str(output_root / source_rel), source)
        cv2.imwrite(str(output_root / target_rel), target)
        entries.append(
            {
                "source": source_rel,
                "target": target_rel,
                "prompt": record["prompt"],
            }
        )

    with (output_root / "prompt.json").open("wt", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    write_preview(output_root, entries, args.preview_count)
    print(f"Wrote {len(entries)} samples to {output_root}")
    print(f"Prompt file: {output_root / 'prompt.json'}")
    print(f"Preview: {output_root / 'preview.png'}")


if __name__ == "__main__":
    main()
