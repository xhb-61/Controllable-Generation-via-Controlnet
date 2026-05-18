import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class MiniCannyDataset(Dataset):
    def __init__(self, data_root="./training/mini_canny", prompt_file="prompt.json"):
        self.data_root = Path(data_root)
        prompt_path = self.data_root / prompt_file

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Missing {prompt_path}. Run scripts/prepare_mini_canny_dataset.py first."
            )

        self.data = []
        with prompt_path.open("rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        if not self.data:
            raise ValueError(f"No training records found in {prompt_path}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_path = self.data_root / item["source"]
        target_path = self.data_root / item["target"]

        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)

        if source is None:
            raise FileNotFoundError(f"Could not read source image: {source_path}")
        if target is None:
            raise FileNotFoundError(f"Could not read target image: {target_path}")

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=item["prompt"], hint=source)
