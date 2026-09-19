from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


class LanguageVLADataset(Dataset):
    """Isaac Lab dual-camera, state, language and action dataset."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        validation_episodes: int = 10,
        seed: int = 101,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(
                f"Unsupported split: {split}"
            )

        self.dataset_path = str(
            Path(dataset_path).resolve()
        )
        self.split = split
        self._h5 = None

        with h5py.File(self.dataset_path, "r") as h5:
            demos = sorted(
                h5["data"].keys(),
                key=lambda name: int(
                    name.split("_")[-1]
                ),
            )

            selected = demos
            labels = {h5['data'][name].attrs['language_instruction'] for name in demos}
            if labels != {'pick up the cube', 'move above the cube'}:
                raise ValueError('This fit check requires both instructions')
            if h5['data'].attrs.get('experiment_protocol') != 'fixed_state_fit_check_not_generalization':
                raise ValueError('Use the paired dataset produced by pack.py')

            self.samples = []

            for demo_name in selected:
                demo = h5["data"][demo_name]
                length = int(demo["actions"].shape[0])

                for step in range(length):
                    self.samples.append(
                        (demo_name, step)
                    )

            self.demo_names = selected

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split={split}"
            )

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.dataset_path,
                "r",
                swmr=True,
            )
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        h5 = getattr(self, "_h5", None)

        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass

    @staticmethod
    def _prepare_image(array) -> torch.Tensor:
        image = np.asarray(
            array,
            dtype=np.uint8,
        )

        tensor = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float()

        tensor = tensor / 255.0
        tensor = (
            tensor - IMAGE_MEAN
        ) / IMAGE_STD

        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        demo_name, step = self.samples[index]

        h5 = self._file()
        demo = h5["data"][demo_name]
        obs = demo["obs"]

        eef_pos = np.asarray(
            obs["eef_pos"][step],
            dtype=np.float32,
        )
        eef_quat = np.asarray(
            obs["eef_quat"][step],
            dtype=np.float32,
        )
        gripper_pos = np.asarray(
            obs["gripper_pos"][step],
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                eef_pos,
                eef_quat,
                gripper_pos,
            np.asarray([step / 1200.0], dtype=np.float32),
            ]
        ).astype(np.float32)

        language = np.asarray(
            demo["language_tokens"],
            dtype=np.int64,
        )

        action = np.asarray(
            demo["actions"][step],
            dtype=np.float32,
        )

        return {
            "table_image": self._prepare_image(
                obs["table_cam"][step]
            ),
            "wrist_image": self._prepare_image(
                obs["wrist_cam"][step]
            ),
            "state": torch.from_numpy(state),
            "language": torch.from_numpy(
                language.copy()
            ),
            "action": torch.from_numpy(action),
            "demo_name": demo_name,
            "step": step,
        }


def collate_language_vla(batch: list[dict]) -> dict:
    max_tokens = max(
        len(item["language"])
        for item in batch
    )

    language = torch.zeros(
        len(batch),
        max_tokens,
        dtype=torch.long,
    )

    for index, item in enumerate(batch):
        length = len(item["language"])
        language[index, :length] = (
            item["language"]
        )

    return {
        "table_image": torch.stack(
            [
                item["table_image"]
                for item in batch
            ]
        ),
        "wrist_image": torch.stack(
            [
                item["wrist_image"]
                for item in batch
            ]
        ),
        "state": torch.stack(
            [item["state"] for item in batch]
        ),
        "language": language,
        "action": torch.stack(
            [item["action"] for item in batch]
        ),
        "demo_name": [
            item["demo_name"]
            for item in batch
        ],
        "step": torch.tensor(
            [item["step"] for item in batch],
            dtype=torch.long,
        ),
    }
