from __future__ import annotations

import copy
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class LanguageConditionedVLA(nn.Module):
    """Dual-camera language-conditioned action policy."""

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 7,
        vocab_size: int = 4096,
        language_dim: int = 64, pretrained_vision: bool = False,
    ) -> None:
        super().__init__()

        if action_dim != 7:
            raise ValueError(
                "This model expects 7D IK-Rel actions"
            )

        vision = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_vision else None)
        vision.fc = nn.Identity()
        self.vision = vision

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.token_embedding = nn.Embedding(
            vocab_size,
            language_dim,
            padding_idx=0,
        )

        self.language_encoder = nn.GRU(
            input_size=language_dim,
            hidden_size=language_dim,
            batch_first=True,
        )

        fusion_dim = (
            512
            + 512
            + 64
            + language_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.arm_head = nn.Linear(256, 6)
        self.gripper_head = nn.Linear(256, 1)

        self.pick_fusion = copy.deepcopy(self.fusion)
        self.pick_arm_head = copy.deepcopy(self.arm_head)
        self.pick_gripper_head = copy.deepcopy(self.gripper_head)
        self._pick_mode = False

    def set_instruction(self, instruction):
        if instruction not in ('pick up the cube', 'move above the cube'):
            raise ValueError('Unsupported instruction: ' + instruction)
        self._pick_mode = instruction == 'pick up the cube'

    def encode_language(
        self,
        language: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.token_embedding(language)

        lengths = (language != 0).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.language_encoder(packed)

        return hidden[-1]

    def forward(
        self,
        table_image: torch.Tensor,
        wrist_image: torch.Tensor,
        state: torch.Tensor,
        language: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = table_image.shape[0]

        combined_images = torch.cat(
            [table_image, wrist_image],
            dim=0,
        )

        combined_features = self.vision(
            combined_images
        )

        table_feature = combined_features[
            :batch_size
        ]
        wrist_feature = combined_features[
            batch_size:
        ]

        state_feature = self.state_encoder(
            state
        )
        language_feature = self.encode_language(
            language
        )

        fused = torch.cat(
            [
                table_feature,
                wrist_feature,
                state_feature,
                language_feature,
            ],
            dim=-1,
        )

        feature = (self.pick_fusion if self._pick_mode else self.fusion)(fused)

        arm_action = torch.tanh(
            (self.pick_arm_head if self._pick_mode else self.arm_head)(feature)
        )
        gripper_logit = (self.pick_gripper_head if self._pick_mode else self.gripper_head)(
            feature
        ).squeeze(-1)

        return {
            "arm_action": arm_action,
            "gripper_logit": gripper_logit,
            "language_feature": language_feature,
        }

    @staticmethod
    def build_action(
        output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        closed = (
            torch.sigmoid(
                output["gripper_logit"]
            )
            >= 0.5
        )

        gripper_action = torch.where(
            closed,
            torch.full_like(
                output["gripper_logit"],
                -1.0,
            ),
            torch.full_like(
                output["gripper_logit"],
                1.0,
            ),
        ).unsqueeze(-1)

        return torch.cat(
            [
                output["arm_action"],
                gripper_action,
            ],
            dim=-1,
        )
