"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

import hashlib
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.qwen_compat import Qwen2TokenizerFast
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.inference_mode()
    def predict_action(
        self, image: Union[Img, List[Img]], instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer

        # ── DEBUG: image identity ──────────────────────────────────────────────
        img_arr = np.asarray(image)
        img_hash = hashlib.md5(img_arr.tobytes()).hexdigest()[:8]
        print(f"[DBG] image  shape={img_arr.shape}  dtype={img_arr.dtype}  "
              f"mean={img_arr.mean():.2f}  std={img_arr.std():.2f}  hash={img_hash}")

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            # do nothing here. I think...
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # ── DEBUG: pixel_values after transform ───────────────────────────────
        _pv = pixel_values if isinstance(pixel_values, torch.Tensor) else next(iter(pixel_values.values()))
        print(f"[DBG] pixel_values  shape={tuple(_pv.shape)}  dtype={_pv.dtype}  device={_pv.device}  "
              f"mean={_pv.float().mean().item():.4f}  std={_pv.float().std().item():.4f}  "
              f"min={_pv.float().min().item():.4f}  max={_pv.float().max().item():.4f}")

        # ── DEBUG: pixel_values dict details ─────────────────────────────────
        _bb_dtype = next(self.vision_backbone.parameters()).dtype
        if isinstance(pixel_values, dict):
            print(f"[DBG] pixel_values keys={list(pixel_values.keys())}  backbone_dtype={_bb_dtype}")
            for k, v in pixel_values.items():
                print(f"[DBG]   [{k}]  shape={tuple(v.shape)}  dtype={v.dtype}  "
                      f"mean={v.float().mean().item():.4f}  std={v.float().std().item():.4f}")
        else:
            print(f"[DBG] pixel_values  shape={tuple(pixel_values.shape)}  dtype={pixel_values.dtype}  backbone_dtype={_bb_dtype}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        autocast_enabled = self.enable_mixed_precision_training and self.device.type == "cuda"
        print(f"[DBG] device={self.device}  autocast_enabled={autocast_enabled}  autocast_dtype={autocast_dtype}  "
              f"model_dtype={next(self.parameters()).dtype}")

        # Cast pixel_values to match model dtype (important on MPS where autocast is disabled)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to(dtype=autocast_dtype) for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to(dtype=autocast_dtype)
        print(f"[DBG] pixel_values cast to dtype={autocast_dtype}")

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs
            )
            # fmt: on

        # ── DEBUG: logit entropy at each generation step ──────────────────────
        if hasattr(generated_ids, "scores") and generated_ids.scores:
            for step_i, score in enumerate(generated_ids.scores):
                probs = torch.softmax(score[0].float(), dim=-1)
                top_prob, top_tok = probs.max(dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum().item()
                print(f"[DBG] gen step {step_i}  top_tok={top_tok.item()}  top_prob={top_prob.item():.4f}  entropy={entropy:.4f}")
        generated_ids = generated_ids.sequences

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        print("PREDICTED ACTION TOKENS:", predicted_action_token_ids.cpu().numpy())
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
