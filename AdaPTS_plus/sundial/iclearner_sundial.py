from __future__ import annotations

from typing import Optional, List, Union
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from adapts.icl.iclearner import ICLTrainer, ICLObject


class SundialICLTrainer(ICLTrainer):
    """
    ICLTrainer wrapper for HuggingFace timer-sundial models (gradient-friendly).
    """

    def __init__(
        self,
        sundial_name: str,
        n_features: int,
        forecast_horizon: int,
        device: str = "cpu",
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        num_samples: int = 1,
        revin: bool = True,
    ):
        self.device = torch.device(device)
        self.n_features = int(n_features)
        self.forecast_horizon = int(forecast_horizon)
        self.num_samples = int(num_samples)
        self.revin = bool(revin)

        self.backbone_config = AutoConfig.from_pretrained(
            sundial_name, trust_remote_code=trust_remote_code
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            sundial_name,
            trust_remote_code=trust_remote_code,
            config=self.backbone_config,
            torch_dtype=torch_dtype,
        ).to(self.device)

        self._context: Optional[torch.Tensor] = None  # (B, C, L)

    def train(self):
        self.backbone.train()
        return self

    def eval(self):
        self.backbone.eval()
        return self

    def compute_statistics(self, **kwargs) -> ICLObject:
        return ICLObject()

    def update_context(
        self, time_series: Union[np.ndarray, torch.Tensor], **kwargs
    ) -> ICLObject:
        """
        AdaPTS 会传入 adapter.transform_torch(X_batch) 的输出：
          time_series: (B, n_components, context_length)
        """
        if isinstance(time_series, np.ndarray):
            ts = torch.tensor(time_series, dtype=torch.float32, device=self.device)
        else:
            ts = time_series.to(self.device)

        assert ts.dim() == 3, f"Expected (B,C,L), got {tuple(ts.shape)}"
        assert ts.shape[1] == self.n_features, f"C={ts.shape[1]} != n_features={self.n_features}"

        self._context = ts
        return ICLObject(time_series=None)

    def _resolve_max_output_length(self, prediction_horizon: int) -> int:
        H = int(prediction_horizon)
        if H <= 0:
            raise ValueError(f"prediction_horizon must be > 0, got {H}")
        return H

    def _predict_one_channel_forward(
        self,
        y_hist: torch.Tensor,
        prediction_horizon: int,
    ) -> torch.Tensor:
        """
        y_hist: (B, 1, L) or (B, L)
        return: (B, prediction_horizon)
        """
        if y_hist.dim() == 3:
            y_hist_in = y_hist.squeeze(1)  # (B,L)
        else:
            y_hist_in = y_hist  # (B,L)

        max_output_length = self._resolve_max_output_length(prediction_horizon)

        outputs = self.backbone(
            input_ids=y_hist_in,
            labels=None,
            return_dict=True,
            max_output_length=max_output_length,
            revin=self.revin,
            num_samples=self.num_samples,
        )

        pred = outputs.logits
        if pred.dim() == 3:
            pred = pred.mean(dim=1)  # (B,H)
        elif pred.dim() == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected prediction shape from forward: {tuple(pred.shape)}")

        # 保证长度 = prediction_horizon
        if pred.shape[-1] < prediction_horizon:
            pad = prediction_horizon - pred.shape[-1]
            pred = torch.nn.functional.pad(pred, (0, pad))
        pred = pred[..., :prediction_horizon]  # (B,H)

        return pred

    def predict_long_horizon(self, prediction_horizon: int, **kwargs) -> List[ICLObject]:
        """
        返回 List[ICLObject]，长度 = n_features。
        每个 ICLObject.predictions: torch.Tensor (B,1,H)
        """
        assert self._context is not None, "Call update_context(...) first"
        ctx = self._context  # (B,C,L)
        B, C, L = ctx.shape
        assert C == self.n_features

        results: List[ICLObject] = []
        for c in range(C):
            y_hist = ctx[:, c:c+1, :]  # (B,1,L)
            pred = self._predict_one_channel_forward(y_hist, prediction_horizon)  # (B,H)
            pred = pred.unsqueeze(1)  # (B,1,H)

            results.append(ICLObject(predictions=pred, mean_arr=pred))

        return results
