from __future__ import annotations

from typing import Optional, List, Union
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from adapts.icl.iclearner import ICLTrainer, ICLObject


class SundialICLTrainer(ICLTrainer):
    """
    ICLTrainer wrapper for HuggingFace timer-sundial models.

    - backbone.generate(y_hist_norm, max_new_tokens=output_token_lens[0]) 返回一个张量，
      需要 .mean(dim=1) 得到 (B, pred_len) 或 (B, H) 的预测序列。
    - AdaPTS 会按通道调用 FM：在这里负责把 (B, C, L) 拆成 C 个 (B,1,L)，分别预测。 
    """

    def __init__(
        self,
        sundial_name: str,
        n_features: int,
        forecast_horizon: int,
        device: str = "cpu",
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = torch.device(device)
        self.n_features = int(n_features)
        self.forecast_horizon = int(forecast_horizon)

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

    def eval(self):
        self.backbone.eval()
        return self

    def compute_statistics(self, **kwargs) -> ICLObject:
        # Sundial 这里只输出 point forecast（predictions / mean_arr）
        return ICLObject()

    @torch.no_grad()
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

    @torch.no_grad()
    def _predict_one_channel(
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

        # sundial 的 max_new_tokens 默认用 config.output_token_lens[0]
        # 但 AdaPTS 会传 prediction_horizon；这里做一个安全处理：
        default_pred_len = None
        if hasattr(self.backbone_config, "output_token_lens"):
            try:
                default_pred_len = int(self.backbone_config.output_token_lens[0])
            except Exception:
                default_pred_len = None

        max_new_tokens = prediction_horizon if default_pred_len is None else max(prediction_horizon, default_pred_len)

        pred = self.backbone.generate(
            y_hist_in,
            max_new_tokens=max_new_tokens,
        ).mean(dim=1)

        # 截断到 prediction_horizon
        if pred.shape[-1] < prediction_horizon:
            # 不够就右侧 padding
            pad = prediction_horizon - pred.shape[-1]
            pred = torch.nn.functional.pad(pred, (0, pad))
        pred = pred[..., :prediction_horizon]  # (B,H)

        return pred

    @torch.no_grad()
    def predict_long_horizon(self, prediction_horizon: int, **kwargs) -> List[ICLObject]:
        """
        返回 List[ICLObject]，长度 = n_features。
        每个 ICLObject.predictions: np.ndarray (B,1,H) float32
        """
        assert self._context is not None, "Call update_context(...) first"
        ctx = self._context  # (B,C,L)
        B, C, L = ctx.shape
        assert C == self.n_features

        results: List[ICLObject] = []
        for c in range(C):
            y_hist = ctx[:, c:c+1, :]  # (B,1,L)
            pred = self._predict_one_channel(y_hist, prediction_horizon)  # (B,H)
            pred = pred.unsqueeze(1)  # (B,1,H)

            pred_t = pred.to(torch.float32)  # 保持 torch.Tensor
            results.append(ICLObject(predictions=pred_t, mean_arr=pred_t))

        return results
