import torch
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


class PointMSE(MultiHorizonMetric):
    """Unreduced squared error for a one-output multi-horizon point head."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (y_pred[..., 0] - target) ** 2

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred[..., 0]
