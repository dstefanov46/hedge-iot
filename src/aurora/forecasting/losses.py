import torch
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


class PointMSE(MultiHorizonMetric):
    """Unreduced squared error for a one-output multi-horizon point head."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (y_pred[..., 0] - target) ** 2

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred[..., 0]


class PointMSEHeavy(MultiHorizonMetric):
    """MSE-heavy robust point loss: 75% MSE and 25% Huber."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = y_pred[..., 0] - target
        mse = error.square()
        huber = torch.nn.functional.huber_loss(error, torch.zeros_like(error), reduction="none")
        return 0.75 * mse + 0.25 * huber

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred[..., 0]


class PointBalancedHuberMSE(MultiHorizonMetric):
    """Balanced point loss: 50% Huber and 50% MSE."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = y_pred[..., 0] - target
        mse = error.square()
        huber = torch.nn.functional.huber_loss(error, torch.zeros_like(error), reduction="none")
        return 0.5 * mse + 0.5 * huber

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred[..., 0]


class PointMAEHeavy(MultiHorizonMetric):
    """MAE-heavy point loss: 75% absolute error and 25% MSE."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = y_pred[..., 0] - target
        return 0.75 * error.abs() + 0.25 * error.square()

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred[..., 0]


class PointMAE(MultiHorizonMetric):
    """Pure mean absolute error point loss."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (y_pred[..., 0] - target).abs()

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred[..., 0]
