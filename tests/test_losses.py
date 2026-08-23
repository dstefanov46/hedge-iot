import pytest
import torch

from aurora.forecasting.losses import (
    PointBalancedHuberMSE,
    PointMAE,
    PointMAEHeavy,
    PointMSE,
    PointMSEHeavy,
)


@pytest.mark.parametrize(
    ("loss_type", "expected"),
    [
        (PointMSE, [0.0, 4.0]),
        (PointMSEHeavy, [0.0, 3.375]),
        (PointBalancedHuberMSE, [0.0, 2.75]),
        (PointMAEHeavy, [0.0, 2.5]),
        (PointMAE, [0.0, 2.0]),
    ],
)
def test_point_loss_profiles(loss_type, expected) -> None:
    prediction = torch.tensor([[[0.0], [3.0]]])
    target = torch.tensor([[0.0, 1.0]])
    actual = loss_type().loss(prediction, target)
    assert actual.flatten().tolist() == pytest.approx(expected)


def test_point_loss_profiles_preserve_point_prediction() -> None:
    prediction = torch.tensor([[[0.2], [0.7]]])
    for loss_type in (PointMSE, PointMSEHeavy, PointBalancedHuberMSE, PointMAEHeavy, PointMAE):
        assert loss_type().to_prediction(prediction).flatten().tolist() == pytest.approx(
            [0.2, 0.7]
        )
