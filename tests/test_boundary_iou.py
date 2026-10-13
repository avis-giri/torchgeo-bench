"""Contracts for the opt-in Boundary IoU segmentation metric."""

import math

import pytest
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_jaccard_index

from torchgeo_bench.boundary_iou import (
    DEFAULT_DILATION_RATIO,
    MulticlassBoundaryIoU,
    boundary_region,
    resolve_boundary_width,
)

NUM_CLASSES = 2


def square_mask(size: int, top: int, left: int, side: int) -> torch.Tensor:
    """One image with a filled square of class 1 on a class 0 ground."""
    mask = torch.zeros(1, size, size, dtype=torch.long)
    mask[0, top : top + side, left : left + side] = 1
    return mask


def score(target: torch.Tensor, preds: torch.Tensor, **kwargs: object) -> float:
    """Mean Boundary IoU for one batch."""
    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES, **kwargs)
    metric.update(preds, target)
    return metric.compute_mean()


def test_resolve_boundary_width_scales_with_the_diagonal() -> None:
    assert resolve_boundary_width(100, 100, 0.02) == round(0.02 * math.sqrt(2) * 100)
    assert resolve_boundary_width(256, 256, DEFAULT_DILATION_RATIO) == 7


def test_resolve_boundary_width_never_falls_below_one_pixel() -> None:
    """A band of zero pixels would make every score undefined."""
    assert resolve_boundary_width(8, 8, 0.001) == 1


def test_boundary_region_keeps_only_the_rim_of_a_thick_region() -> None:
    binary = torch.zeros(1, 9, 9, dtype=torch.bool)
    binary[0, 2:8, 2:8] = True

    band = boundary_region(binary, width=1)

    assert band[0, 2, 2]
    assert band[0, 7, 7]
    assert not band[0, 4, 4]
    assert band.sum() == binary.sum() - 4**2


def test_boundary_region_treats_the_image_edge_as_background() -> None:
    """A region running off the edge has a contour there, as in the reference."""
    binary = torch.ones(1, 5, 5, dtype=torch.bool)

    band = boundary_region(binary, width=1)

    assert band[0, 0, 0]
    assert not band[0, 2, 2]


def test_separable_erosion_matches_a_single_two_dimensional_pass() -> None:
    """The two 1-D max-pool passes are an optimisation, not a different metric."""
    torch.manual_seed(0)
    binary = torch.rand(3, 32, 32) > 0.4
    width = 3

    padded = F.pad(binary.unsqueeze(1).float(), (width,) * 4, value=0.0)
    direct = -F.max_pool2d(-padded, kernel_size=2 * width + 1, stride=1)
    expected = binary & (direct.squeeze(1) < 0.5)

    assert torch.equal(boundary_region(binary, width), expected)


def test_identical_masks_score_one() -> None:
    mask = square_mask(32, 8, 8, 16)

    assert score(mask, mask) == pytest.approx(1.0)


def test_a_class_predicted_somewhere_else_entirely_scores_zero() -> None:
    """Scored per class, so the shared background cannot rescue a misplaced region."""
    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    metric.update(square_mask(64, 48, 48, 12), square_mask(64, 2, 2, 12))

    assert metric.compute()[1].item() == pytest.approx(0.0)


def test_a_shifted_mask_costs_more_boundary_iou_than_ordinary_iou() -> None:
    """The reason the metric exists: contour error must not be forgiven."""
    size, side = 128, 64
    target = square_mask(size, 32, 32, side)
    preds = square_mask(size, 32, 40, side)

    boundary = score(target, preds)
    ordinary = multiclass_jaccard_index(
        preds, target, num_classes=NUM_CLASSES, average="macro"
    ).item()

    assert boundary < ordinary
    # An 8 px shift of a 64 px square keeps most of the interior but little of the rim.
    assert ordinary > 0.7
    assert boundary < 0.5


def test_boundary_iou_falls_as_the_shift_grows() -> None:
    size, side = 128, 64
    target = square_mask(size, 32, 32, side)
    scores = [score(target, square_mask(size, 32, 32 + shift, side)) for shift in (0, 4, 8, 16)]

    assert scores == sorted(scores, reverse=True)


def test_ignored_pixels_never_reach_the_score() -> None:
    """Corrupting only ignored pixels must leave the number alone."""
    target = square_mask(48, 12, 12, 24)
    preds = square_mask(48, 12, 14, 24)
    clean = score(target, preds, ignore_index=255)

    ignored = target.clone()
    ignored[0, 40:, :] = 255
    wrong = preds.clone()
    wrong[0, 40:, :] = 1

    assert score(ignored, wrong, ignore_index=255) == pytest.approx(
        score(ignored, preds, ignore_index=255)
    )
    assert clean != pytest.approx(score(ignored, preds, ignore_index=255))


def test_an_absent_class_is_nan_and_leaves_the_mean_alone() -> None:
    """Adding a class nobody predicts must not drag the mean toward zero."""
    target = square_mask(48, 12, 12, 24)
    preds = square_mask(48, 12, 14, 24)

    two = MulticlassBoundaryIoU(num_classes=2)
    two.update(preds, target)
    three = MulticlassBoundaryIoU(num_classes=3)
    three.update(preds, target)

    per_class = three.compute()
    assert math.isnan(per_class[2].item())
    assert three.compute_mean() == pytest.approx(two.compute_mean())


def test_an_empty_evaluation_is_nan_rather_than_zero() -> None:
    assert math.isnan(MulticlassBoundaryIoU(num_classes=2).compute_mean())


def test_batches_accumulate_over_the_dataset_not_per_image() -> None:
    """One total over every image, never the mean of per-image scores."""
    first = (square_mask(48, 4, 4, 16), square_mask(48, 4, 8, 16))
    second = (square_mask(48, 24, 24, 16), square_mask(48, 24, 24, 16))

    streamed = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    for target, preds in (first, second):
        streamed.update(preds, target)

    pooled = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    pooled.update(
        torch.cat([first[1], second[1]]),
        torch.cat([first[0], second[0]]),
    )

    assert streamed.compute_mean() == pytest.approx(pooled.compute_mean())
    assert streamed.compute_mean() != pytest.approx(
        (score(*first[::-1]) + score(*second[::-1])) / 2
    )


def test_reset_clears_the_totals() -> None:
    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    metric.update(square_mask(32, 4, 4, 8), square_mask(32, 4, 4, 8))
    metric.reset()

    assert math.isnan(metric.compute_mean())


def test_logits_and_labels_give_the_same_score() -> None:
    target = square_mask(48, 12, 12, 20)
    labels = square_mask(48, 12, 14, 20)
    logits = F.one_hot(labels, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()

    assert score(target, logits) == pytest.approx(score(target, labels))


def test_the_resolved_width_is_recorded_for_reporting() -> None:
    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES, dilation_ratio=0.02)
    assert metric.boundary_width is None

    metric.update(square_mask(256, 8, 8, 32), square_mask(256, 8, 8, 32))

    assert metric.boundary_width == resolve_boundary_width(256, 256, 0.02)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_classes": 0}, "num_classes"),
        ({"num_classes": 2, "dilation_ratio": 0.0}, "dilation_ratio"),
        ({"num_classes": 2, "dilation_ratio": 1.5}, "dilation_ratio"),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MulticlassBoundaryIoU(**kwargs)


def test_mismatched_shapes_are_rejected() -> None:
    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    with pytest.raises(ValueError, match="same pixels"):
        metric.update(
            torch.zeros(1, 8, 8, dtype=torch.long), torch.zeros(1, 4, 4, dtype=torch.long)
        )


def test_unbatched_labels_are_rejected() -> None:
    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    with pytest.raises(ValueError, match=r"\(N, H, W\)"):
        metric.update(torch.zeros(8, 8, dtype=torch.long), torch.zeros(8, 8, dtype=torch.long))


def test_boundary_region_rejects_a_zero_width() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        boundary_region(torch.zeros(1, 4, 4, dtype=torch.bool), width=0)
