"""Geometry, laterality and resampling checks.

These run without any DICOM on disk, because every assertion here is about arithmetic on
header values. The laterality tests are the important ones: a wrong side flips five of the
twelve targets onto an axis the model cannot observe, and nothing about it raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from rsnaknee.dicom_io import (
    MIDLINE_BAND_MM,
    Slice,
    normalise_laterality,
    order_slices,
    resample_to_fov,
    study_side,
    to_uint8,
)


def make_slice(x: float, z: float = 0.0, plane: str = "sagittal", rows: int = 256, cols: int = 256) -> Slice:
    """A slice whose image centre sits at patient x ≈ `x`.

    Orientation is chosen per plane so the centre calculation exercises the real direction
    cosines rather than an identity shortcut.
    """
    if plane == "sagittal":  # rows run A->P, cols run S->I; normal is +x
        orientation = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
        position = np.array([x, -60.0, 60.0])
    elif plane == "coronal":  # normal is +y
        orientation = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        position = np.array([x - 60.0, z, 60.0])
    else:  # axial, normal is +z
        orientation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        position = np.array([x - 60.0, -60.0, z])
    return Slice(
        path=None,
        position=position,
        orientation=orientation,
        spacing=np.array([0.469, 0.469]),
        rows=rows,
        cols=cols,
    )


class TestOrdering:
    def test_slices_sort_along_the_normal(self):
        shuffled = [make_slice(x=v) for v in (5.0, -3.0, 11.0, 0.0)]
        offsets = [s.offset for s in order_slices(shuffled)]
        assert offsets == sorted(offsets)

    def test_ordering_is_independent_of_input_order(self):
        forward = [make_slice(x=v) for v in (-3.0, 0.0, 5.0, 11.0)]
        assert [s.offset for s in order_slices(forward)] == [
            s.offset for s in order_slices(list(reversed(forward)))
        ]


class TestLateralitySide:
    """+x points to the patient's left, so negative image-centre x is a right knee."""

    @pytest.mark.parametrize("plane", ["sagittal", "coronal", "axial"])
    def test_positive_x_is_left(self, plane):
        assert study_side([make_slice(x=80.0, plane=plane)]) == "left"

    @pytest.mark.parametrize("plane", ["sagittal", "coronal", "axial"])
    def test_negative_x_is_right(self, plane):
        assert study_side([make_slice(x=-80.0, plane=plane)]) == "right"

    def test_near_midline_is_unknown_not_guessed(self):
        assert study_side([make_slice(x=MIDLINE_BAND_MM / 2)]) == "unknown"

    def test_empty_study_is_unknown(self):
        assert study_side([]) == "unknown"

    def test_median_resists_a_single_outlier(self):
        """One stray slice must not flip a study's side."""
        slices = [make_slice(x=v) for v in (80.0, 82.0, 79.0, -95.0)]
        assert study_side(slices) == "left"


class TestLateralityCorrection:
    def test_left_knee_is_untouched(self):
        stack = np.random.randint(0, 255, (8, 16, 16), dtype=np.uint8)
        for plane in ("sagittal", "coronal", "axial"):
            assert np.array_equal(normalise_laterality(stack, plane, "left"), stack)

    def test_unknown_side_is_untouched(self):
        """A guess inside the midline band is as likely to break a left knee as fix a right."""
        stack = np.random.randint(0, 255, (8, 16, 16), dtype=np.uint8)
        assert np.array_equal(normalise_laterality(stack, "coronal", "unknown"), stack)

    @pytest.mark.parametrize("plane", ["coronal", "axial"])
    def test_in_plane_axes_mirror_the_last_axis(self, plane):
        stack = np.random.randint(0, 255, (8, 16, 16), dtype=np.uint8)
        out = normalise_laterality(stack, plane, "right")
        assert np.array_equal(out, stack[:, :, ::-1])
        assert not np.array_equal(out, stack[::-1]), "must not reorder slices in-plane"

    def test_sagittal_reverses_slice_order_and_leaves_pixels_alone(self):
        """The assertion that catches the classic bug.

        Mirroring a sagittal slice is a no-op for laterality — the medial-lateral axis runs
        through the stack, not across the image — so the correction has to be a reversal.
        """
        stack = np.random.randint(0, 255, (8, 16, 16), dtype=np.uint8)
        out = normalise_laterality(stack, "sagittal", "right")

        assert np.array_equal(out, stack[::-1])
        assert not np.array_equal(out, stack[:, :, ::-1]), "sagittal must not mirror in-plane"
        for i, frame in enumerate(out):  # every slice unchanged, only their order differs
            assert np.array_equal(frame, stack[len(stack) - 1 - i])

    def test_correction_is_an_involution(self):
        """Applying it twice returns the original, in every plane."""
        stack = np.random.randint(0, 255, (8, 16, 16), dtype=np.uint8)
        for plane in ("sagittal", "coronal", "axial"):
            once = normalise_laterality(stack, plane, "right")
            assert np.array_equal(normalise_laterality(once, plane, "right"), stack)


class TestResampling:
    def test_output_is_always_the_requested_size(self):
        for rows, cols in ((256, 256), (320, 260), (128, 128)):
            image = np.random.randint(0, 255, (rows, cols), dtype=np.uint8)
            out = resample_to_fov(image, np.array([0.5, 0.5]), size=224)
            assert out.shape == (224, 224)

    def test_fixed_physical_scale_across_different_pixel_spacing(self):
        """A feature of a given millimetre size must occupy the same pixels at any spacing.

        This is the point of resampling to a field of view rather than resizing: two scanners
        with different zoom must present the joint at the same magnification.
        """
        size, fov = 224, 160.0
        results = []
        for spacing in (0.4, 0.5, 0.8):
            extent = int(round(fov / spacing))
            image = np.zeros((extent, extent), dtype=np.uint8)
            half = int(round(20.0 / spacing))  # a 40 mm square at the centre
            c = extent // 2
            image[c - half : c + half, c - half : c + half] = 255
            out = resample_to_fov(image, np.array([spacing, spacing]), size=size, fov_mm=fov)
            results.append((out > 127).sum() / out.size)
        assert max(results) - min(results) < 0.02, f"scale drifted across spacings: {results}"

    def test_smaller_acquisition_is_padded_not_stretched(self):
        image = np.full((100, 100), 200, dtype=np.uint8)
        out = resample_to_fov(image, np.array([1.0, 1.0]), size=224, fov_mm=160.0)
        assert out.shape == (224, 224)
        assert out[0, 0] == 0, "border should be padding, not stretched content"
        assert out[112, 112] == 200


class TestWindowing:
    def test_output_spans_the_byte_range(self):
        out = to_uint8(np.random.RandomState(0).normal(500, 100, (64, 64)).astype(np.float32))
        assert out.dtype == np.uint8
        assert out.min() == 0 and out.max() == 255

    def test_constant_image_does_not_divide_by_zero(self):
        out = to_uint8(np.full((32, 32), 7.0, dtype=np.float32))
        assert out.shape == (32, 32) and out.max() == 0

    def test_single_bright_artefact_does_not_crush_contrast(self):
        """Percentile windowing, not min/max — one hot pixel must not flatten the joint."""
        image = np.random.RandomState(0).uniform(100, 200, (64, 64)).astype(np.float32)
        image[0, 0] = 1e6
        out = to_uint8(image)
        assert out[1:, 1:].std() > 40, "contrast collapsed around the artefact"
