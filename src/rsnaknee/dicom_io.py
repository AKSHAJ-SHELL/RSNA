"""DICOM -> normalised uint8 slot stacks.

Four things happen here and three of them fail silently, which is why each one is a named
function with its own test rather than a step inside a loop.

**Slice ordering.** DICOM files in a directory are in no meaningful order. Slices are sorted by
their projection onto the slice normal, so the stack traverses the joint monotonically. Sorting
by `InstanceNumber` looks equivalent and is not: some vendors number backwards.

**Laterality.** Five of the twelve targets are named for a side — the two menisci, the two
tibiofemoral compartments, and the MCL. Medial and lateral are defined against the body's
midline, so which side of the *image* they fall on depends on which knee was scanned. The
`Laterality` tag is Type 2C and absent on about half of these studies, by whole vendor rather
than scattered, so trusting it silently declares every untagged right knee to be a left one.
The patient coordinate system supplies the answer instead: +x points to the patient's left, so
the sign of the image centre's x says which knee this is.

**The correction differs by plane.** Coronally and axially the medial-lateral axis lies in the
image plane, so mirroring the last axis maps one knee onto the other. Sagittally it does not:
each slice is unchanged by mirroring, and what differs is the order in which the stack crosses
the joint. Reversing the slice order is the correction there. Applying a flip to a sagittal
stack is a no-op that looks like it worked.

**Physical scale.** Resampling to a fixed millimetre field of view rather than a fixed pixel
count means a knee occupies the same fraction of the frame regardless of scanner or zoom. A
plain resize hands the encoder a different magnification per site, across 16 sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut

#: Field of view kept around the image centre, in millimetres. A knee joint is comfortably
#: inside 160 mm; going wider spends resolution on air and soft tissue outside the joint.
FOV_MM = 160.0

#: Studies whose image centre sits within this distance of the midline are left unresolved
#: rather than guessed — inside that band the sign of x is no better than a coin flip.
MIDLINE_BAND_MM = 20.0


@dataclass(frozen=True)
class Slice:
    path: Path
    position: np.ndarray  # ImagePositionPatient
    orientation: np.ndarray  # ImageOrientationPatient, 6 values
    spacing: np.ndarray  # PixelSpacing, [row, col]
    rows: int
    cols: int

    @property
    def normal(self) -> np.ndarray:
        """Slice normal — the cross product of the row and column direction cosines."""
        return np.cross(self.orientation[:3], self.orientation[3:])

    @property
    def offset(self) -> float:
        """Projection onto the normal. Sorting on this orders the stack through the joint."""
        return float(np.dot(self.position, self.normal))

    @property
    def centre(self) -> np.ndarray:
        """Patient-space coordinate of the image centre.

        The centre, not `ImagePositionPatient`, which is a corner half a field of view away —
        enough to flip the sign of x on a knee scanned near the midline.
        """
        row_cos, col_cos = self.orientation[:3], self.orientation[3:]
        return (
            self.position
            + row_cos * self.spacing[1] * (self.cols / 2.0)
            + col_cos * self.spacing[0] * (self.rows / 2.0)
        )


def read_header(path: Path) -> Slice | None:
    """Read geometry without decoding pixels. Returns None if the geometry is unusable."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return Slice(
            path=path,
            position=np.asarray(ds.ImagePositionPatient, dtype=float),
            orientation=np.asarray(ds.ImageOrientationPatient, dtype=float),
            spacing=np.asarray(ds.PixelSpacing, dtype=float),
            rows=int(ds.Rows),
            cols=int(ds.Columns),
        )
    except Exception:
        # A slice missing geometry cannot be placed in the stack. Dropping it is correct;
        # substituting defaults would put it at an arbitrary depth in the joint.
        return None


def order_slices(slices: list[Slice]) -> list[Slice]:
    return sorted(slices, key=lambda s: s.offset)


def study_side(slices: list[Slice]) -> str:
    """'left', 'right' or 'unknown', from the patient coordinate system.

    The median over the study is used rather than any single slice: an axial stack sweeps
    through z, not x, but a sagittal stack genuinely spans x, and the median is stable against
    both while a first-slice reading is not.
    """
    if not slices:
        return "unknown"
    centre_x = float(np.median([s.centre[0] for s in slices]))
    if abs(centre_x) < MIDLINE_BAND_MM:
        return "unknown"
    return "right" if centre_x < 0 else "left"


def decode(path: Path) -> np.ndarray | None:
    """Decode one slice to float32, applying the modality LUT. None if it will not decode."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        return apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        return None


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Percentile-window to uint8.

    MRI has no absolute intensity scale — the same tissue takes different raw values on
    different scanners and sequences — so windowing is per-slice against its own percentiles.
    The 1st/99th rather than min/max, because a single bright artefact would otherwise
    compress the entire joint into a few levels.
    """
    lo, hi = np.percentile(image, (1.0, 99.0))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def resample_to_fov(image: np.ndarray, spacing: np.ndarray, size: int, fov_mm: float = FOV_MM) -> np.ndarray:
    """Centre-crop/pad to a fixed millimetre field of view, then resize to `size`.

    Physical scale is fixed first and pixel count second, so the same anatomy lands at the same
    magnification across 16 sites with different zoom and matrix conventions.
    """
    import cv2

    rows_mm, cols_mm = fov_mm / spacing[0], fov_mm / spacing[1]
    want_r, want_c = int(round(rows_mm)), int(round(cols_mm))
    r, c = image.shape

    # Crop toward the centre, then pad if the acquisition was smaller than the requested FOV.
    r0, c0 = max(0, (r - want_r) // 2), max(0, (c - want_c) // 2)
    cropped = image[r0 : r0 + want_r, c0 : c0 + want_c]
    pad_r, pad_c = max(0, want_r - cropped.shape[0]), max(0, want_c - cropped.shape[1])
    if pad_r or pad_c:
        cropped = np.pad(
            cropped,
            ((pad_r // 2, pad_r - pad_r // 2), (pad_c // 2, pad_c - pad_c // 2)),
            mode="constant",
        )
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def normalise_laterality(stack: np.ndarray, plane: str, side: str) -> np.ndarray:
    """Map a right knee onto the left-knee convention.

    Sagittal is the case that catches people: mirroring a sagittal slice leaves medial and
    lateral exactly where they were, because the medial-lateral axis runs *through* the stack.
    The slice order is what encodes the side there.

    'unknown' is left untouched on purpose — a guess inside the midline band is as likely to
    mirror a left knee as to correct a right one.
    """
    if side != "right":
        return stack
    if plane.lower() == "sagittal":
        return stack[::-1].copy()
    return stack[:, :, ::-1].copy()


def load_series(
    paths: list[Path],
    plane: str,
    side: str,
    n_slices: int,
    image_size: int,
) -> np.ndarray | None:
    """Decode a series into a `(n_slices, image_size, image_size)` uint8 stack.

    A centred band is kept: the joint is in the middle of the acquisition and the outer slices
    are largely soft tissue and air. Series thinner than the band repeat their end slices
    rather than pad with zeros, which would teach the encoder that black is anatomy.
    """
    headers = [h for h in (read_header(p) for p in paths) if h is not None]
    if not headers:
        return None
    ordered = order_slices(headers)

    total = len(ordered)
    if total >= n_slices:
        start = (total - n_slices) // 2
        chosen = ordered[start : start + n_slices]
    else:
        chosen = ordered

    frames = []
    for header in chosen:
        pixels = decode(header.path)
        if pixels is None:
            continue
        frames.append(resample_to_fov(to_uint8(pixels), header.spacing, image_size))
    if not frames:
        return None

    stack = np.stack(frames)
    if len(stack) < n_slices:  # repeat edges to reach the band
        deficit = n_slices - len(stack)
        stack = np.concatenate(
            [stack, np.repeat(stack[-1:], deficit, axis=0)], axis=0
        )
    return normalise_laterality(stack, plane, side)
