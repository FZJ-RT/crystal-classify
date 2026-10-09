#!/usr/bin/env python3
"""
crystal_classify.py
===================
Classification of CaCO3 / (Ba,Ca)CO3 crystals on large optical-microscopy images.

Classes:
  * "vaterite"  - spherulites: near-circular outline + radial fibrous internal texture
  * "other"     - rhombohedral calcite, disordered dumbbells, dumbbell spherulites
                  of the Ba-Ca carbonate solid solution
  * "cluster"   - multiple crystals grown together that could not be confidently
                  resolved into individual crystals (touching aggregate with poor
                  post-split solidity/circularity, or a region too large to be a
                  single crystal - see --cluster-diam-px)
  * "discarded" - objects cut by the image border (outline/size unreliable)

Objects smaller than --min-diam are treated as dust/noise and dropped before
any classification (not written to the output at all).

Designed for images > 1 GB:
  1. A downsampled overview is built tile-by-tile (TIFF: via zarr chunks,
     never loading the full image).
  2. Candidate objects are detected on the overview.
  3. For every candidate a full-resolution window is read from disk and
     fine segmentation + feature extraction is done there.
  4. Output: annotated overview PNG/JPG (+ optional full-resolution
     annotated TIFF) and a CSV with one row per crystal.

Usage:
    python crystal_classify.py INPUT_IMAGE [options]

Important options (see --help for all):
    --pixel-size 0.1         micrometres per pixel (full resolution)
    --min-diam 40            minimum object diameter in px (full res)
    --outdir results
    --annotate-fullres       also write a full-resolution annotated TIFF
    --overview-width 6000    width of the working overview in px
"""

import argparse
import glob
import math
import os
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops, label as sk_label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
#  Large-image access layer
# --------------------------------------------------------------------------- #
class BigImage:
    """Random window access to an image that may be far larger than RAM.

    TIFF  -> chunked access through tifffile/zarr (nothing fully loaded)
    other -> loaded once with cv2 (fine for "small" multi-100MB files,
             will warn if it is going to be painful).
    """

    def __init__(self, path):
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        self._zarr = None
        self._arr = None
        if ext in (".tif", ".tiff"):
            import tifffile
            try:                                   # 1) chunked zarr access
                import zarr
                store = tifffile.imread(path, aszarr=True)
                z = zarr.open(store, mode="r")
                if hasattr(z, "keys") and not hasattr(z, "shape"):
                    z = z[sorted(z.keys())[0]]
                self._zarr = z
                self.shape = z.shape[:2]
                return
            except Exception as e:
                print(f"[info] zarr access unavailable ({e}); "
                      "trying memmap", file=sys.stderr)
            try:                                   # 2) memmap (uncompressed)
                self._zarr = tifffile.memmap(path, mode="r")
                self.shape = self._zarr.shape[:2]
                return
            except Exception as e:
                print(f"[info] memmap unavailable ({e}); loading fully. "
                      "For >1 GB images save as UNCOMPRESSED (Big)TIFF, or "
                      "install a tifffile-compatible zarr version.",
                      file=sys.stderr)
            self._arr = tifffile.imread(path)
            self.shape = self._arr.shape[:2]
        else:
            if os.path.getsize(path) > 1.5e9:
                print("[warn] non-TIFF file >1.5 GB will be loaded fully into "
                      "RAM; convert to TIFF for tiled access.", file=sys.stderr)
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise IOError(f"cannot read {path}")
            self._arr = arr
            self.shape = arr.shape[:2]

    def window(self, y0, y1, x0, x1):
        """Return a grayscale uint8 window [y0:y1, x0:x1]."""
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(self.shape[0], y1), min(self.shape[1], x1)
        if self._zarr is not None:
            w = np.asarray(self._zarr[y0:y1, x0:x1])
        else:
            w = self._arr[y0:y1, x0:x1]
        return to_gray_u8(w)


def to_gray_u8(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3].astype(np.uint8) if a.dtype == np.uint8
                         else a[..., :3], cv2.COLOR_BGR2GRAY)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = np.percentile(a, (0.1, 99.9))
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return a


def build_overview(big, target_w):
    """Tile-wise downsampling -> (overview_u8, scale_factor)."""
    H, W = big.shape
    scale = max(1, int(math.ceil(W / target_w)))
    oh, ow = H // scale, W // scale
    out = np.zeros((oh, ow), np.uint8)
    band = max(scale * 64, 2048) // scale * scale          # row band height
    for y0 in range(0, oh * scale, band):
        y1 = min(y0 + band, oh * scale)
        strip = big.window(y0, y1, 0, ow * scale)
        out[y0 // scale:y1 // scale] = cv2.resize(
            strip, (ow, (y1 - y0) // scale), interpolation=cv2.INTER_AREA)
    return out, scale


# --------------------------------------------------------------------------- #
#  Segmentation
# --------------------------------------------------------------------------- #
def flatten_background(gray, sigma_px):
    """Remove uneven illumination: gray - large-scale background + 128."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigma_px)
    flat = np.clip(gray.astype(np.int16) - bg.astype(np.int16) + 128,
                   0, 255).astype(np.uint8)
    return flat


def foreground_mask(gray, min_diam):
    """Binary mask of crystal candidates on a flat-background image.

    Crystals show up as dark rims + textured interiors on a flat field, so we
    combine (a) deviation from the background level and (b) local texture.
    """
    flat = flatten_background(gray, sigma_px=max(min_diam, 25))
    dev = cv2.absdiff(flat, np.full_like(flat, 128))

    # local std-dev (texture) on a small window
    f = flat.astype(np.float32)
    k = 7
    mu = cv2.blur(f, (k, k))
    mu2 = cv2.blur(f * f, (k, k))
    std = cv2.sqrt(cv2.max(mu2 - mu * mu, 0)).astype(np.uint8)

    score = cv2.max(dev, std)
    score = cv2.GaussianBlur(score, (0, 0), 2)
    # noise level from the darkest (most background-like) half of the score
    # distribution, so the estimate is valid even if objects dominate the view
    lowhalf = score[score <= np.percentile(score, 50)].astype(np.float32)
    noise = lowhalf.mean() + 1.4826 * np.median(np.abs(lowhalf - np.median(lowhalf)))
    otsu, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(8.0, min(4.0 * max(noise, 1.0), float(otsu)))
    mask = (score > thr).astype(np.uint8)

    r = max(2, min_diam // 12)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    return mask


def split_touching(mask, min_diam, dumbbell_merge_waist=0.80):
    """Watershed on the distance transform; returns (labels, touching, regions).

    `regions` is the regionprops(labels) list, computed once here and handed
    back so callers don't need to recompute it.

    A dumbbell crystal (two lobes + a real neck - a single-crystal habit, see
    module docstring) produces the same "two distance-transform peaks with a
    saddle between them" signature as two genuinely separate crystals just
    touching, so watershed alone can't tell them apart and would otherwise
    split one dumbbell into two counted crystals. Before finalizing a 2-way
    split, we re-test the *original, unsplit* blob's waist_ratio - checked
    against real examples, counterintuitively a genuine single dumbbell has
    the HIGHER (gentler) waist_ratio of the two: a solid crystal can't grow
    an infinitely sharp reentrant notch, so its neck stays relatively broad,
    while two independent crystals merely touching pinch down to a much
    sharper, narrower contact. So a high whole-blob waist_ratio here means
    "one continuous dumbbell" and the split is undone (merged back to one
    region, to be counted - and later shape-classified - as a single
    crystal); a low one means "two separate crystals" and the split stands.
    This is deliberately a separate, independently-tunable parameter from
    classify()'s --min-waist dumbbell veto, which runs the other direction
    (low waist_ratio -> flag as dumbbell) on an already-final single region.
    """
    dist = ndi.distance_transform_edt(mask)
    min_sep = max(3, int(min_diam * 0.45))
    peaks = peak_local_max(dist, min_distance=min_sep,
                           threshold_abs=max(2.0, min_diam * 0.15),
                           labels=mask.astype(bool), exclude_border=False)
    markers = np.zeros_like(mask, np.int32)
    for i, (py, px) in enumerate(peaks, 1):
        markers[py, px] = i
    if markers.max() == 0:
        labels = sk_label(mask)
        return labels, {}, regionprops(labels)
    labels = watershed(-dist, markers, mask=mask.astype(bool))
    # how many seeds did each original connected blob contain?
    blobs = sk_label(mask)
    seeds_per_blob, peak_blob = {}, {}
    for i, (py, px) in enumerate(peaks, 1):
        b = blobs[py, px]
        seeds_per_blob[b] = seeds_per_blob.get(b, 0) + 1
        peak_blob[i] = b

    for b, n_seeds in list(seeds_per_blob.items()):
        if n_seeds != 2:
            continue
        wr = waist_ratio(blobs == b)
        if not np.isnan(wr) and wr > dumbbell_merge_waist:
            ids = [m for m, bb in peak_blob.items() if bb == b]
            labels[np.isin(labels, ids)] = ids[0]
            seeds_per_blob[b] = 1

    regions = regionprops(labels)
    touching = np.zeros(labels.max() + 1, bool)
    for reg in regions:
        y, x = map(int, reg.centroid)
        b = blobs[y, x] if mask[y, x] else 0
        if b and seeds_per_blob.get(b, 1) > 1:
            touching[reg.label] = True
    return labels, touching, regions


# --------------------------------------------------------------------------- #
#  Features
# --------------------------------------------------------------------------- #
def radial_texture_score(gray, region_mask, cx, cy, rim_frac=0.15):
    """How radially organised is the internal texture? (vaterite -> high)

    Spherulite fibres run along the radius.  We high-pass the interior (to
    kill the global bright-centre/dark-rim ramp), estimate the local texture
    orientation with a structure tensor, and measure how well the dominant
    local intensity-variation direction aligns with the radius:

        score = < cos^2(grad_angle - radial_angle) >   (coherence-weighted)

    Empirically (checked against confirmed spherulites) the intensity varies
    ALONG each fibre's length (growth banding) rather than across its width,
    so the dominant gradient direction itself tracks the fibre/radial
    direction directly - no perpendicular correction needed.

    Random texture -> ~0.5 ; radial fibres -> 0.65-0.95 ; facets -> variable.
    """
    d = 2 * math.sqrt(region_mask.sum() / math.pi)
    er = max(2, int(rim_frac * d))
    interior = cv2.erode(region_mask.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * er + 1, 2 * er + 1)))
    if interior.sum() < 50:
        return np.nan, np.nan

    f = gray.astype(np.float32)
    s_hp = max(1.5, d / 30.0)            # high-pass to remove radial ramp
    hp = f - cv2.GaussianBlur(f, (0, 0), s_hp)
    gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)
    s_st = max(1.5, d / 40.0)            # structure-tensor smoothing
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), s_st)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), s_st)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), s_st)

    ys, xs = np.nonzero(interior)
    jxx, jyy, jxy = Jxx[ys, xs], Jyy[ys, xs], Jxy[ys, xs]
    energy = jxx + jyy + 1e-9
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / energy

    # gradient orientation (double-angle form) - see docstring for why this
    # is compared to the radial angle directly, with no +90 deg correction
    two_theta_g = np.arctan2(2 * jxy, jxx - jyy)
    phi = np.arctan2(ys - cy, xs - cx)            # radial angle
    # cos^2(a-b) = 0.5*(1+cos(2a-2b))
    align = 0.5 * (1.0 + np.cos(two_theta_g - 2 * phi))

    w = coherence * energy
    if w.sum() <= 0:
        return np.nan, np.nan
    score = float(np.average(align, weights=w))
    texture = float(np.mean(np.hypot(gx[ys, xs], gy[ys, xs])))
    return score, texture


def waist_ratio(region_mask):
    """Dumbbell detector: ratio of the neck width to the lobe width along the
    major axis.  ~1 for convex shapes, <0.85 for dumbbells."""
    m = region_mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.nan
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return np.nan
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
    ang_r = math.radians(ang - 90)          # major axis direction
    ux, uy = math.cos(ang_r), math.sin(ang_r)
    ys, xs = np.nonzero(m)
    t = (xs - cx) * ux + (ys - cy) * uy     # coordinate along major axis
    L = t.max() - t.min()
    if L < 8:
        return np.nan
    nbins = 25
    h, edges = np.histogram(t, bins=nbins)
    widths = h / np.maximum((edges[1] - edges[0]), 1e-6)
    q = nbins // 4
    lobe = max(widths[:q].max(), widths[-q:].max())
    neck = widths[q:-q].min() if nbins - 2 * q > 0 else widths.min()
    return float(neck / max(lobe, 1e-6))


def extract_basic_features(reg):
    """Cheap features straight off an existing regionprops object (no pixel
    access) - safe/fast to compute for every candidate, including ones that
    will end up discarded."""
    area = reg.area
    per = reg.perimeter if reg.perimeter > 0 else 1.0
    circ = min(1.0, 4 * math.pi * area / per ** 2)
    minr = reg.axis_minor_length
    majr = reg.axis_major_length
    ar = majr / max(minr, 1e-6)
    eqd = reg.equivalent_diameter_area
    sol = reg.solidity
    cy, cx = reg.centroid
    return dict(area_px=int(area), equiv_diam_px=float(eqd),
                major_px=float(majr), minor_px=float(minr),
                circularity=float(circ), aspect_ratio=float(ar),
                solidity=float(sol), centroid_x=float(cx), centroid_y=float(cy))


def extract_texture_features(gray, reg, margin_frac=0.15):
    """Radial-texture, waist-ratio and intensity stats - the expensive part.

    Computed on a crop around the region's bounding box (+ margin) instead
    of the full window, since the filters used here (Gaussian blur, Sobel,
    structure tensor) only need local support.
    """
    ry0, rx0, ry1, rx1 = reg.bbox
    h, w = ry1 - ry0, rx1 - rx0
    mgn = max(4, int(margin_frac * max(h, w)))
    gy0, gx0 = max(0, ry0 - mgn), max(0, rx0 - mgn)
    gy1, gx1 = min(gray.shape[0], ry1 + mgn), min(gray.shape[1], rx1 + mgn)

    local_mask = np.zeros((gy1 - gy0, gx1 - gx0), np.uint8)
    oy, ox = ry0 - gy0, rx0 - gx0
    local_mask[oy:oy + h, ox:ox + w] = reg.image
    local_gray = gray[gy0:gy1, gx0:gx1]

    cy, cx = reg.centroid
    rad_score, tex = radial_texture_score(local_gray, local_mask,
                                          cx - gx0, cy - gy0)
    wr = waist_ratio(local_mask)
    inten = local_gray[local_mask > 0]
    return dict(waist_ratio=wr, radial_score=rad_score, texture_energy=tex,
                mean_intensity=float(inten.mean()),
                std_intensity=float(inten.std()))


# --------------------------------------------------------------------------- #
#  Classification
# --------------------------------------------------------------------------- #
def classify(f, args):
    """Rule-based score.  Returns (label, vaterite_score 0..1, reason)."""
    reasons = []
    # hard geometric vetoes -> other phase
    if f["aspect_ratio"] > args.max_ar:
        reasons.append(f"elongated (AR={f['aspect_ratio']:.2f})")
    if not np.isnan(f["waist_ratio"]) and f["waist_ratio"] < args.min_waist:
        reasons.append(f"dumbbell waist ({f['waist_ratio']:.2f})")
    if f["circularity"] < args.min_circ_hard:
        reasons.append(f"low circularity ({f['circularity']:.2f})")

    # soft score: circular outline + radial fibre texture + (mild) size prior
    s_circ = np.clip((f["circularity"] - 0.70) / 0.25, 0, 1)
    s_ar = np.clip((args.max_ar - f["aspect_ratio"]) / (args.max_ar - 1.0), 0, 1)
    rs = f["radial_score"]
    s_rad = 0.5 if np.isnan(rs) else np.clip((rs - 0.50) / 0.25, 0, 1)
    s_size = np.clip((f["equiv_diam_px"] - args.size_prior_px) /
                     max(args.size_prior_px, 1), 0, 1) * 0.5 + 0.5
    score = 0.30 * s_circ + 0.20 * s_ar + 0.40 * s_rad + 0.10 * s_size

    if reasons:
        return "other", float(score), "; ".join(reasons)
    label = "vaterite" if score >= args.score_thr else "other"
    why = (f"score={score:.2f} (circ {f['circularity']:.2f}, "
           f"radial {rs if not np.isnan(rs) else float('nan'):.2f})")
    return label, float(score), why


def estimate_cluster_threshold(diam_px, k=3.0, min_candidates=20,
                               median_cap_ratio=25.0):
    """Robust upper-outlier size cutoff (px), from the overview-level
    candidate population of THIS image, above which an object is treated as
    a cluster outright - without ever reading its full-resolution window.

    "Big" only means anything relative to the surrounding single-crystal
    population, and that population's scale differs a lot between images
    (e.g. across the Ca_020...Ca_100 series), so a single fixed pixel cutoff
    for every image is unreliable. Tukey's outlier rule in log-space (beyond
    Q3 + k*IQR of log10(diameter)) adapts the cutoff to each image's own
    distribution instead.

    The outliers we're trying to catch are already IN the population this is
    computed from (there's no other data to use), which can backfire: a
    heavy right tail inflates Q3/IQR themselves, and 10**(...) turns that
    inflation into an exponential blowup - checked against real data, one
    image with a few huge artifact blobs among ~600 candidates produced an
    auto threshold of 40,000+ px (far above even the artifacts themselves,
    so nothing got caught at all). The median barely moves under a few
    outliers, so the result is also capped at a generous multiple of it as
    a robust backstop against this failure mode.

    Returns None if there aren't enough candidates to trust the statistics
    (caller should fall back to a fixed default in that case).
    """
    diam_px = np.asarray(diam_px, dtype=float)
    diam_px = diam_px[diam_px > 0]
    if len(diam_px) < min_candidates:
        return None
    logd = np.log10(diam_px)
    q1, q3 = np.percentile(logd, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    tukey_hi = 10 ** (q3 + k * iqr)
    median_cap = float(np.median(diam_px)) * median_cap_ratio
    return float(min(tukey_hi, median_cap))


def estimate_vaterite_min_diam(all_candidate_diam_px, percentile=90,
                               min_candidates=20):
    """Minimum diameter (px) for a shape/texture-scored vaterite candidate to
    be trusted, from the size distribution of ALL of this image's candidates
    (the same population used for the cluster threshold).

    A small object can score well on circularity+radial-texture by pure
    chance - a handful of pixels give a very unstable coherence estimate -
    so shape/texture alone isn't enough to trust for small candidates.
    Checked against real data: the overall crystal-size distribution here is
    a single smooth, heavy right tail (many small crystals, progressively
    fewer large ones), NOT two separate size populations - so there is no
    "gap" to split on, and confirmed genuine vaterites simply sit in the
    upper tail of that one distribution (matching them being described as
    "relatively big" - see module docstring). A percentile cutoff against
    the whole population is the right tool for a tail, not a bimodal split.

    Returns None if there are too few candidates to trust the percentile.
    """
    diam = np.asarray(all_candidate_diam_px, dtype=float)
    diam = diam[diam > 0]
    if len(diam) < min_candidates:
        return None
    return float(np.percentile(diam, percentile))


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #
COLORS = {"vaterite": (60, 200, 60),      # green  (BGR)
          "other": (60, 120, 245),        # orange/red
          "cluster": (200, 60, 180),      # purple/magenta
          "discarded": (160, 160, 160)}   # gray


def run(args, path):
    big = BigImage(path)
    H, W = big.shape
    print(f"[info] image {W} x {H} px")

    overview, scale = build_overview(big, args.overview_width)
    print(f"[info] overview {overview.shape[1]} x {overview.shape[0]} "
          f"(scale 1/{scale})")

    min_diam_ov = max(4, args.min_diam // scale)
    mask_ov = foreground_mask(overview, min_diam_ov)
    blobs_ov = sk_label(mask_ov)
    cands = [r for r in regionprops(blobs_ov)
             if r.equivalent_diameter_area >= min_diam_ov * 0.8]
    print(f"[info] {len(cands)} candidate object(s) on overview", flush=True)

    # pre-emptive size statistics over the whole candidate population, so
    # oversized/merged objects can be flagged as clusters and skip expensive
    # full-resolution processing entirely, instead of only finding out after
    # reading and segmenting them
    diam_ov_full = np.array([c.equivalent_diameter_area * scale for c in cands])
    if args.cluster_diam_px is not None:
        cluster_thr = args.cluster_diam_px
        print(f"[info] cluster size threshold: {cluster_thr:.0f}px "
              f"(manual --cluster-diam-px)", flush=True)
    else:
        cluster_thr = estimate_cluster_threshold(diam_ov_full,
                                                  k=args.cluster_outlier_k)
        if cluster_thr is None:
            cluster_thr = 400.0
            print(f"[info] only {len(diam_ov_full)} candidate(s); too few for "
                  f"size statistics, using fallback cluster threshold "
                  f"{cluster_thr:.0f}px", flush=True)
        else:
            p10, p50, p90 = np.percentile(diam_ov_full, [10, 50, 90])
            print(f"[info] candidate diameter stats (n={len(diam_ov_full)}): "
                  f"p10={p10:.0f}px  median={p50:.0f}px  p90={p90:.0f}px  "
                  f"-> auto cluster threshold={cluster_thr:.0f}px "
                  f"(k={args.cluster_outlier_k})", flush=True)

    if args.limit:
        cands = cands[: args.limit]
        print(f"[info] --limit {args.limit}: only the first {len(cands)} "
              f"will be processed", flush=True)

    annot = cv2.cvtColor(overview, cv2.COLOR_GRAY2BGR)
    rows, next_id = [], 1
    t_start = time.time()
    # vaterite is decided in two passes: shape/texture first (below), then a
    # size-based filter over all of THIS image's vaterite-scoring candidates
    # (see estimate_vaterite_min_diam) - so drawing/labeling for them is
    # deferred until after the loop instead of happening immediately
    pending_vaterite = []

    for ci, cand in enumerate(cands, 1):
        y0o, x0o, y1o, x1o = cand.bbox
        eqd_ov_full = float(cand.equivalent_diameter_area * scale)
        pad = int(max(8, 0.25 * (max(y1o - y0o, x1o - x0o))))
        y0, x0 = (y0o - pad) * scale, (x0o - pad) * scale
        y1, x1 = (y1o + pad) * scale, (x1o + pad) * scale
        win_h, win_w = y1 - y0, x1 - x0

        cxg = ((x0o + x1o) // 2) * scale
        cyg = ((y0o + y1o) // 2) * scale
        dt = time.time() - t_start
        eta = (dt / ci) * (len(cands) - ci)
        print(f"[{ci:3d}/{len(cands)}] at ({cxg:6d},{cyg:6d})  "
              f"win {win_w}x{win_h} px   "
              f"elapsed {dt:5.1f}s   eta {eta:5.1f}s",
              flush=True)

        oversized = eqd_ov_full >= cluster_thr
        toowide = args.max_window_px and max(win_h, win_w) > args.max_window_px
        if oversized or toowide:
            reason = (f"overview-level diameter {eqd_ov_full:.0f}px >= cluster "
                      f"threshold {cluster_thr:.0f}px" if oversized else
                      f"candidate window {win_w}x{win_h}px exceeds "
                      f"--max-window-px ({args.max_window_px})")
            print(f"            -> {reason}; classified as cluster directly, "
                  f"full-resolution analysis skipped", flush=True)
            feats = dict.fromkeys(
                ["area_px", "major_px", "minor_px", "circularity",
                 "aspect_ratio", "solidity", "waist_ratio", "radial_score",
                 "texture_energy", "mean_intensity", "std_intensity"], np.nan)
            feats.update(dict(
                id=next_id, label="cluster", vaterite_score=np.nan,
                note=reason + "; too large to attempt separation",
                touching=True, equiv_diam_px=eqd_ov_full,
                equiv_diam_um=(eqd_ov_full * args.pixel_size
                               if args.pixel_size else np.nan),
                major_um=np.nan, minor_um=np.nan,
                centroid_x=cxg, centroid_y=cyg))
            rows.append(feats)

            cnts, _ = cv2.findContours(cand.image.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cv2.drawContours(annot, [c + [x0o, y0o]], -1,
                                 COLORS["cluster"], 2)
            cv2.putText(annot, str(next_id),
                        (int(cxg / scale) - 8, int(cyg / scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["cluster"], 2)
            next_id += 1
            continue

        win = big.window(y0, y1, x0, x1)
        y0c, x0c = max(0, y0), max(0, x0)

        # fine segmentation in the full-resolution window
        m = foreground_mask(win, args.min_diam)
        _, touching, regions = split_touching(m, args.min_diam,
                                              args.dumbbell_merge_waist)

        for reg in regions:
            if reg.equivalent_diameter_area < args.min_diam:
                continue
            ry0, rx0, ry1, rx1 = reg.bbox
            # skip duplicates: keep only regions whose centroid lies in the
            # un-padded candidate area (avoids double counting across windows)
            cyw, cxw = reg.centroid
            gy, gx = cyw + y0c, cxw + x0c
            if not (y0o * scale <= gy < y1o * scale and
                    x0o * scale <= gx < x1o * scale):
                continue

            border = (ry0 == 0 and y0c > 0) or (rx0 == 0 and x0c > 0) \
                or (ry1 == m.shape[0]) or (rx1 == m.shape[1]) \
                or gy < args.min_diam or gx < args.min_diam \
                or gy > H - args.min_diam or gx > W - args.min_diam

            feats = extract_basic_features(reg)
            is_touch = bool(touching[reg.label]) if len(touching) else False

            unresolved_touch = is_touch and (
                feats["solidity"] < args.min_solidity or
                feats["circularity"] < args.min_circ_hard)

            if border:
                # unreliable outline/size either way - skip the expensive
                # radial-texture / waist-ratio / intensity computation
                feats.update(dict(waist_ratio=np.nan, radial_score=np.nan,
                                  texture_energy=np.nan, mean_intensity=np.nan,
                                  std_intensity=np.nan))
                lab, score, why = "discarded", np.nan, "touches image border"
            elif unresolved_touch:
                feats.update(dict(waist_ratio=np.nan, radial_score=np.nan,
                                  texture_energy=np.nan, mean_intensity=np.nan,
                                  std_intensity=np.nan))
                lab, score, why = "cluster", np.nan, (
                    "unresolved touching aggregate "
                    f"(solidity={feats['solidity']:.2f}, "
                    f"circularity={feats['circularity']:.2f})")
            else:
                feats.update(extract_texture_features(win, reg))
                lab, score, why = classify(feats, args)
                if feats["equiv_diam_px"] > cluster_thr:
                    # size backstop: catches merged crystals that still look
                    # convex/circular as a whole and dodged the check above
                    # (shape/multi-peak splitting is the primary cluster
                    # detector - this only guards against it missing)
                    lab = "cluster"
                    why = (f"oversized (diam={feats['equiv_diam_px']:.0f}px "
                           f">= cluster threshold {cluster_thr:.0f}px); {why}")
                elif is_touch:
                    why += "; separated from touching neighbour"

            feats.update(dict(
                id=next_id, label=lab, vaterite_score=score, note=why,
                touching=is_touch,
                centroid_x=gx, centroid_y=gy,
                equiv_diam_um=(feats["equiv_diam_px"] * args.pixel_size
                               if args.pixel_size else np.nan),
                major_um=(feats["major_px"] * args.pixel_size
                          if args.pixel_size else np.nan),
                minor_um=(feats["minor_px"] * args.pixel_size
                          if args.pixel_size else np.nan)))
            rows.append(feats)

            # overview-space contour, shared by the immediate-draw and
            # deferred-vaterite paths below
            rm = reg.image.astype(np.uint8)
            cnts, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cnts = [((c + [rx0 + x0c, ry0 + y0c]) / scale).astype(np.int32)
                    for c in cnts]

            if lab == "vaterite":
                pending_vaterite.append(dict(
                    feats=feats, cnts=cnts, next_id=next_id, gx=gx, gy=gy,
                    diam=feats["equiv_diam_px"]))
            else:
                for c in cnts:
                    cv2.drawContours(annot, [c], -1, COLORS[lab], 2)
                cv2.putText(annot, str(next_id),
                            (int(gx / scale) - 8, int(gy / scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[lab], 2)
            next_id += 1

    # second pass: demote shape/texture-scored vaterite candidates that are
    # too small to trust, relative to THIS image's overall candidate size
    # distribution - see estimate_vaterite_min_diam
    if pending_vaterite:
        if args.vaterite_min_diam_px is not None:
            vat_min = args.vaterite_min_diam_px
            print(f"[info] vaterite minimum diameter: {vat_min:.0f}px "
                  f"(manual --vaterite-min-diam-px)", flush=True)
        else:
            vat_min = estimate_vaterite_min_diam(
                diam_ov_full, percentile=args.vaterite_size_percentile)
            if vat_min is not None:
                print(f"[info] auto vaterite minimum diameter={vat_min:.0f}px "
                      f"(p{args.vaterite_size_percentile:.0f} of this image's "
                      f"{len(diam_ov_full)} candidates)", flush=True)
            else:
                print(f"[info] only {len(diam_ov_full)} candidate(s); too few "
                      f"for a size percentile, no minimum-size filter "
                      f"applied", flush=True)

        n_demoted = 0
        for p in pending_vaterite:
            feats, lab = p["feats"], "vaterite"
            if vat_min is not None and p["diam"] < vat_min:
                lab = "other"
                n_demoted += 1
                feats["label"] = "other"
                feats["note"] = (f"too small for vaterite (diam="
                                 f"{p['diam']:.0f}px < {vat_min:.0f}px); "
                                 f"{feats['note']}")
            for c in p["cnts"]:
                cv2.drawContours(annot, [c], -1, COLORS[lab], 2)
            cv2.putText(annot, str(p["next_id"]),
                        (int(p["gx"] / scale) - 8, int(p["gy"] / scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[lab], 2)
        if vat_min is not None:
            print(f"[info] {n_demoted}/{len(pending_vaterite)} vaterite "
                  f"candidate(s) demoted to 'other' for being under the "
                  f"minimum diameter", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(args.outdir, base + "_crystals.csv")
    img_path = os.path.join(args.outdir, base + "_annotated.png")

    cols = ["id", "label", "vaterite_score", "equiv_diam_px", "equiv_diam_um",
            "major_px", "minor_px", "major_um", "minor_um", "area_px",
            "circularity", "aspect_ratio", "solidity", "waist_ratio",
            "radial_score", "texture_energy", "mean_intensity",
            "std_intensity", "touching", "centroid_x", "centroid_y", "note"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]] if len(df) else df
    df.to_csv(csv_path, index=False)
    cv2.imwrite(img_path, annot)
    print(f"[done] {len(df)} crystals -> {csv_path}")
    print(f"[done] annotated overview -> {img_path}")
    if len(df):
        print(df["label"].value_counts().to_string())

    if args.annotate_fullres:
        write_fullres_annotation(big, df, args, base)


def write_fullres_annotation(big, df, args, base):
    """Full-resolution annotated copy, written tile-by-tile as a compressed
    BigTIFF (never holding the full image in memory).

    tifffile's generator support for `imwrite` requires tiled output: with
    plain strips, each yielded item must be a *complete page*; with tiles,
    the iterator may yield partial (edge) tiles, which tifffile zero-pads
    automatically. So annotation is done in row bands (for efficient
    sequential disk reads), then each band is re-sliced into tile-sized
    chunks for the writer.
    """
    import tifffile
    H, W = big.shape
    out = os.path.join(args.outdir, base + "_annotated_fullres.tif")
    tile = 1024          # must be a multiple of 16
    band = tile * 4       # read/annotate in bigger bands, sliced into tiles

    def gen():
        for y0 in range(0, H, band):
            y1 = min(H, y0 + band)
            strip = cv2.cvtColor(big.window(y0, y1, 0, W), cv2.COLOR_GRAY2BGR)
            sub = df[(df.centroid_y > y0 - 4000) & (df.centroid_y < y1 + 4000)]
            for _, r in sub.iterrows():
                col = COLORS[r["label"]]
                cv2.circle(strip, (int(r.centroid_x), int(r.centroid_y - y0)),
                           int(r.equiv_diam_px * 0.6), col, 6)
                cv2.putText(strip, f"{int(r.id)}:{r['label'][0].upper()}",
                            (int(r.centroid_x), int(r.centroid_y - y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 5)
            strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            for sy0 in range(0, strip.shape[0], tile):
                sy1 = min(strip.shape[0], sy0 + tile)
                for x0 in range(0, W, tile):
                    x1 = min(W, x0 + tile)
                    yield strip[sy0:sy1, x0:x1]

    tifffile.imwrite(out, gen(), shape=(H, W, 3), dtype=np.uint8,
                     tile=(tile, tile), photometric="rgb",
                     compression="zlib", bigtiff=True)
    print(f"[done] full-resolution annotated copy -> {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="+",
                   help="one or more image files, and/or glob patterns "
                        "(e.g. \"Ca_100\\*.tif\") - PowerShell does not "
                        "expand wildcards for external programs itself, so "
                        "quote any pattern and let the script expand it")
    p.add_argument("--outdir", default="results")
    p.add_argument("--pixel-size", type=float, default=None,
                   help="micrometres per pixel at full resolution")
    p.add_argument("--min-diam", type=int, default=40,
                   help="minimum crystal diameter in full-res pixels")
    p.add_argument("--overview-width", type=int, default=6000)
    p.add_argument("--annotate-fullres", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N candidates (0=all); for "
                        "quickly testing on a fraction of the image")
    p.add_argument("--max-window-px", type=int, default=4000,
                   help="skip candidates whose full-resolution window exceeds "
                        "this size in pixels (0=no limit). This is an "
                        "absolute, statistics-independent backstop: the auto "
                        "cluster-size threshold can (rarely) be thrown off by "
                        "the very artifacts it's meant to catch (a heavy-"
                        "tailed size distribution inflates its own outlier "
                        "cutoff), so this hard ceiling guarantees a "
                        "pathological blob - a scan-seam artifact spanning "
                        "much of the image, for example - never triggers a "
                        "full-resolution read no matter what the statistics "
                        "say. Default is far above any real crystal or "
                        "cluster observed so far; lower it if your images "
                        "run at a very different scale")
    # classification thresholds (tune on your data)
    p.add_argument("--score-thr", type=float, default=0.50,
                   help="vaterite score threshold. Checked against confirmed "
                        "large, near-perfectly-circular vaterites that scored "
                        "0.50-0.53 (moderate radial-texture score alone "
                        "shouldn't override otherwise-excellent shape+size "
                        "evidence); pairs with --vaterite-min-diam-px, which "
                        "independently guards against small objects that "
                        "pass this threshold by chance")
    p.add_argument("--max-ar", type=float, default=1.45,
                   help="max aspect ratio for vaterite")
    p.add_argument("--min-waist", type=float, default=0.80,
                   help="waist/lobe ratio below which a shape is a dumbbell")
    p.add_argument("--dumbbell-merge-waist", type=float, default=0.80,
                   help="when watershed wants to split a touching-blob into "
                        "exactly 2 pieces, and the ORIGINAL unsplit blob's "
                        "waist_ratio is above this, treat it as one single "
                        "dumbbell-habit crystal instead (undo the split) - "
                        "checked against real examples, a genuine dumbbell's "
                        "neck reads as a HIGHER (gentler) waist_ratio than "
                        "two separate crystals merely touching at a sharp "
                        "point, since a solid crystal can't grow an "
                        "infinitely sharp notch. Independent from "
                        "--min-waist, which runs the opposite direction on "
                        "an already-final single region")
    p.add_argument("--min-circ-hard", type=float, default=0.60,
                   help="circularity below this is never vaterite")
    p.add_argument("--min-solidity", type=float, default=0.90)
    p.add_argument("--size-prior-px", type=float, default=150,
                   help="diameter (px) around which the soft size prior acts")
    p.add_argument("--cluster-diam-px", type=float, default=None,
                   help="equivalent diameter (px) above which an object is "
                        "labeled 'cluster' outright - checked cheaply at "
                        "overview resolution (skipping the full-resolution "
                        "read/segmentation entirely for anything over it) and "
                        "again after full-res splitting as a backstop. "
                        "Default: auto-estimated per image from the overview "
                        "candidate size distribution (see --cluster-outlier-k) "
                        "since 'big' is relative to each image's own crystal "
                        "population, not a fixed pixel count. Set explicitly "
                        "to disable auto-estimation for this run")
    p.add_argument("--cluster-outlier-k", type=float, default=3.0,
                   help="only used when --cluster-diam-px is not set: "
                        "candidates whose overview-estimated diameter is more "
                        "than k * IQR above the 75th percentile (in log10 "
                        "space) of this image's candidate size distribution "
                        "are auto-classified as clusters. Lower = more "
                        "aggressive/faster, higher = more conservative")
    p.add_argument("--vaterite-min-diam-px", type=float, default=None,
                   help="equivalent diameter (px) below which a shape/texture "
                        "vaterite candidate is demoted to 'other' - a small "
                        "object can still score well on circularity+radial "
                        "texture by chance (few pixels to average over), so "
                        "this is a second, independent filter. Default: "
                        "auto-estimated per image as a high percentile (see "
                        "--vaterite-size-percentile) of this image's overall "
                        "candidate size distribution, since genuine vaterites "
                        "are simply the large tail of that one distribution, "
                        "not a separate size population. Set explicitly to "
                        "disable auto-estimation for this run")
    p.add_argument("--vaterite-size-percentile", type=float, default=90,
                   help="only used when --vaterite-min-diam-px is not set: "
                        "percentile (0-100) of this image's overall candidate "
                        "diameter distribution used as the vaterite minimum "
                        "size. Lower = more permissive (more small objects "
                        "kept as vaterite), higher = stricter")
    return p.parse_args(argv)


def expand_inputs(patterns):
    """Expand glob patterns; plain paths are passed through as-is even if
    they don't currently exist (so a clear per-file error can be raised)."""
    paths, seen = [], set()
    for pat in patterns:
        matches = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        if not matches:
            print(f"[warn] no files matched: {pat}", file=sys.stderr)
        for m in matches:
            if m not in seen:
                seen.add(m)
                paths.append(m)
    return paths


def main():
    args = parse_args()
    paths = expand_inputs(args.input)
    if not paths:
        print("[error] no input files found", file=sys.stderr)
        sys.exit(1)

    failed = []
    for i, path in enumerate(paths, 1):
        print(f"\n===== [{i}/{len(paths)}] {path} =====", flush=True)
        try:
            run(args, path)
        except Exception as e:
            print(f"[error] failed on {path}: {e}", file=sys.stderr)
            failed.append(path)

    if failed:
        print(f"\n[done] {len(paths) - len(failed)}/{len(paths)} succeeded; "
              f"failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
