#!/usr/bin/env python3
"""Compare two OpenCV camera calibrations and their checkerboard views.

The rational distortion coefficients are strongly correlated, so coefficient
subtraction alone is not a useful equivalence test.  This tool also compares
effective field of view, ray direction across the image, same-ray pixel
projection, checkerboard coverage, reprojection residuals, and the PnP pose
change produced by swapping calibration files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrate_camera import (  # noqa: E402
    board_candidates,
    coverage_grid,
    load_saved_views,
    max_corner_radius,
    validate_model,
)


def _float(value) -> float:
    return float(np.asarray(value).reshape(()))


def load_calibration(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"camera_matrix", "dist_coeffs", "image_w", "image_h"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        calibration = {
            "path": str(path),
            "K": np.asarray(archive["camera_matrix"], dtype=float),
            "dist": np.asarray(archive["dist_coeffs"], dtype=float),
            "width": int(_float(archive["image_w"])),
            "height": int(_float(archive["image_h"])),
        }
    if calibration["K"].shape != (3, 3):
        raise ValueError(f"{path}: camera_matrix is not 3x3")
    if not np.isfinite(calibration["K"]).all() or not np.isfinite(
            calibration["dist"]).all():
        raise ValueError(f"{path}: non-finite calibration value")
    if calibration["width"] <= 0 or calibration["height"] <= 0:
        raise ValueError(f"{path}: invalid image size")
    return calibration


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    dots = np.sum(a * b, axis=-1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def rays_for_pixels(points: np.ndarray, calibration: dict) -> np.ndarray:
    xy = cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        calibration["K"], calibration["dist"],
    ).reshape(-1, 2)
    return np.column_stack([xy, np.ones(len(xy))])


def effective_fov(calibration: dict) -> dict:
    width, height = calibration["width"], calibration["height"]
    points = np.array([
        [0.0, height / 2.0],
        [width - 1.0, height / 2.0],
        [width / 2.0, 0.0],
        [width / 2.0, height - 1.0],
    ])
    rays = rays_for_pixels(points, calibration)
    return {
        "horizontal_deg": float(angle_deg(rays[0], rays[1])),
        "vertical_deg": float(angle_deg(rays[2], rays[3])),
    }


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def summarize_signed(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def geometry_comparison(old: dict, new: dict) -> dict:
    if (old["width"], old["height"]) != (new["width"], new["height"]):
        raise ValueError("calibrations have different image sizes")

    width, height = old["width"], old["height"]
    xs = np.linspace(0.0, width - 1.0, 65)
    ys = np.linspace(0.0, height - 1.0, 37)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    old_rays = rays_for_pixels(pixels, old)
    new_rays = rays_for_pixels(pixels, new)
    angular_delta = angle_deg(old_rays, new_rays)

    zero = np.zeros(3)
    old_projected, _ = cv2.projectPoints(
        old_rays, zero, zero, old["K"], old["dist"])
    new_projected, _ = cv2.projectPoints(
        old_rays, zero, zero, new["K"], new["dist"])
    projection_delta = np.linalg.norm(
        new_projected.reshape(-1, 2) - old_projected.reshape(-1, 2), axis=1)

    fov_old = effective_fov(old)
    fov_new = effective_fov(new)
    K_old, K_new = old["K"], new["K"]
    intrinsic_names = {
        "fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)
    }
    intrinsics = {}
    for name, index in intrinsic_names.items():
        before = float(K_old[index])
        after = float(K_new[index])
        intrinsics[name] = {
            "old_px": before,
            "new_px": after,
            "delta_px": after - before,
            "delta_percent_of_old": 100.0 * (after - before) / before,
        }

    angular_summary = summarize(angular_delta)
    return {
        "image_size": [width, height],
        "intrinsics": intrinsics,
        "effective_fov_deg": {
            "old": fov_old,
            "new": fov_new,
            "delta": {
                axis: fov_new[axis] - fov_old[axis]
                for axis in ("horizontal_deg", "vertical_deg")
            },
        },
        "same_pixel_ray_angle_delta_deg": angular_summary,
        "equivalent_lateral_delta_at_2m_cm": {
            key: 200.0 * math.tan(math.radians(value))
            for key, value in angular_summary.items()
        },
        "same_ray_projection_delta_px": summarize(projection_delta),
        "dist_coefficients": {
            "old": old["dist"].ravel().tolist(),
            "new": new["dist"].ravel().tolist(),
            "delta": (new["dist"] - old["dist"]).ravel().tolist(),
            "interpretation": (
                "Diagnostic only: rational-model coefficients are correlated; "
                "use ray/projection differences for functional comparison."
            ),
        },
    }


def load_view_set(folder: Path, cols: int, rows: int, square: float) -> dict:
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    objects, images, shape = load_saved_views(
        str(folder), cols, rows, square, criteria,
        candidates=board_candidates(cols, rows, "auto"),
    )
    if not images or shape is None:
        raise ValueError(f"{folder}: no usable checkerboard views")
    points = [image.reshape(-1, 2) for image in images]
    coverage = coverage_grid(points, shape)
    reached, corner = max_corner_radius(points, shape)
    return {
        "folder": str(folder),
        "objects": objects,
        "images": images,
        "shape": shape,
        "coverage": {
            "usable_views": len(images),
            "grid": coverage.tolist(),
            "empty_cells": int(np.count_nonzero(coverage == 0)),
            "minimum_views_per_cell": int(coverage.min()),
            "furthest_corner_radius_fraction": float(reached / corner),
        },
    }


def fit_rms(view_set: dict) -> dict:
    shape = view_set["shape"]
    rms, K, dist, _, _ = cv2.calibrateCamera(
        view_set["objects"], view_set["images"],
        (shape[1], shape[0]), None, None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )
    return {"rms_px": float(rms), "K": K, "dist": dist}


def pnp_results(view_set: dict, calibration: dict) -> dict:
    per_view_rms = []
    rotations = []
    translations = []
    total_squared_error = 0.0
    total_points = 0
    for object_points, image_points in zip(
            view_set["objects"], view_set["images"]):
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, calibration["K"],
            calibration["dist"], flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, calibration["K"], calibration["dist"])
        residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        squared = np.sum(residual * residual, axis=1)
        per_view_rms.append(float(np.sqrt(np.mean(squared))))
        total_squared_error += float(np.sum(squared))
        total_points += len(squared)
        rotations.append(rvec.reshape(3))
        translations.append(tvec.reshape(3))
    return {
        "global_rms_px": math.sqrt(total_squared_error / total_points),
        "per_view_rms_px": summarize(np.asarray(per_view_rms)),
        "solved_views": len(per_view_rms),
        "rotations": np.asarray(rotations),
        "translations": np.asarray(translations),
    }


def pose_delta(first: dict, second: dict) -> dict:
    count = min(len(first["translations"]), len(second["translations"]))
    translation_delta = []
    range_delta_percent = []
    rotation_delta = []
    for index in range(count):
        t_first = first["translations"][index]
        t_second = second["translations"][index]
        translation_delta.append(100.0 * np.linalg.norm(t_second - t_first))
        range_first = np.linalg.norm(t_first)
        range_second = np.linalg.norm(t_second)
        range_delta_percent.append(
            100.0 * (range_second - range_first) / range_first)
        R_first, _ = cv2.Rodrigues(first["rotations"][index])
        R_second, _ = cv2.Rodrigues(second["rotations"][index])
        relative = R_first.T @ R_second
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_delta.append(math.degrees(math.acos(float(cosine))))
    range_delta_percent = np.asarray(range_delta_percent)
    return {
        "translation_norm_cm": summarize(np.asarray(translation_delta)),
        "range_delta_percent_signed": summarize_signed(range_delta_percent),
        "range_delta_percent_abs": summarize(np.abs(range_delta_percent)),
        "rotation_angle_deg": summarize(np.asarray(rotation_delta)),
    }


def model_validation(calibration: dict) -> dict:
    report = validate_model(
        calibration["K"], calibration["dist"],
        (calibration["height"], calibration["width"]),
    )
    return {
        "roundtrip_max_error_px": report["max_error_px"],
        "roundtrip_bad_fraction_over_1px": report["bad_fraction"],
        "safe_radius_fraction": (
            report["safe_radius_px"] / report["corner_radius_px"]
        ),
    }


def compare(
    old_calibration: Path,
    new_calibration: Path,
    old_views: Path,
    new_views: Path,
    cols: int = 10,
    rows: int = 8,
    square: float = 0.20,
) -> dict:
    old = load_calibration(old_calibration)
    new = load_calibration(new_calibration)
    old_set = load_view_set(old_views, cols, rows, square)
    new_set = load_view_set(new_views, cols, rows, square)
    old_fit = fit_rms(old_set)
    new_fit = fit_rms(new_set)

    old_on_old = pnp_results(old_set, old)
    new_on_old = pnp_results(old_set, new)
    old_on_new = pnp_results(new_set, old)
    new_on_new = pnp_results(new_set, new)

    result = {
        "calibrations": {
            "old": str(old_calibration),
            "new": str(new_calibration),
        },
        "geometry": geometry_comparison(old, new),
        "model_validation": {
            "old": model_validation(old),
            "new": model_validation(new),
        },
        "view_quality": {
            "old": {
                **old_set["coverage"],
                "refit_rms_px": old_fit["rms_px"],
                "stored_refit_K_max_abs_delta_px": float(
                    np.max(np.abs(old_fit["K"] - old["K"]))),
            },
            "new": {
                **new_set["coverage"],
                "refit_rms_px": new_fit["rms_px"],
                "stored_refit_K_max_abs_delta_px": float(
                    np.max(np.abs(new_fit["K"] - new["K"]))),
            },
        },
        "checkerboard_cross_validation": {
            "old_views": {
                "old_model_global_rms_px": old_on_old["global_rms_px"],
                "new_model_global_rms_px": new_on_old["global_rms_px"],
                "rms_ratio_new_over_old": (
                    new_on_old["global_rms_px"] / old_on_old["global_rms_px"]
                ),
                "pose_delta": pose_delta(old_on_old, new_on_old),
            },
            "new_views": {
                "new_model_global_rms_px": new_on_new["global_rms_px"],
                "old_model_global_rms_px": old_on_new["global_rms_px"],
                "rms_ratio_old_over_new": (
                    old_on_new["global_rms_px"] / new_on_new["global_rms_px"]
                ),
                "pose_delta": pose_delta(new_on_new, old_on_new),
            },
        },
        "interpretation_notes": [
            "PnP cross-validation refits board pose separately for each model.",
            "Pose deltas use the stated checkerboard square size for metric scale.",
            "Same-pixel ray deltas are the most direct wrong-calibration impact.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_calibration", type=Path)
    parser.add_argument("new_calibration", type=Path)
    parser.add_argument("--old-views", required=True, type=Path)
    parser.add_argument("--new-views", required=True, type=Path)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--square", type=float, default=0.20)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = compare(
        args.old_calibration,
        args.new_calibration,
        args.old_views,
        args.new_views,
        cols=args.cols,
        rows=args.rows,
        square=args.square,
    )
    payload = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"[compare] saved {args.json_out}")
    print(payload)


if __name__ == "__main__":
    main()
