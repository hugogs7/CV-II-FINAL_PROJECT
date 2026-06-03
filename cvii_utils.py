"""
cvii_utils.py - Utility functions for:
  "Temporal Stability of 6D Head Pose Estimation on BIWI-360 Video Sequences"
  Computer Vision II - MIA - Universidade de Santiago de Compostela
  Authors: Hugo Garcia Souto and Adrian Martinez Balea

All functions here are pure (they depend only on their arguments, numpy and
pandas). Model-dependent inference code (e.g. predict_single_image) stays in
the notebook because it needs the loaded model, device and repository helpers.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Angle and rotation math
# ----------------------------------------------------------------------------

def angular_error_deg(pred, gt):
    """Smallest absolute angular difference in degrees."""
    return min(abs(pred - gt), abs(pred + 360 - gt), abs(pred - 360 - gt))


def angular_diff_array_deg(pred, gt):
    """Smallest absolute angular difference in degrees, vectorized."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    
    diff1 = np.abs(pred - gt)
    diff2 = np.abs(pred + 360 - gt)
    diff3 = np.abs(pred - 360 - gt)
    
    return np.minimum(np.minimum(diff1, diff2), diff3)


def signed_circular_diff_deg(current, previous):
    """
    Signed circular difference current - previous in degrees.
    Output is in [-180, 180).
    """
    diff = np.asarray(current) - np.asarray(previous)
    return (diff + 180) % 360 - 180


def circular_angle_diff_deg(current, previous):
    diff = current - previous
    return (diff + 180) % 360 - 180


def circular_mean_deg(values):
    """Circular mean of angles in degrees."""
    values_rad = np.deg2rad(values)
    mean_sin = np.mean(np.sin(values_rad))
    mean_cos = np.mean(np.cos(values_rad))
    return np.rad2deg(np.arctan2(mean_sin, mean_cos))


def wrap_angle_deg(angle):
    """Wrap angle values to [-180, 180)."""
    return (angle + 180) % 360 - 180


def unwrap_angle_sequence_deg(values):
    """Unwrap a sequence of angles in degrees to avoid artificial ±180 jumps."""
    return np.rad2deg(np.unwrap(np.deg2rad(values)))


def shrink_angles_towards_mean_deg(values, mean_angle, shrink_strength):
    """
    Circular shrinkage towards a constant mean angle.
    shrink_strength = 0.0 keeps the values unchanged.
    shrink_strength = 1.0 collapses all values to the mean angle.
    """
    delta = signed_circular_diff_deg(values, mean_angle)
    return wrap_angle_deg(mean_angle + (1.0 - shrink_strength) * delta)


def align_angle_series_to_reference(angle_values, reference_values):
    """
    For visualization only.
    Shifts each angle by multiples of 360 so that it is closest to the reference angle.
    This avoids misleading vertical jumps at the ±180 degree boundary.
    """
    angle_values = np.asarray(angle_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    
    candidates = np.stack([
        angle_values - 360,
        angle_values,
        angle_values + 360
    ], axis=0)
    
    best_indices = np.argmin(np.abs(candidates - reference_values), axis=0)
    aligned = candidates[best_indices, np.arange(len(angle_values))]
    
    return aligned


def assign_level(value, q75, q90):
    if value >= q90:
        return "high"
    elif value >= q75:
        return "medium"
    else:
        return "low"


def mean_mae_for_signs(data, sy=1, sr=1, sp=1):
    yaw_errors = [
        angular_error_deg(sy * p, g)
        for p, g in zip(data["pred_yaw_raw"], data["gt_yaw"])
    ]
    roll_errors = [
        angular_error_deg(sr * p, g)
        for p, g in zip(data["pred_roll_raw"], data["gt_roll"])
    ]
    pitch_errors = [
        angular_error_deg(sp * p, g)
        for p, g in zip(data["pred_pitch_raw"], data["gt_pitch"])
    ]
    
    return {
        "sign_yaw": sy,
        "sign_roll": sr,
        "sign_pitch": sp,
        "mae_yaw": np.mean(yaw_errors),
        "mae_roll": np.mean(roll_errors),
        "mae_pitch": np.mean(pitch_errors),
        "mean_mae": np.mean([np.mean(yaw_errors), np.mean(roll_errors), np.mean(pitch_errors)])
    }


# ----------------------------------------------------------------------------
# 1D temporal filters
# ----------------------------------------------------------------------------

def ema_1d(values, alpha):
    """Standard exponential moving average."""
    smoothed = np.zeros_like(values, dtype=np.float32)
    smoothed[0] = values[0]
    
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]
    
    return smoothed


def ema_with_jump_clipping_1d(values, alpha, max_jump_deg):
    """
    Causal EMA with jump clipping.
    Before updating the EMA, the raw value is clipped so that it cannot move
    more than max_jump_deg away from the previous smoothed value.
    """
    smoothed = np.zeros_like(values, dtype=np.float32)
    smoothed[0] = values[0]
    
    for i in range(1, len(values)):
        delta = values[i] - smoothed[i - 1]
        clipped_delta = np.clip(delta, -max_jump_deg, max_jump_deg)
        clipped_target = smoothed[i - 1] + clipped_delta
        smoothed[i] = alpha * clipped_target + (1.0 - alpha) * smoothed[i - 1]
    
    return smoothed


def rolling_median_1d(values, window_size, centered):
    """Rolling median filter for a 1D sequence."""
    return (
        pd.Series(values)
        .rolling(window=window_size, center=centered, min_periods=1)
        .median()
        .values
    )


def rolling_mean_1d(values, window_size, centered):
    """Rolling mean filter for a 1D sequence."""
    return (
        pd.Series(values)
        .rolling(window=window_size, center=centered, min_periods=1)
        .mean()
        .values
    )


# ----------------------------------------------------------------------------
# Temporal / dataframe operations
# ----------------------------------------------------------------------------

def add_temporal_deltas(dataframe):
    df_out = dataframe.sort_values(["sequence_id", "frame_id"]).copy()
    
    for angle in ["yaw", "roll", "pitch"]:
        df_out[f"d_{angle}"] = df_out.groupby("sequence_id")[angle].diff()
        df_out[f"abs_d_{angle}"] = df_out[f"d_{angle}"].abs()
    
    df_out["gt_motion_magnitude"] = np.sqrt(
        df_out["d_yaw"].fillna(0) ** 2 +
        df_out["d_roll"].fillna(0) ** 2 +
        df_out["d_pitch"].fillna(0) ** 2
    )
    
    return df_out


def smooth_predictions_by_sequence(dataframe, alpha):
    """
    Apply EMA independently to each sequence and each predicted angle.
    We unwrap angles before smoothing to avoid artificial jumps around ±180 degrees.
    """
    out = dataframe.sort_values(["sequence_id", "frame_id"]).copy()
    
    for angle in ["yaw", "roll", "pitch"]:
        out[f"smooth_pred_{angle}"] = np.nan
    
    for sequence_id, seq_df in out.groupby("sequence_id"):
        idx = seq_df.index
        
        for angle in ["yaw", "roll", "pitch"]:
            raw_values_deg = seq_df[f"pred_{angle}"].values
            
            # Unwrap in radians, smooth, then convert back to degrees and wrap
            raw_values_rad = np.deg2rad(raw_values_deg)
            unwrapped_deg = np.rad2deg(np.unwrap(raw_values_rad))
            
            smoothed_deg = ema_1d(unwrapped_deg, alpha=alpha)
            smoothed_deg = wrap_angle_deg(smoothed_deg)
            
            out.loc[idx, f"smooth_pred_{angle}"] = smoothed_deg
    
    return out


def compute_smoothed_frame_errors(dataframe):
    """Compute frame-level angular errors for smoothed predictions."""
    out = dataframe.copy()
    
    for angle in ["yaw", "roll", "pitch"]:
        out[f"smooth_err_{angle}"] = angular_diff_array_deg(
            out[f"smooth_pred_{angle}"],
            out[angle]
        )
    
    out["smooth_err_mean"] = out[[
        "smooth_err_yaw",
        "smooth_err_roll",
        "smooth_err_pitch"
    ]].mean(axis=1)
    
    return out


def compute_temporal_metrics_for_prediction_columns(dataframe, pred_prefix):
    """
    Compute temporal metrics for either raw predictions or smoothed predictions.
    pred_prefix should be 'pred' or 'smooth_pred'.
    """
    temp = dataframe.sort_values(["sequence_id", "frame_id"]).copy()
    
    temp["prev_frame_id_eval"] = temp.groupby("sequence_id")["frame_id"].shift(1)
    temp["frame_gap_eval"] = temp["frame_id"] - temp["prev_frame_id_eval"]
    temp = temp[temp["frame_gap_eval"] == 1].copy().reset_index(drop=True)
    
    for angle in ["yaw", "roll", "pitch"]:
        temp[f"prev_{angle}"] = temp.groupby("sequence_id")[angle].shift(1)
        temp[f"prev_{pred_prefix}_{angle}"] = temp.groupby("sequence_id")[f"{pred_prefix}_{angle}"].shift(1)
        
        temp[f"gt_delta_{angle}"] = signed_circular_diff_deg(
            temp[angle],
            temp[f"prev_{angle}"]
        )
        
        temp[f"{pred_prefix}_delta_{angle}"] = signed_circular_diff_deg(
            temp[f"{pred_prefix}_{angle}"],
            temp[f"prev_{pred_prefix}_{angle}"]
        )
        
        temp[f"{pred_prefix}_jitter_{angle}"] = np.abs(temp[f"{pred_prefix}_delta_{angle}"])
        temp[f"{pred_prefix}_temporal_error_{angle}"] = np.abs(
            temp[f"{pred_prefix}_delta_{angle}"] - temp[f"gt_delta_{angle}"]
        )
    
    temp = temp.dropna().reset_index(drop=True)
    
    temp["gt_motion_magnitude"] = np.sqrt(
        temp["gt_delta_yaw"] ** 2 +
        temp["gt_delta_roll"] ** 2 +
        temp["gt_delta_pitch"] ** 2
    )
    
    temp[f"{pred_prefix}_jitter_magnitude"] = np.sqrt(
        temp[f"{pred_prefix}_delta_yaw"] ** 2 +
        temp[f"{pred_prefix}_delta_roll"] ** 2 +
        temp[f"{pred_prefix}_delta_pitch"] ** 2
    )
    
    temp[f"{pred_prefix}_temporal_motion_error_magnitude"] = np.sqrt(
        temp[f"{pred_prefix}_temporal_error_yaw"] ** 2 +
        temp[f"{pred_prefix}_temporal_error_roll"] ** 2 +
        temp[f"{pred_prefix}_temporal_error_pitch"] ** 2
    )
    
    return temp


def apply_temporal_filter_by_sequence(dataframe, method, params, output_prefix="robust_pred"):
    """
    Apply a temporal filter independently to each sequence and each predicted angle.
    The filter uses only model predictions, never ground truth.
    """
    out = dataframe.sort_values(["sequence_id", "frame_id"]).copy()
    
    for angle in ["yaw", "roll", "pitch"]:
        out[f"{output_prefix}_{angle}"] = np.nan
    
    for sequence_id, seq_df in out.groupby("sequence_id"):
        idx = seq_df.index
        
        for angle in ["yaw", "roll", "pitch"]:
            raw_values = seq_df[f"pred_{angle}"].values.astype(float)
            unwrapped = unwrap_angle_sequence_deg(raw_values)
            
            if method == "rolling_median":
                filtered = rolling_median_1d(
                    unwrapped,
                    window_size=params["window_size"],
                    centered=params["centered"]
                )
            
            elif method == "rolling_mean":
                filtered = rolling_mean_1d(
                    unwrapped,
                    window_size=params["window_size"],
                    centered=params["centered"]
                )
            
            elif method == "median_then_ema":
                median_values = rolling_median_1d(
                    unwrapped,
                    window_size=params["window_size"],
                    centered=params["centered"]
                )
                filtered = ema_1d(
                    median_values,
                    alpha=params["alpha"]
                )
            
            elif method == "ema_clip":
                filtered = ema_with_jump_clipping_1d(
                    unwrapped,
                    alpha=params["alpha"],
                    max_jump_deg=params["max_jump_deg"]
                )
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            out.loc[idx, f"{output_prefix}_{angle}"] = wrap_angle_deg(filtered)
    
    return out


def add_frame_errors_for_prefix(dataframe, pred_prefix, error_prefix):
    """Add frame-level angular errors for prediction columns with a given prefix."""
    out = dataframe.copy()
    
    for angle in ["yaw", "roll", "pitch"]:
        out[f"{error_prefix}_{angle}"] = angular_diff_array_deg(
            out[f"{pred_prefix}_{angle}"],
            out[angle]
        )
    
    out[f"{error_prefix}_mean"] = out[
        [f"{error_prefix}_yaw", f"{error_prefix}_roll", f"{error_prefix}_pitch"]
    ].mean(axis=1)
    
    return out


def compute_temporal_metrics_for_prefix(dataframe, pred_prefix):
    """
    Compute temporal metrics for prediction columns with a given prefix.
    Expects columns like '{pred_prefix}_yaw', '{pred_prefix}_roll', '{pred_prefix}_pitch'.
    """
    temp = dataframe.sort_values(["sequence_id", "frame_id"]).copy()
    
    temp["prev_frame_id_eval"] = temp.groupby("sequence_id")["frame_id"].shift(1)
    temp["frame_gap_eval"] = temp["frame_id"] - temp["prev_frame_id_eval"]
    
    for angle in ["yaw", "roll", "pitch"]:
        temp[f"prev_{angle}"] = temp.groupby("sequence_id")[angle].shift(1)
        temp[f"prev_{pred_prefix}_{angle}"] = temp.groupby("sequence_id")[f"{pred_prefix}_{angle}"].shift(1)
    
    temp = temp[temp["frame_gap_eval"] == 1].copy().reset_index(drop=True)
    
    for angle in ["yaw", "roll", "pitch"]:
        temp[f"gt_delta_{angle}"] = signed_circular_diff_deg(
            temp[angle],
            temp[f"prev_{angle}"]
        )
        
        temp[f"{pred_prefix}_delta_{angle}"] = signed_circular_diff_deg(
            temp[f"{pred_prefix}_{angle}"],
            temp[f"prev_{pred_prefix}_{angle}"]
        )
        
        temp[f"{pred_prefix}_jitter_{angle}"] = np.abs(temp[f"{pred_prefix}_delta_{angle}"])
        temp[f"{pred_prefix}_temporal_error_{angle}"] = np.abs(
            temp[f"{pred_prefix}_delta_{angle}"] - temp[f"gt_delta_{angle}"]
        )
    
    temp = temp.dropna().reset_index(drop=True)
    
    temp["gt_motion_magnitude"] = np.sqrt(
        temp["gt_delta_yaw"] ** 2 +
        temp["gt_delta_roll"] ** 2 +
        temp["gt_delta_pitch"] ** 2
    )
    
    temp[f"{pred_prefix}_jitter_magnitude"] = np.sqrt(
        temp[f"{pred_prefix}_delta_yaw"] ** 2 +
        temp[f"{pred_prefix}_delta_roll"] ** 2 +
        temp[f"{pred_prefix}_delta_pitch"] ** 2
    )
    
    temp[f"{pred_prefix}_temporal_motion_error_magnitude"] = np.sqrt(
        temp[f"{pred_prefix}_temporal_error_yaw"] ** 2 +
        temp[f"{pred_prefix}_temporal_error_roll"] ** 2 +
        temp[f"{pred_prefix}_temporal_error_pitch"] ** 2
    )
    
    return temp


def apply_single_axis_filter(seq_df, angle, filter_name, validation_means):
    """
    Apply one axis-specific filter to one sequence and one angle.
    Uses predictions only, except for validation_mean, which is estimated
    from the validation split and then applied as a constant baseline.
    """
    raw_values = seq_df[f"pred_{angle}"].values.astype(float)
    unwrapped = unwrap_angle_sequence_deg(raw_values)

    if filter_name == "raw":
        filtered = unwrapped

    elif filter_name == "centered_median_w5":
        filtered = rolling_median_1d(unwrapped, window_size=5, centered=True)

    elif filter_name == "centered_median_w7":
        filtered = rolling_median_1d(unwrapped, window_size=7, centered=True)

    elif filter_name == "centered_median_w11":
        filtered = rolling_median_1d(unwrapped, window_size=11, centered=True)

    elif filter_name == "centered_median_w11_ema05":
        median_values = rolling_median_1d(unwrapped, window_size=11, centered=True)
        filtered = ema_1d(median_values, alpha=0.5)

    elif filter_name == "centered_median_w11_shrink025":
        median_values = rolling_median_1d(unwrapped, window_size=11, centered=True)
        filtered = shrink_angles_towards_mean_deg(
            wrap_angle_deg(median_values),
            validation_means[angle],
            shrink_strength=0.25
        )

    elif filter_name == "centered_median_w11_shrink050":
        median_values = rolling_median_1d(unwrapped, window_size=11, centered=True)
        filtered = shrink_angles_towards_mean_deg(
            wrap_angle_deg(median_values),
            validation_means[angle],
            shrink_strength=0.50
        )

    elif filter_name == "validation_mean":
        filtered = np.full_like(raw_values, validation_means[angle], dtype=float)

    else:
        raise ValueError(f"Unknown filter name: {filter_name}")

    return wrap_angle_deg(filtered)


def apply_axis_specific_filter(dataframe, axis_config, output_prefix, validation_means):
    """
    Apply a different filter to yaw, roll and pitch independently.
    """
    out = dataframe.sort_values(["sequence_id", "frame_id"]).copy()

    for angle in ["yaw", "roll", "pitch"]:
        out[f"{output_prefix}_{angle}"] = np.nan

    for sequence_id, seq_df in out.groupby("sequence_id"):
        idx = seq_df.index

        for angle in ["yaw", "roll", "pitch"]:
            filtered = apply_single_axis_filter(
                seq_df,
                angle=angle,
                filter_name=axis_config[angle],
                validation_means=validation_means
            )

            out.loc[idx, f"{output_prefix}_{angle}"] = filtered

    return out


def build_candidate_prediction_df(base_df, candidate_name, candidate_configs, best_alpha):
    """
    Return a dataframe containing:
    - GT
    - raw predictions
    - original EMA predictions
    - validation-mean baseline
    - selected robust-filter predictions
    """
    out = base_df.copy().sort_values(["sequence_id", "frame_id"]).reset_index(drop=True)
    
    # Validation-mean baseline
    validation_df_local = out[out["split"] == "validation"].copy()
    
    mean_yaw_val = circular_mean_deg(validation_df_local["yaw"])
    mean_roll_val = circular_mean_deg(validation_df_local["roll"])
    mean_pitch_val = circular_mean_deg(validation_df_local["pitch"])
    
    out["valmean_yaw"] = mean_yaw_val
    out["valmean_roll"] = mean_roll_val
    out["valmean_pitch"] = mean_pitch_val
    
    if candidate_name == "raw predictions":
        for angle in ["yaw", "roll", "pitch"]:
            out[f"bestrobust_pred_{angle}"] = out[f"pred_{angle}"]
    
    elif candidate_name == f"original EMA alpha={best_alpha}":
        for angle in ["yaw", "roll", "pitch"]:
            out[f"bestrobust_pred_{angle}"] = out[f"smooth_pred_{angle}"]
    
    else:
        config = get_candidate_config_by_name(candidate_name, candidate_configs)
        if config is None:
            raise ValueError(f"Could not find config for candidate: {candidate_name}")
        
        robust_df = apply_temporal_filter_by_sequence(
            out,
            method=config["method"],
            params=config["params"],
            output_prefix="bestrobust_pred"
        )
        
        for angle in ["yaw", "roll", "pitch"]:
            out[f"bestrobust_pred_{angle}"] = robust_df[f"bestrobust_pred_{angle}"].values
    
    return out


def get_candidate_config_by_name(candidate_name, candidate_configs):
    for config in candidate_configs:
        if config["name"] == candidate_name:
            return config
    return None


def axis_config_name(axis_config):
    return (
        f"yaw={axis_config['yaw']} | "
        f"roll={axis_config['roll']} | "
        f"pitch={axis_config['pitch']}"
    )


def attach_pair_outlier_flags(temporal_df, frame_flags):
    temp = temporal_df.copy()
    
    temp = temp.merge(
        frame_flags.rename(columns={"has_extreme_raw_angle": "current_extreme_raw_angle"}),
        on=["sequence_id", "frame_id"],
        how="left"
    )
    
    temp = temp.merge(
        frame_flags.rename(columns={
            "frame_id": "prev_frame_id_eval",
            "has_extreme_raw_angle": "previous_extreme_raw_angle"
        }),
        on=["sequence_id", "prev_frame_id_eval"],
        how="left"
    )
    
    temp["current_extreme_raw_angle"] = temp["current_extreme_raw_angle"].fillna(False)
    temp["previous_extreme_raw_angle"] = temp["previous_extreme_raw_angle"].fillna(False)
    
    temp["pair_has_extreme_raw_angle"] = (
        temp["current_extreme_raw_angle"] |
        temp["previous_extreme_raw_angle"]
    )
    
    return temp


# ----------------------------------------------------------------------------
# Metric summaries and reporting helpers
# ----------------------------------------------------------------------------

def summarize_frame_metrics(dataframe, split_name):
    subset = dataframe[dataframe["split"] == split_name]
    
    metrics = {
        "split": split_name,
        "num_frames": len(subset),
        "mae_yaw": subset["err_yaw"].mean(),
        "mae_roll": subset["err_roll"].mean(),
        "mae_pitch": subset["err_pitch"].mean(),
        "mean_mae": subset[["err_yaw", "err_roll", "err_pitch"]].mean().mean(),
        "rmse_yaw": np.sqrt(np.mean(subset["err_yaw"] ** 2)),
        "rmse_roll": np.sqrt(np.mean(subset["err_roll"] ** 2)),
        "rmse_pitch": np.sqrt(np.mean(subset["err_pitch"] ** 2)),
        "mean_rmse": np.mean([
            np.sqrt(np.mean(subset["err_yaw"] ** 2)),
            np.sqrt(np.mean(subset["err_roll"] ** 2)),
            np.sqrt(np.mean(subset["err_pitch"] ** 2)),
        ]),
    }
    
    return metrics


def summarize_temporal_metrics(dataframe, split_name):
    subset = dataframe[dataframe["split"] == split_name]
    
    return {
        "split": split_name,
        "num_pairs": len(subset),
        "mean_gt_motion": subset["gt_motion_magnitude_eval"].mean(),
        "mean_pred_jitter": subset["pred_jitter_magnitude"].mean(),
        "mean_temporal_motion_error": subset["temporal_motion_error_magnitude"].mean(),
        "jitter_yaw": subset["pred_jitter_yaw"].mean(),
        "jitter_roll": subset["pred_jitter_roll"].mean(),
        "jitter_pitch": subset["pred_jitter_pitch"].mean(),
        "temporal_error_yaw": subset["temporal_error_yaw"].mean(),
        "temporal_error_roll": subset["temporal_error_roll"].mean(),
        "temporal_error_pitch": subset["temporal_error_pitch"].mean(),
    }


def summarize_smoothed_metrics(dataframe, temporal_dataframe, alpha, split_name):
    frame_subset = dataframe[dataframe["split"] == split_name]
    temporal_subset = temporal_dataframe[temporal_dataframe["split"] == split_name]
    
    return {
        "alpha": alpha,
        "split": split_name,
        "num_frames": len(frame_subset),
        "num_pairs": len(temporal_subset),
        "smooth_mae_yaw": frame_subset["smooth_err_yaw"].mean(),
        "smooth_mae_roll": frame_subset["smooth_err_roll"].mean(),
        "smooth_mae_pitch": frame_subset["smooth_err_pitch"].mean(),
        "smooth_mean_mae": frame_subset[[
            "smooth_err_yaw",
            "smooth_err_roll",
            "smooth_err_pitch"
        ]].mean().mean(),
        "smooth_mean_jitter": temporal_subset["smooth_pred_jitter_magnitude"].mean(),
        "smooth_mean_temporal_motion_error": temporal_subset["smooth_pred_temporal_motion_error_magnitude"].mean(),
    }


def summarize_raw_vs_smooth_frame(dataframe, split_name):
    subset = dataframe[dataframe["split"] == split_name]
    
    return {
        "split": split_name,
        "num_frames": len(subset),
        "raw_mae_yaw": subset["err_yaw"].mean(),
        "raw_mae_roll": subset["err_roll"].mean(),
        "raw_mae_pitch": subset["err_pitch"].mean(),
        "raw_mean_mae": subset[["err_yaw", "err_roll", "err_pitch"]].mean().mean(),
        "smooth_mae_yaw": subset["smooth_err_yaw"].mean(),
        "smooth_mae_roll": subset["smooth_err_roll"].mean(),
        "smooth_mae_pitch": subset["smooth_err_pitch"].mean(),
        "smooth_mean_mae": subset[["smooth_err_yaw", "smooth_err_roll", "smooth_err_pitch"]].mean().mean(),
        "delta_mean_mae": (
            subset[["smooth_err_yaw", "smooth_err_roll", "smooth_err_pitch"]].mean().mean()
            -
            subset[["err_yaw", "err_roll", "err_pitch"]].mean().mean()
        )
    }


def summarize_raw_vs_smooth_temporal(raw_temporal, smooth_temporal, split_name):
    raw_subset = raw_temporal[raw_temporal["split"] == split_name]
    smooth_subset = smooth_temporal[smooth_temporal["split"] == split_name]
    
    return {
        "split": split_name,
        "num_pairs_raw": len(raw_subset),
        "num_pairs_smooth": len(smooth_subset),
        "raw_mean_jitter": raw_subset["pred_jitter_magnitude"].mean(),
        "smooth_mean_jitter": smooth_subset["smooth_pred_jitter_magnitude"].mean(),
        "delta_jitter": (
            smooth_subset["smooth_pred_jitter_magnitude"].mean()
            -
            raw_subset["pred_jitter_magnitude"].mean()
        ),
        "jitter_reduction_percent": (
            100.0 *
            (raw_subset["pred_jitter_magnitude"].mean() - smooth_subset["smooth_pred_jitter_magnitude"].mean())
            / raw_subset["pred_jitter_magnitude"].mean()
        ),
        "raw_temporal_motion_error": raw_subset["temporal_motion_error_magnitude"].mean(),
        "smooth_temporal_motion_error": smooth_subset["smooth_pred_temporal_motion_error_magnitude"].mean(),
        "delta_temporal_motion_error": (
            smooth_subset["smooth_pred_temporal_motion_error_magnitude"].mean()
            -
            raw_subset["temporal_motion_error_magnitude"].mean()
        ),
        "temporal_error_reduction_percent": (
            100.0 *
            (raw_subset["temporal_motion_error_magnitude"].mean() - smooth_subset["smooth_pred_temporal_motion_error_magnitude"].mean())
            / raw_subset["temporal_motion_error_magnitude"].mean()
        ),
    }


def compare_temporal_by_group(raw_df, smooth_df, group_col, split_name):
    raw_subset = raw_df[raw_df["split"] == split_name].copy()
    smooth_subset = smooth_df[smooth_df["split"] == split_name].copy()
    
    raw_grouped = (
        raw_subset
        .groupby(group_col)
        .agg(
            num_pairs=("frame_id", "count"),
            raw_mean_gt_motion=("gt_motion_magnitude_eval", "mean"),
            raw_mean_jitter=("pred_jitter_magnitude", "mean"),
            raw_temporal_motion_error=("temporal_motion_error_magnitude", "mean"),
        )
        .reset_index()
    )
    
    smooth_grouped = (
        smooth_subset
        .groupby(group_col)
        .agg(
            smooth_mean_jitter=("smooth_pred_jitter_magnitude", "mean"),
            smooth_temporal_motion_error=("smooth_pred_temporal_motion_error_magnitude", "mean"),
        )
        .reset_index()
    )
    
    comparison = raw_grouped.merge(smooth_grouped, on=group_col, how="inner")
    
    comparison["jitter_reduction_percent"] = (
        100.0 *
        (comparison["raw_mean_jitter"] - comparison["smooth_mean_jitter"])
        / comparison["raw_mean_jitter"]
    )
    
    comparison["temporal_error_reduction_percent"] = (
        100.0 *
        (comparison["raw_temporal_motion_error"] - comparison["smooth_temporal_motion_error"])
        / comparison["raw_temporal_motion_error"]
    )
    
    comparison["split"] = split_name
    
    return comparison


def summarize_candidate(candidate_df, candidate_temporal_df, pred_prefix, error_prefix, candidate_name, split_name):
    frame_subset = candidate_df[candidate_df["split"] == split_name]
    temporal_subset = candidate_temporal_df[candidate_temporal_df["split"] == split_name]
    
    return {
        "candidate": candidate_name,
        "split": split_name,
        "num_frames": len(frame_subset),
        "num_pairs": len(temporal_subset),
        "mae_yaw": frame_subset[f"{error_prefix}_yaw"].mean(),
        "mae_roll": frame_subset[f"{error_prefix}_roll"].mean(),
        "mae_pitch": frame_subset[f"{error_prefix}_pitch"].mean(),
        "mean_mae": frame_subset[f"{error_prefix}_mean"].mean(),
        "mean_jitter": temporal_subset[f"{pred_prefix}_jitter_magnitude"].mean(),
        "mean_temporal_motion_error": temporal_subset[f"{pred_prefix}_temporal_motion_error_magnitude"].mean(),
    }


def summarize_aflw_subset(dataframe, label):
    return {
        "subset": label,
        "num_frames": len(dataframe),
        "mae_yaw": dataframe["err_yaw"].mean(),
        "mae_roll": dataframe["err_roll"].mean(),
        "mae_pitch": dataframe["err_pitch"].mean(),
        "mean_mae": dataframe[["err_yaw", "err_roll", "err_pitch"]].mean().mean(),
        "rmse_yaw": np.sqrt(np.mean(dataframe["err_yaw"] ** 2)),
        "rmse_roll": np.sqrt(np.mean(dataframe["err_roll"] ** 2)),
        "rmse_pitch": np.sqrt(np.mean(dataframe["err_pitch"] ** 2)),
        "mean_rmse": np.mean([
            np.sqrt(np.mean(dataframe["err_yaw"] ** 2)),
            np.sqrt(np.mean(dataframe["err_roll"] ** 2)),
            np.sqrt(np.mean(dataframe["err_pitch"] ** 2)),
        ]),
    }


def evaluate_prediction_columns(dataframe, pred_yaw_col, pred_roll_col, pred_pitch_col, label):
    """Evaluate MAE/RMSE for arbitrary prediction columns."""
    err_yaw = angular_diff_array_deg(dataframe[pred_yaw_col], dataframe["yaw"])
    err_roll = angular_diff_array_deg(dataframe[pred_roll_col], dataframe["roll"])
    err_pitch = angular_diff_array_deg(dataframe[pred_pitch_col], dataframe["pitch"])
    
    return {
        "method": label,
        "num_frames": len(dataframe),
        "mae_yaw": err_yaw.mean(),
        "mae_roll": err_roll.mean(),
        "mae_pitch": err_pitch.mean(),
        "mean_mae": np.mean([err_yaw.mean(), err_roll.mean(), err_pitch.mean()]),
        "rmse_yaw": np.sqrt(np.mean(err_yaw ** 2)),
        "rmse_roll": np.sqrt(np.mean(err_roll ** 2)),
        "rmse_pitch": np.sqrt(np.mean(err_pitch ** 2)),
        "mean_rmse": np.mean([
            np.sqrt(np.mean(err_yaw ** 2)),
            np.sqrt(np.mean(err_roll ** 2)),
            np.sqrt(np.mean(err_pitch ** 2)),
        ]),
    }


def axis_mae_table(dataframe, label, methods):
    rows = []
    
    for method_name, (yaw_col, roll_col, pitch_col) in methods.items():
        rows.append({
            "subset": label,
            "method": method_name,
            "mae_yaw": angular_diff_array_deg(dataframe[yaw_col], dataframe["yaw"]).mean(),
            "mae_roll": angular_diff_array_deg(dataframe[roll_col], dataframe["roll"]).mean(),
            "mae_pitch": angular_diff_array_deg(dataframe[pitch_col], dataframe["pitch"]).mean(),
        })
    
    out = pd.DataFrame(rows)
    out["mean_mae"] = out[["mae_yaw", "mae_roll", "mae_pitch"]].mean(axis=1)
    return out


def summarize_frame_outlier_subset(dataframe, subset_name):
    return {
        "subset": subset_name,
        "num_frames": len(dataframe),
        "extreme_raw_angle_rate_percent": (
            100.0 * dataframe["has_extreme_raw_angle"].mean()
            if len(dataframe) > 0 else np.nan
        ),
        "raw_mean_mae": dataframe["err_mean"].mean(),
        "smooth_mean_mae": dataframe["smooth_err_mean"].mean(),
        "delta_smooth_minus_raw_mae": dataframe["smooth_err_mean"].mean() - dataframe["err_mean"].mean(),
        "raw_mae_yaw": dataframe["err_yaw"].mean(),
        "raw_mae_roll": dataframe["err_roll"].mean(),
        "raw_mae_pitch": dataframe["err_pitch"].mean(),
        "smooth_mae_yaw": dataframe["smooth_err_yaw"].mean(),
        "smooth_mae_roll": dataframe["smooth_err_roll"].mean(),
        "smooth_mae_pitch": dataframe["smooth_err_pitch"].mean(),
    }


def summarize_temporal_outlier_subset(raw_df, smooth_df, mask_function, subset_name):
    raw_subset = raw_df[mask_function(raw_df)]
    smooth_subset = smooth_df[mask_function(smooth_df)]
    
    raw_jitter = raw_subset["pred_jitter_magnitude"].mean()
    smooth_jitter = smooth_subset["smooth_pred_jitter_magnitude"].mean()
    
    raw_temporal_error = raw_subset["temporal_motion_error_magnitude"].mean()
    smooth_temporal_error = smooth_subset["smooth_pred_temporal_motion_error_magnitude"].mean()
    
    return {
        "subset": subset_name,
        "num_raw_pairs": len(raw_subset),
        "num_smooth_pairs": len(smooth_subset),
        "raw_mean_jitter": raw_jitter,
        "smooth_mean_jitter": smooth_jitter,
        "jitter_reduction_percent": 100.0 * (raw_jitter - smooth_jitter) / raw_jitter,
        "raw_temporal_motion_error": raw_temporal_error,
        "smooth_temporal_motion_error": smooth_temporal_error,
        "temporal_error_reduction_percent": (
            100.0 * (raw_temporal_error - smooth_temporal_error) / raw_temporal_error
        ),
    }


# ----------------------------------------------------------------------------
# Other utilities
# ----------------------------------------------------------------------------

def check_url_available(url, timeout=10):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return True, response.status
    except Exception as e:
        return False, str(e)
# ============================================================================
# Geodesic (SO(3)) rotation error  [added for substantive fix #2]
# ----------------------------------------------------------------------------
# 6DRepNet360 regresses a rotation matrix and is trained with a geodesic loss
# on SO(3). Evaluating it through Euler angles reintroduces the wrap-around and
# axis-coupling that the rotation representation exists to avoid (large roll/
# pitch errors and "jitter" near the +/-180 boundary). We therefore also report
# the geodesic error, which is the faithful, bounded (0-180 deg) rotation metric.
# Rotation matrices are built with the exact repository convention
# (utils.get_R): R = Rz(roll) @ Ry(yaw) @ Rx(pitch), with x=pitch, y=yaw, z=roll.
# ============================================================================

def rotation_matrices_from_euler_deg(yaw, roll, pitch):
    """Vectorized rotation matrices from yaw/roll/pitch in degrees.

    Uses the 6DRepNet360 convention R = Rz(roll) @ Ry(yaw) @ Rx(pitch), which is
    the exact inverse of compute_euler_angles_from_rotation_matrices. Accepts
    scalars or arrays; returns an array of shape (..., 3, 3)."""
    x = np.deg2rad(np.asarray(pitch, dtype=float))  # Rx
    y = np.deg2rad(np.asarray(yaw, dtype=float))    # Ry
    z = np.deg2rad(np.asarray(roll, dtype=float))   # Rz
    x, y, z = np.broadcast_arrays(x, y, z)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    R = np.empty(x.shape + (3, 3), dtype=float)
    R[..., 0, 0] = cy * cz
    R[..., 0, 1] = sx * sy * cz - cx * sz
    R[..., 0, 2] = cx * sy * cz + sx * sz
    R[..., 1, 0] = cy * sz
    R[..., 1, 1] = sx * sy * sz + cx * cz
    R[..., 1, 2] = cx * sy * sz - sx * cz
    R[..., 2, 0] = -sy
    R[..., 2, 1] = sx * cy
    R[..., 2, 2] = cx * cy
    return R

def geodesic_error_deg(R_a, R_b):
    """Geodesic distance on SO(3) in degrees, in [0, 180]. Batched over (...,3,3)."""
    Rrel = np.matmul(np.swapaxes(R_a, -1, -2), R_b)
    trace = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))

def geodesic_errors_from_euler_deg(pred_yaw, pred_roll, pred_pitch,
                                   gt_yaw, gt_roll, gt_pitch):
    """Per-sample geodesic error (deg) between predicted and ground-truth poses."""
    R_pred = rotation_matrices_from_euler_deg(pred_yaw, pred_roll, pred_pitch)
    R_gt = rotation_matrices_from_euler_deg(gt_yaw, gt_roll, gt_pitch)
    return geodesic_error_deg(R_pred, R_gt)

def add_geodesic_frame_error(dataframe, pred_prefix="pred", out_col=None):
    """Add a per-frame geodesic error column for the given prediction columns
    ('<pred_prefix>_yaw/roll/pitch') against the ground-truth 'yaw/roll/pitch'."""
    out = dataframe.copy()
    if out_col is None:
        out_col = f"geo_err_{pred_prefix}"
    out[out_col] = geodesic_errors_from_euler_deg(
        out[f"{pred_prefix}_yaw"], out[f"{pred_prefix}_roll"], out[f"{pred_prefix}_pitch"],
        out["yaw"], out["roll"], out["pitch"],
    )
    return out

def add_geodesic_temporal_columns(temporal_dataframe, pred_prefix="pred"):
    """Add geodesic frame-to-frame columns on a consecutive-pair table that
    already carries current and previous angles ('<p>_yaw' and 'prev_<p>_yaw',
    plus 'yaw'/'prev_yaw'). Adds:
      geo_jitter           : geodesic angle between consecutive predicted rotations
      geo_gt_motion        : geodesic angle between consecutive GT rotations
      geo_temporal_error   : |geo_jitter - geo_gt_motion|
    All are bounded to [0, 180] deg, unlike the Euler L2 jitter."""
    out = temporal_dataframe.copy()
    R_cur_pred = rotation_matrices_from_euler_deg(
        out[f"{pred_prefix}_yaw"], out[f"{pred_prefix}_roll"], out[f"{pred_prefix}_pitch"])
    R_prev_pred = rotation_matrices_from_euler_deg(
        out[f"prev_{pred_prefix}_yaw"], out[f"prev_{pred_prefix}_roll"], out[f"prev_{pred_prefix}_pitch"])
    R_cur_gt = rotation_matrices_from_euler_deg(out["yaw"], out["roll"], out["pitch"])
    R_prev_gt = rotation_matrices_from_euler_deg(out["prev_yaw"], out["prev_roll"], out["prev_pitch"])
    out["geo_jitter"] = geodesic_error_deg(R_cur_pred, R_prev_pred)
    out["geo_gt_motion"] = geodesic_error_deg(R_cur_gt, R_prev_gt)
    out["geo_temporal_error"] = np.abs(out["geo_jitter"] - out["geo_gt_motion"])
    return out


# ----------------------------------------------------------------------------
# Fisheye location encoding  [added for location-guided fine-tuning, Appendix D]
# ----------------------------------------------------------------------------

def location_features_deg(rho, theta_deg):
    """Encode the fisheye location as [rho, sin(theta), cos(theta)] (theta in
    degrees). Using sin/cos avoids the +/-180 discontinuity of the raw angle."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    rho = np.asarray(rho, dtype=float)
    return np.stack([rho, np.sin(theta), np.cos(theta)], axis=-1)
