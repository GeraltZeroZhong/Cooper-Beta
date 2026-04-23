import os

class Config:
    # --- Loader ---
    # DSSP executable path. Set this to an absolute path if automatic discovery
    # fails, for example:
    # "/usr/bin/mkdssp" or "/home/user/miniconda3/bin/mkdssp"
    DSSP_BIN_PATH = None

    # --- Slicer ---
    # Slice thickness in angstroms.
    SLICE_STEP_SIZE = 1.0

    # --- Analyzer (Elliptical Fitting) ---
    # A general ellipse fit (5 parameters) needs at least 5 points. We use a
    # slightly larger default for stability.
    MIN_POINTS_PER_SLICE = 7

    # Maximum allowed fit error (RMSE, angstroms).
    MAX_FIT_RMSE = 3

    # Allowed ellipse semi-axis range (angstroms).
    MIN_AXIS = 3.0
    MAX_AXIS = 199.0

    # Maximum flattening ratio (major_axis / minor_axis). Very large values mean
    # the slice looks overly compressed and is unlikely to be a barrel.
    MAX_FLATTENING = 3.5

    # --- Robust fitting (least_squares) ---
    LSQ_METHOD = 'trf'         # 'trf' supports robust losses.
    LSQ_LOSS = 'soft_l1'       # You can switch this to 'huber' if needed.
    LSQ_F_SCALE = 1.0          # Scale parameter for the robust loss.

    # --- Decision ---
    # Final decision threshold: classify as BARREL when the valid-slice ratio
    # exceeds BARREL_VALID_RATIO.
    BARREL_VALID_RATIO = 0.5

    # --- Scoring adjustment ---
    # Minimum number of intersections required for a slice to contribute to the
    # score_adjust denominator. Slices with too few points are treated as junk.
    MIN_INTERSECTIONS_FOR_SCORING = 7

    # Whether to use score_adjust as the final decision score.
    USE_ADJUSTED_SCORE = True

    # --- Control condition ---
    # Require enough scored slices; otherwise score_adjust becomes unstable.
    # Condition: all_adjusted_layers > all_layers * MIN_SCORED_LAYER_FRAC
    MIN_SCORED_LAYER_FRAC = 0.20

    # --- Nearest-neighbor spacing rule (optional) ---
    # Uniformity check based on geometric nearest-neighbor distances. Useful for
    # flagging noisy, sparse, or duplicated intersections.
    NN_RULE_ENABLED = True

    # Dispersion threshold for nearest-neighbor distances, using robust CV =
    # (1.4826 * MAD) / median. Smaller is stricter, larger is more permissive.
    NN_MAX_ROBUST_CV = 0.40

    # Minimum inlier fraction, where inliers satisfy
    # |d - median| <= 3 * robust_sigma.
    NN_MIN_INLIER_FRAC = 0.75

    # Whether a slice that fails the NN rule should be excluded from scoring as
    # junk. True tends to favor recall; False tends to favor precision.
    NN_FAIL_AS_JUNK = True

    # --- Angular coverage / gap rule (for jelly-roll rejection) ---
    # Beta-barrel cross sections should cover nearly 360 degrees. Jelly-roll
    # structures often leave large angular gaps.
    ANGLE_RULE_ENABLED = True

    # Maximum allowed angular gap, in degrees. Smaller values are stricter.
    ANGLE_MAX_GAP_DEG = 80

    # --- Sequence-order vs angle-order consistency rule ---
    # For each slice, every intersection has both:
    #   - seq_order: the order induced by segment positions i + 0.5
    #   - angle_order: the order induced by polar angle around the slice center
    # True beta barrels usually advance locally around the circumference as
    # sequence order advances. Jelly-roll / beta-sandwich structures often jump
    # across the opposite side of the slice.
    #
    # local_step: circular distance between adjacent seq_order neighbors in
    # angle_order rank space.
    # local_frac: fraction of local_step values that are within the threshold.
    ANGLE_ORDER_RULE_ENABLED = True

    # Maximum local step that still counts as "locally adjacent" in angle_order.
    ANGLE_ORDER_LOCAL_STEP_MAX = 1

    # Pass condition: local_frac must be at least this threshold.
    ANGLE_ORDER_MIN_LOCAL_FRAC = 1

    # Additional global consistency score, normalized to [0, 1], while allowing
    # circular shifts and reversed direction.
    # 0 means a perfect match to some shift/direction; 1 means maximally inconsistent.
    ANGLE_ORDER_MAX_MEAN_CIRC_DIST_NORM = 0

    # Whether slices that fail the angle rule should be excluded from scoring as
    # junk. Default False keeps them in the denominator and lowers the score,
    # which helps reject jelly-roll-like structures.
    ANGLE_FAIL_AS_JUNK = False
