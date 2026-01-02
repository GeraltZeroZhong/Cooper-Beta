import os

class Config:
    # --- Loader ---
    # DSSP 可执行文件路径 (如果自动查找失败，请在此填入绝对路径)
    # 例如: "/usr/bin/mkdssp" 或 "/home/user/miniconda3/bin/mkdssp"
    DSSP_BIN_PATH = None  
    
    # --- Slicer ---
    # 切片层厚度 (Angstrom)
    SLICE_STEP_SIZE = 1.0
    
    # --- Analyzer (Elliptical Fitting) ---
    # 拟合通用椭圆 (5参数) 至少需要 5 个点，建议设为 6 以增加稳定性
    MIN_POINTS_PER_SLICE = 7     
    
    # 允许的最大拟合误差 (RMSE, Å)
    MAX_FIT_RMSE = 3       
    
    # 椭圆半轴长范围 (Å)
    # 对应 min_axis 和 max_axis
    MIN_AXIS = 3.0
    MAX_AXIS = 199.0
    
    # 最大扁平率 (长轴/短轴)
    # 如果该值过大 (如 > 3.0)，说明切面被压得极扁，可能不是桶
    MAX_FLATTENING = 3.5

    # --- Robust fitting (least_squares) ---
    LSQ_METHOD = 'trf'         # 'trf' 支持鲁棒 loss
    LSQ_LOSS = 'soft_l1'       # 可改为 'huber'
    LSQ_F_SCALE = 1.0          # 鲁棒损失的尺度参数
    
    # --- Decision ---
    # 最终判定阈值：有效切片占比 > VALID_RATIO 即认为是 Barrel
    BARREL_VALID_RATIO = 0.5

    # --- Scoring adjustment ---
    # 计入 score_adjust 分母的最小交点数（点数太少的切片视为 junk，不计入分母）
    MIN_INTERSECTIONS_FOR_SCORING = 7

    # 是否用 score_adjust 作为最终判定分数
    USE_ADJUSTED_SCORE = True

    # --- Control condition ---
    # 要求 score_adjust 的分母（all_adjusted_layers）不能太小，否则分数不稳定
    # 条件：all_adjusted_layers > all_layers * MIN_SCORED_LAYER_FRAC
    MIN_SCORED_LAYER_FRAC = 0.20
    # --- Nearest-neighbor spacing rule (optional) ---
    # 基于几何最近邻距离的“均匀性”检验：用于识别噪声/离散/重复点导致的异常切片
    NN_RULE_ENABLED = True

    # 最近邻距离的离散程度阈值：使用 robust CV = (1.4826*MAD) / median
    # 值越小越严格（更偏 precision），值越大越宽松（更偏 recall）
    NN_MAX_ROBUST_CV = 0.40

    # 允许的离群点比例：inlier 定义为 |d - median| <= 3 * robust_sigma
    NN_MIN_INLIER_FRAC = 0.75

    # 若切片不通过 NN 规则，是否视为 junk（不计入 score_adjust 分母）
    # True：更偏 recall；False：更偏 precision（算作无效层，会拉低分数）
    NN_FAIL_AS_JUNK = True


    # --- Angular coverage / gap rule (for jelly-roll rejection) ---
    # 基于截面点的角度分布：beta-barrel 截面应接近 360° 覆盖，jelly-roll 常出现大缺口
    ANGLE_RULE_ENABLED = True

    # 允许的最大角度缺口（度）。越小越严格（更偏 precision）
    ANGLE_MAX_GAP_DEG = 80

    # --- New: seq_order vs angle_order consistency rule ---
    # 同一切片内，每个交点同时具备：
    #   - seq_order：由线段 (i, i+1) 的序列位置 i+0.5 排序得到
    #   - angle_order：以切片中心为极点，按角度 [0,360) 排序得到
    # 真正 β-barrel 的交点沿序列推进时，通常也会沿圆周“近邻”推进；
    # jelly-roll / β-sandwich 常出现跨对侧的大跳跃。
    #
    # local_step：相邻 seq_order 在 angle_order 上的圆周距离（以“名次数”计，环状取最小距离）。
    # local_frac：local_step <= ANGLE_ORDER_LOCAL_STEP_MAX 的比例。
    ANGLE_ORDER_RULE_ENABLED = True

    # 判定“局部邻近”的最大步长（以 angle_order 名次数计）。
    ANGLE_ORDER_LOCAL_STEP_MAX = 1

    # 通过条件：local_frac >= 该阈值。值越大越严格。
    ANGLE_ORDER_MIN_LOCAL_FRAC = 1

    # 额外：全局顺序一致性（考虑环状 shift 与方向翻转），归一化到 [0,1]。
    # 0 表示与某个环状 shift / 方向完全一致；1 表示完全不一致。
    ANGLE_ORDER_MAX_MEAN_CIRC_DIST_NORM = 0

    # 角度规则失败时是否视为 junk（不计入 score_adjust 分母）
    # 为了更好剔除 jelly-roll，默认 False（失败计入分母并拉低分数）
    ANGLE_FAIL_AS_JUNK = False
