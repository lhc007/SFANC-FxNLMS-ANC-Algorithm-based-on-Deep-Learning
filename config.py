# ANC 配置 - 次级路径测量参数
# 修复版：统一参数命名，优化默认值

anc_config = {
    
    # ========================= 基本音频参数 =========================
    'fs': 48000,                    # 采样率（Hz）
    'debug': True,                  # 调试模式

    # ========================= 扫频信号参数 =========================
    'sweepFreqStartHz': 50,         # 扫频起始频率（Hz）
    'sweepFreqEndHz': 1500,         # 扫频结束频率（Hz）
    'sweepDuration': 5,             # 扫频时长（秒）- 较长的扫频时间可提高 SNR
    'amplitude': 0.8,               # 扫频信号幅值 (0-1)
    
    # 前后静音时间（统一使用这两个参数）
    'ante_silence': 0.5,            # 前静音时长（秒）- 让系统稳定
    'post_silence': 0.5,            # 后静音时长（秒）- 捕获混响尾音
    
    # 逆滤波器参数
    'searchRadiusMs': 200,          # 逆滤波器搜索半径（毫秒）

    # ========================= FxNLMS算法参数 =========================
    'mu': 0.01,                  # FxNLMS算法的学习率
    'delta': 0.01,               # FxNLMS算法的步长
    'classify_interval': 16,      # 分类间隔（秒）- 每 16 帧重新分类一次（约 1 秒）


    # ========================= 硬件设备配置 =========================
    'micDeviceName': '六通道麦克风阵列 (YDM6MIC Audio)',
    'spkDeviceName': '扬声器2 (Realtek(R) Audio)',
    
    # ========================= 音频流参数 =========================
    # 麦克风通道配置
    'micNumChannels': 6,            # 麦克风阵列总通道数
    'micChannels': {
        'reference': [1],  # 参考麦克风通道 (1-based)
        'error': [2, 3, 4,5, 6]             # 误差麦克风通道 (1-based)
    },
    
    # 扬声器通道配置
    'spkChannels': [1, 2],          # 扬声器通道 (1-based)
    
    # 音频流参数
    'frameSize': 1024,              # 音频块大小（采样点）- 较大的块可避免 underflow

    # ========================= 测量参数 =========================
    'filePath': "secondary_path_irs.npz",   # 输出文件路径
    'irLengthSec': 0.5,             # 临时 IR 存储长度（秒）
    'irLengthTargetMs': 500,        # 目标 IR 长度（毫秒）
    'numAverages': 3,               # 平均次数 - 建议至少 3 次以提高 SNR

    # ========================= 质量评估参数 =========================
    'target_freq_range': (50, 1500),    # 目标频率范围（Hz）
    'snr_threshold': 20.0,              # SNR 最低要求 (dB)
    'flatness_threshold': 3.0,          # 频响平坦度最大值 (dB)
    'causality_threshold': 0.01,        # 因果性比值最大值
    'min_ir_length_ms': 80,             # 最小期望 IR 长度（毫秒）
    'min_gain_threshold': -30.0,        # 目标频带内最小增益阈值 (dB)
    'phase_smoothness_threshold': 0.5,  # 相位差分标准差阈值 (rad)
    'channel_similarity_threshold': 0.95,  # 通道间相关系数阈值
    
    # 可视化参数
    'save_plots': False,                # 是否保存图像
    'outputDir': "analysis",            # 图像输出目录
    'plot_individual': True,            # 是否绘制单独图像
    'array_dim_spec': None,             # 数据维度说明
    'expected_delays': None,            # 预期延迟（可选）


}