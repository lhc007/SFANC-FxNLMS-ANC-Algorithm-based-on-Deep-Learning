import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
"""
生成模拟房间脉冲响应(次级路径)。
"""
# ================== 1. 设置仿真参数 ==================
FS = 16000                  # 提高采样率以覆盖论文中的 1000Hz 高频分析
C = 343.0                   # 声速 (m/s)
RIR_LENGTH = 8192           # 增加长度以容纳房间反射 (RT60 ~0.4s)

# --- 论文参数定义 (参考 Figure 1 & Section III) ---
WINDOW_WIDTH_M = 0.96       # 论文图1标注 960mm
WINDOW_HEIGHT_M = 0.65      # 论文图1标注 650mm

# ================== 2. 定义设备位置 (5-Mic Array) ==================
# 坐标系: x轴垂直于窗户, y轴水平, z轴垂直
# 原点: 窗户中心

# ---- 次级扬声器 (4个) - 论文最佳集群 (Section IV.B.1) ----
# 位于室外开口的角落和边缘 (S1, S2, S5, S8)
SPK_POSITIONS  = np.array([
    # X(m)    Y(m)    Z(m)     # 对应论文标签
    [0.0,   -0.40,   -0.25],   # S1: 左下角
    [0.0,    0.40,   -0.25],   # S2: 右下角
    [0.0,   -0.40,    0.25],   # S5: 左上角 
    [0.0,    0.40,    0.25]    # S8: 右上角
])

# ---- 误差麦克风 (5个) - 适配你的硬件限制 ----
# 策略: 采用 "中心十字形" 或 "紧凑网格" 布局
# 理由: 论文 (Table III) 显示，覆盖中心区域 ([e6, e7, e10]) 是最有效的。
#       我们将 5 个麦克风布置在中心 20cm x 20cm 的区域内。
MIC_POSITIONS  = np.array([
    # X(m)    Y(m)    Z(m)
    [0.05,   -0.10,   -0.10],  # Mic 1: 左下
    [0.05,    0.10,   -0.10],  # Mic 2: 右下 
    [0.05,   -0.10,    0.10],  # Mic 3: 左上
    [0.05,    0.10,    0.10],  # Mic 4: 右上
    [0.05,    0.00,    0.00]   # Mic 5: 正中心 (核心)
])

# ================== 3. 高级 RIR 生成函数 (环境模拟) ==================
def get_rir_paper_env(src_pos, mic_pos, fs, c, length, rt60=0.4):
    """
    生成模拟实验环境的脉冲响应。
    包含：1. 自由场直达声 2. 窗户边缘衍射 (Diffraction) 3. 房间混响 (Reverberation)
    """
    # 1. 直达声 (Direct Path)
    dist = np.linalg.norm(mic_pos - src_pos)
    n_direct = int((dist / c) * fs)
    amp_direct = 1.0 / (dist + 1e-8)
    
    h = np.zeros(length)
    if n_direct < length:
        h[n_direct] = amp_direct

    # 2. 边缘衍射模拟 (Edge Diffraction - Plenum Window Effect)
    # 论文中的声音需要绕过窗框，产生轻微散射
    diffraction_delay = n_direct + int(0.3 * fs / 1000) # ~0.3ms 延迟
    if diffraction_delay < length:
        h[diffraction_delay] += amp_direct * 0.15 # 论文环境中的绕射能量

    # 3. 房间混响 (Schroeder Model Approximation)
    # 根据论文 Fig. 4(a) Reverberation time (~0.4 seconds)
    if rt60 > 0:
        n_reverb = int(rt60 * fs)
        if n_reverb < length:
            # 生成白噪声并应用指数衰减包络
            tail = np.random.normal(0, 1, n_reverb) 
            decay = np.exp(np.linspace(0, -6, n_reverb)) # 指数衰减
            energy = 0.05 # 混响能量系数
            start_idx = n_direct + 50
            end_idx = start_idx + n_reverb
            if end_idx < length:
                h[start_idx:end_idx] += tail * decay * energy

    return h

# ================== 4. 次级路径生成函数 ==================
def generate_secondary_path(spk_positions, mic_positions, fs, c, rir_length,
                            save_path='secondary_path_4spk_5mic.npy'):
    """
    生成次级路径脉冲响应矩阵。
    形状: (扬声器数 J, 误差麦克风数 K, 脉冲响应长度 L)
    """
    num_spks = spk_positions.shape[0]
    num_mics = mic_positions.shape[0]
    print(f"\n正在生成次级路径 (J={num_spks}, K={num_mics}, L={rir_length}) ...")

    S_matrix = np.zeros((num_spks, num_mics, rir_length))
    for spk_idx in range(num_spks):
        for mic_idx in range(num_mics):
            rir = get_rir_paper_env(
                spk_positions[spk_idx],
                mic_positions[mic_idx],
                fs, c, rir_length
            )
            S_matrix[spk_idx, mic_idx, :] = rir

    np.save(save_path, S_matrix)
    print(f"次级路径已保存至: {save_path}")
    return S_matrix

# ================== 5. 主路径生成函数 ==================
def generate_primary_path(I, mic_positions, fs, c, rir_length,
                          source_distance=5.0, angle_deg=0,
                          save_path='primary_path_1ref_5mic.npy'):
    """
    生成主路径脉冲响应矩阵（模拟室外平面波或点声源入射）。
    形状: (参考通道数 I, 误差麦克风数 K, 脉冲响应长度 L)
    """
    K = mic_positions.shape[0]
    print(f"\n正在生成主路径 (I={I}, K={K}, L={rir_length}) ...")

    pri_path = np.zeros((I, K, rir_length), dtype=np.float32)

    for i in range(I):
        angle = angle_deg + i * 30   # 不同参考通道可对应不同入射方向
        for k in range(K):
            mic_y, mic_z = mic_positions[k, 1], mic_positions[k, 2]
            dir_vec = np.array([0, np.sin(np.radians(angle)), np.cos(np.radians(angle))])
            projection_delay = np.dot([0, mic_y, mic_z], dir_vec) / c
            total_delay = source_distance / c + projection_delay
            n_delay = int(total_delay * fs)

            amplitude = 1.0 / (source_distance + 1e-8)

            if n_delay < rir_length:
                pri_path[i, k, n_delay] = amplitude

            # 绕射延迟
            diff_delay = n_delay + int(0.5 * fs / 1000)
            if diff_delay < rir_length:
                pri_path[i, k, diff_delay] += amplitude * 0.2

            # 简单混响尾部
            reverb_len = int(0.2 * fs)
            if n_delay + 100 < rir_length:
                tail = np.random.randn(reverb_len) * 0.01 * amplitude
                tail *= np.exp(-np.linspace(0, 5, reverb_len))
                end_idx = min(n_delay + 100 + reverb_len, rir_length)
                pri_path[i, k, n_delay+100:end_idx] += tail[:end_idx-(n_delay+100)]

    np.save(save_path, pri_path)
    print(f"主路径已保存至: {save_path}")
    return pri_path

# ================== 6. 可视化函数 ==================
def visualize_paths(spk_positions, mic_positions, S_matrix, Pri_path):
    """绘制空间布局与路径波形"""
    plt.figure(figsize=(15, 5))

    # 子图1: 空间布局 (Y-Z平面)
    plt.subplot(1, 3, 1)
    win_y = [-WINDOW_WIDTH_M/2, WINDOW_WIDTH_M/2, WINDOW_WIDTH_M/2,
             -WINDOW_WIDTH_M/2, -WINDOW_WIDTH_M/2]
    win_z = [-WINDOW_HEIGHT_M/2, -WINDOW_HEIGHT_M/2, WINDOW_HEIGHT_M/2,
             WINDOW_HEIGHT_M/2, -WINDOW_HEIGHT_M/2]
    plt.plot(win_y, win_z, 'k--', alpha=0.5, label='Window Frame')
    plt.scatter(spk_positions[:, 1], spk_positions[:, 2],
                c='red', s=150, marker='^', label='Speakers (4)')
    plt.scatter(mic_positions[:, 1], mic_positions[:, 2],
                c='blue', s=100, marker='s', label='Mics (5)')
    plt.title('Top View: Spatial Layout (Y-Z Plane)')
    plt.xlabel('Width (Y) [m]')
    plt.ylabel('Height (Z) [m]')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 次级路径示例 (扬声器0 → 中心麦克风4)
    plt.subplot(1, 3, 2)
    plt.plot(S_matrix[0, 4, :1000])
    plt.title('Secondary Path: Speaker S1 → Center Mic')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # 子图3: 主路径示例 (参考通道0 → 中心麦克风4)
    plt.subplot(1, 3, 3)
    plt.plot(Pri_path[0, 4, :1000])
    plt.title('Primary Path: Noise Source → Center Mic')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ================== 7. 主函数 ==================
def main():
    # 生成次级路径
    S_matrix = generate_secondary_path(
        SPK_POSITIONS, MIC_POSITIONS, FS, C, RIR_LENGTH,
        save_path='Primary and Secondary Path/secondary_path_4spk_5mic.npy'
    )

    # 生成主路径
    Pri_path = generate_primary_path(
        I=1,
        mic_positions=MIC_POSITIONS,
        fs=FS, c=C, rir_length=RIR_LENGTH,
        source_distance=5.0, angle_deg=0,
        save_path='Primary and Secondary Path/primary_path_1ref_5mic.npy'
    )

    # 可视化
    visualize_paths(SPK_POSITIONS, MIC_POSITIONS, S_matrix, Pri_path)

    print("\n所有路径生成完毕，可用于 Disturbance_generation_from_real_noise_MIMO 函数。")


if __name__ == "__main__":
    main()