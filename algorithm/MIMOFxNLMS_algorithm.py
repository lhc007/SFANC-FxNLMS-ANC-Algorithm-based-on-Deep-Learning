import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar

#------------------------------------------------------------------------------
# 类: MIMOFxNLMS
# 描述: 实现多输入多输出滤波-x归一化最小均方算法，用于多通道主动噪声控制。
#       MIMO = Multiple-Output Multiple-Input
#       I: 参考输入通道数
#       J: 次级声源（控制输出）通道数
#       K: 误差传感器通道数
#------------------------------------------------------------------------------
class MIMOFxNLMS():
    
    def __init__(self, I, J, K, Len, SecPath):
        """
        初始化MIMOFxNLMS算法
        
        参数:
        - I: 参考麦克风数量（输入通道数）
        - J: 次级扬声器数量（输出通道数）
        - K: 误差麦克风数量
        - Len: 自适应滤波器长度
        - SecPath: 次级路径脉冲响应估计，形状为 (K, J, SecLen) 的 torch.Tensor
                   其中 SecLen 为次级路径滤波器长度
        """
        self.I = I
        self.J = J
        self.K = K
        self.Len = Len
        self.SecLen = SecPath.shape[-1]
        
        # 控制滤波器系数矩阵: 形状 (J, I, Len)
        # 初始化全零，requires_grad=True 用于自动微分
        self.Wc = torch.zeros(J, I, Len, dtype=torch.float, requires_grad=True)
        
        # 参考信号延迟线: 形状 (I, Len)
        self.Xd = torch.zeros(I, Len, dtype=torch.float)
        
        # 次级路径估计 (固定，无需梯度)
        self.SecPath = SecPath.to(torch.float)
        
        # 滤波参考信号缓冲区 (用于手动更新时存储)
        # 形状 (K, I, Len) ，对应每个误差通道 k 的滤波参考信号
        self.FiltX = torch.zeros(K, I, Len, dtype=torch.float)
        
    def feedforward(self, x_vec):
        """
        前向传播：计算控制信号
        
        参数:
        - x_vec: 当前时刻的参考信号向量，形状 (I,)
        
        返回:
        - y_vec: 控制信号向量，形状 (J,)
        - power: 各输入通道的信号功率向量，形状 (I,)
        """
        # 更新参考信号延迟线（每个输入通道独立）
        # 将延迟线右移一位，新样本放入最左列
        self.Xd = torch.roll(self.Xd, 1, dims=1)
        self.Xd[:, 0] = x_vec
        
        # 计算控制信号: y_j = sum_i ( w_{j,i} * x_i )
        # 使用爱因斯坦求和约定: j,i,l 与 i,l 点积 -> j
        y_vec = torch.einsum('jil,il->j', self.Wc, self.Xd)
        
        # 计算各输入通道的信号功率（用于归一化）
        power = torch.sum(self.Xd ** 2, dim=1) + 1e-8  # 避免除零
        
        return y_vec, power
    
    def compute_error(self, y_vec, d_vec):
        """
        计算误差信号（考虑次级路径）
        
        参数:
        - y_vec: 控制信号向量，形状 (J,)
        - d_vec: 当前时刻的干扰信号（在误差麦克风处测量），形状 (K,)
        
        返回:
        - e_vec: 误差信号向量，形状 (K,)
        - anti_noise: 经过次级路径后的反噪声，形状 (K,)
        """
        # 为简化，此处假设次级路径已经通过卷积在线实现。
        # 在实际应用中，次级路径滤波需要维护每个 (k,j) 的延迟线。
        # 这里我们采用向量化卷积的方式：
        # anti_noise_k = sum_j ( s_{k,j} * y_j )   (卷积)
        
        # 为了保持代码简洁，此处使用一个辅助函数 s_conv 来完成卷积。
        # 实际实现时，您可以维护次级路径的延迟线以提升效率。
        anti_noise = self._secondary_path_filtering(y_vec)
        
        e_vec = d_vec - anti_noise
        return e_vec, anti_noise
    
    def _secondary_path_filtering(self, y_vec):
        """
        次级路径滤波（示意性实现）
        
        实际系统中需要维护延迟线以在线计算。此处仅为逻辑示意。
        返回: anti_noise, 形状 (K,)
        """
        # 注意：完整实现需为每个 (k,j) 维护长度为 SecLen 的延迟线，
        # 并逐样本计算卷积。此处仅展示结构，实际使用时请替换为在线卷积代码。
        K, J, SecLen = self.SecPath.shape
        anti_noise = torch.zeros(K)
        # 伪代码，假设已有 y_history 缓冲
        # for k in range(K):
        #     for j in range(J):
        #         anti_noise[k] += torch.dot(self.SecPath[k,j], y_history[j])
        return anti_noise  # 占位符，实际需在线计算
    
    def update_weights(self, e_vec, power, mu):
        """
        手动更新权重（归一化LMS，多通道版本）
        
        参数:
        - e_vec: 误差信号，形状 (K,)
        - power: 各输入通道功率，形状 (I,)
        - mu: 步长参数（可全局或单独指定）
        """
        # 计算滤波参考信号：对于每个误差通道k，输入i经过次级路径s_{k,j}滤波后得到
        # x'_{i,j,k} ，形状 (K, J, I, Len) 但通常我们合并j维度用于更新Wc。
        # 标准MEFxNLMS更新公式：
        # w_{j,i} += mu * sum_k ( e_k * x'_{i,j,k} ) / (||x_i||^2 + eps)
        
        # 此处为简化说明，假设我们已经在线计算了 FiltX (K, I, Len)
        # 实际更新代码需结合次级路径延迟线实时计算 FiltX。
        
        with torch.no_grad():
            for j in range(self.J):
                for i in range(self.I):
                    # 梯度项: sum_{k} e_k * filt_x_{k,i}
                    grad = torch.einsum('k,kl->l', e_vec, self.FiltX[:, i, :])
                    # 归一化更新
                    self.Wc[j, i, :] += mu * grad / power[i]
    
    def LossFunction(self, e_vec, power):
        """
        损失函数（用于基于优化器的训练，可选）
        
        参数:
        - e_vec: 误差信号向量，形状 (K,)
        - power: 各通道功率，形状 (I,)
        
        返回:
        - loss: 标量损失值
        """
        # 使用平均功率进行全局归一化
        avg_power = torch.mean(power)
        loss = torch.sum(e_vec ** 2) / (2 * avg_power)
        return loss
    
    def _get_coeff_(self):
        """获取滤波器系数 (转为numpy)"""
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# 函数: train_mimofxnlms_algorithm()
# 描述: 在线训练多通道FxNLMS算法（手动更新版本，更贴合DSP实现）
#------------------------------------------------------------------------------
def train_mimofxnlms_algorithm(Model, Ref, Disturbance, Stepsize=0.0001):
    """
    训练MIMOFxxNLMS算法
    
    参数:
    - Model: MIMOFxNLMS模型实例
    - Ref: 参考信号，形状 (样本数, I)
    - Disturbance: 干扰信号（误差麦克风处），形状 (样本数, K)
    - Stepsize: 步长参数 mu
    
    返回:
    - Error_signals: 各通道误差信号历史，形状 (样本数, K)
    """
    # 创建进度条
    total_samples = Disturbance.shape[0]
    bar = progressbar.ProgressBar(maxval=total_samples,
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    bar.start()
    Error_signals = []
    
    for n in range(total_samples):
        # 当前时刻输入
        x_vec = Ref[n, :]      # (I,)
        d_vec = Disturbance[n, :]  # (K,)
        
        # 前向传播：计算控制信号和功率
        y_vec, power = Model.feedforward(x_vec)
        
        # 计算误差（需包含次级路径滤波）
        e_vec, anti_noise = Model.compute_error(y_vec, d_vec)
        
        # 手动更新权重（归一化LMS）
        Model.update_weights(e_vec, power, Stepsize)
        
        Error_signals.append(e_vec.detach().numpy())
        
        bar.update(n + 1)
    
    bar.finish()
    return np.array(Error_signals)


# 辅助函数：预计算滤波后的参考信号矩阵（用于离线训练）
def precompute_filtered_reference(ref_signal, secondary_path_matrix):
    """
    预计算所有时刻的滤波-x信号
    
    参数:
    - ref_signal: 原始参考信号，形状 [N]
    - secondary_path_matrix: 次级路径矩阵，形状 [num_error, num_spk, filt_len]
    
    返回:
    - filtered_ref: 形状 [N, num_error, num_spk]
    """
    import scipy.signal
    N = len(ref_signal)
    num_error, num_spk, filt_len = secondary_path_matrix.shape
    filtered_ref = np.zeros((N, num_error, num_spk))
    for m in range(num_error):
        for l in range(num_spk):
            conv_full = scipy.signal.convolve(ref_signal, secondary_path_matrix[m, l], mode='full')
            # 截取有效部分（与实时系统对齐，从 filt_len-1 开始）
            filtered_ref[:, m, l] = conv_full[filt_len-1:filt_len-1+N]
    return torch.from_numpy(filtered_ref).float()


# 改进的离线训练函数（使用预计算滤波-x信号）
def train_mimo_fxnlms_precomputed(model, ref_signal, error_signal, filtered_ref, mu=0.01, delta=0.01):
    """
    使用预计算的滤波-x信号训练 MIMO FxNLMS
    
    参数:
    - model: MIMOFxNLMS 实例
    - ref_signal: 原始参考信号，形状 [N]
    - error_signal: 误差信号，形状 [N, num_error]
    - filtered_ref: 预计算的滤波-x信号，形状 [N, num_error, num_spk]
    - mu: 步长
    - delta: 正则化项
    """
    N = ref_signal.shape[0]
    optimizer = optim.SGD([model.Wc], lr=mu)
    error_history = []
    
    bar = progressbar.ProgressBar(maxval=N, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for t in range(N):
        x_ref = ref_signal[t]
        x_filt_matrix = filtered_ref[t]   # [num_error, num_spk]
        d = error_signal[t]               # [num_error]
        
        y, power = model.feedforward(x_ref, x_filt_matrix)
        power = power + delta
        loss, e = model.LossFunction(y, d, power)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        error_history.append(e.detach().numpy())
        bar.update(t+1)
    
    bar.finish()
    return np.array(error_history)

#------------------------------------------------------------
## 测试用函数
#------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy import signal

def run_mimo_fxnlms_demo():
    # ------------------------------
    # 1. 参数设置
    # ------------------------------
    fs = 16000                 # 采样率 (Hz)
    duration = 5             # 信号时长 (秒)
    N = fs * duration        # 样本点数
    num_spk = 4              # 扬声器数量
    num_error = 4            # 误差麦克风数量
    filter_len = 64          # 控制滤波器长度
    mu = 0.05                # 步长
    delta = 0.01             # 正则化项

    # ------------------------------
    # 2. 加载次级路径（从文件）
    # ------------------------------
    sec_path_file = r"Primary and Secondary Path\secondary_path_5mic_4spk.npy"
    if not os.path.exists(sec_path_file):
        raise FileNotFoundError(f"次级路径文件不存在: {sec_path_file}")
    secondary_path = np.load(sec_path_file)
    num_error, num_spk, sec_path_len = secondary_path.shape
    print(f"加载次级路径: 误差麦克风数量 = {num_error}, 扬声器数量 = {num_spk}, 滤波器长度 = {sec_path_len}")

    # 可选：对次级路径进行归一化或预处理（这里保持原样）
    secondary_path_tensor = torch.from_numpy(secondary_path).float()

    # ------------------------------
    # 3. 生成参考信号（宽带噪声）
    # ------------------------------
    # 使用带限白噪声，频率范围 100~1000 Hz
    lowcut = 100.0
    highcut = 1000.0
    nyquist = 0.5 * fs
    b, a = signal.butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
    white_noise = np.random.randn(N + 100)  # 多出一点避免边界
    ref_signal = signal.lfilter(b, a, white_noise)[100:100+N]
    ref_signal = ref_signal / np.std(ref_signal)  # 归一化
    ref_tensor = torch.from_numpy(ref_signal).float()   # [N]

    # ------------------------------
    # 4. 生成初级噪声和期望的误差信号（模拟）
    # 假设：初级噪声 d(n) 由参考信号经过初级路径产生
    # 初级路径：随机 FIR 滤波器，形状 [num_error]（每个误差麦克风独立）
    # 为简化，这里用相同的初级路径，但实际可以不同
    primary_path_len = 32
    primary_path = np.random.randn(num_error, primary_path_len) * 0.1
    for m in range(num_error):
        decay = np.exp(-np.arange(primary_path_len) / (primary_path_len/3))
        primary_path[m] *= decay

    # 计算初级噪声（干扰信号） d(n)
    d_signal = np.zeros((N, num_error))
    for m in range(num_error):
        # 卷积并截取有效长度
        conv_full = np.convolve(ref_signal, primary_path[m])
        d_signal[:, m] = conv_full[:N]

    # 初始误差信号 e0 = d (此时控制信号为0)
    error_signal = d_signal.copy()   # 之后在训练循环中会被模型输出更新
    error_tensor = torch.from_numpy(error_signal).float()  # [N, num_error]

    # ------------------------------
    # 5. 预计算滤波后的参考信号
    # ------------------------------
    filtered_ref_tensor = precompute_filtered_reference(ref_signal, secondary_path)

    # ------------------------------
    # 6. 创建 MIMO FxNLMS 模型 （使用加载的次级路径维度）
    # ------------------------------
    model = MIMOFxNLMS(filter_len, num_spk, num_error, dtype=torch.float)

    # ------------------------------
    # 7. 离线训练（使用预计算的滤波-x信号）
    # ------------------------------
    # 由于原代码中的 train_mimo_fxnlms_precomputed 已经定义，直接使用
    # 这里为了独立运行，重新实现一个简化版本（但最好复用原函数）
    # 原函数需要传入 model, ref_signal, error_signal, filtered_ref, mu, delta
    # 注意：原函数中的 error_signal 实际是“期望的误差信号”，在我们的模拟中即为 d(n)
    # 因为训练中模型会逐步减小误差，但离线训练时我们直接使用真实的误差信号 e(n)，
    # 然而标准FxNLMS更新需要误差信号 e(n)（即残余噪声）。在模拟中，我们需要在每一步
    # 计算出控制信号并更新误差，而不是直接使用固定的 error_tensor。
    # 因此，上面的 error_tensor 不能作为固定输入，而是需要在线计算。
    # 所以最好重新写一个训练循环，模拟实际物理过程。

    print("开始训练 MIMO FxNLMS ...")

    # 初始化滤波器系数（零）
    # 复制模型内部状态
    Wc = model.Wc.data.clone()
    # 准备延迟线
    Xd = torch.zeros(num_spk, filter_len, dtype=torch.float)
    Xfilt_delay = torch.zeros(num_error, num_spk, filter_len, dtype=torch.float)

    # 记录误差
    error_history = []

    # 使用进度条（可选，需安装 progressbar2）
    try:
        from progressbar import ProgressBar, Bar, Percentage
        bar = ProgressBar(maxval=N, widgets=[Bar('=', '[', ']'), ' ', Percentage()])
        bar.start()
    except ImportError:
        bar = None

    for t in range(N):
        # 当前参考信号
        x_ref = ref_tensor[t]                     # 标量
        # 当前滤波后的参考信号矩阵 (来自预计算)
        x_filt_matrix = filtered_ref_tensor[t]    # [num_error, num_spk]

        # 更新延迟线（模拟 feedforward 内部操作，但手动更新以获取 y）
        Xd = torch.roll(Xd, 1, dims=1)
        Xd[:, 0] = x_ref
        Xfilt_delay = torch.roll(Xfilt_delay, 1, dims=2)
        Xfilt_delay[:, :, 0] = x_filt_matrix

        # 计算控制信号 y (扬声器输出)
        y = torch.einsum('li,li->l', Wc, Xd)      # [num_spk]

        # 计算次级路径作用后的控制信号（到达误差麦克风）
        # 这里需要卷积：每个误差麦克风收到的信号 = sum_l (y[l] 与 secondary_path[m,l] 的卷积)
        # 为简化在线模拟，我们直接使用预计算的滤波-x信号乘以当前控制系数？不对。
        # 正确方法：利用预计算的滤波-x信号和当前滤波器系数 Wc 可以快速得到次级路径输出：
        # 对于每个误差麦克风 m，次级输出 = sum_l (Wc[l] 与 filtered_ref[:,m,l] 的卷积) 在时刻 t 的值。
        # 实际上，利用 Xfilt_delay 可以计算：对于每个 m，次级输出 = sum_l (Wc[l] · Xfilt_delay[m,l,:])
        # 因为 Xfilt_delay[m,l,:] 存储了参考信号与次级路径卷积后的最后 filter_len 个样本，
        # 与 Wc[l] 点乘即得到当前时刻次级输出。
        sec_output = torch.einsum('ml...,l...->m', Xfilt_delay, Wc)   # [num_error]

        # 当前时刻的残余误差 e(n) = d(n) - sec_output
        d_current = torch.tensor(d_signal[t], dtype=torch.float)  # [num_error]
        e_current = d_current - sec_output

        # 计算功率 (每个扬声器的所有误差通道功率和)
        power = torch.einsum('mli->l', Xfilt_delay ** 2) + delta   # [num_spk]

        # 更新滤波器系数（标准 FxNLMS 公式）
        # grad[l] = sum_m e[m] * Xfilt_delay[m,l,:]
        grad = torch.einsum('m,ml...->l...', e_current, Xfilt_delay)   # [num_spk, filter_len]
        Wc = Wc + mu * grad / power.unsqueeze(1)   # 广播除法

        # 记录误差
        error_history.append(e_current.numpy().copy())

        if bar:
            bar.update(t+1)

    if bar:
        bar.finish()

    # 将训练好的系数放回模型
    model.Wc.data = Wc

    # ------------------------------
    # 8. 绘制结果
    # ------------------------------
    error_array = np.array(error_history)  # [N, num_error]
    plt.figure(figsize=(12, 5))
    for m in range(num_error):
        plt.subplot(1, num_error, m+1)
        plt.plot(20*np.log10(np.abs(error_array[:, m]) + 1e-8))
        plt.xlabel('Sample')
        plt.ylabel('Error (dB)')
        plt.title(f'Error microphone {m+1}')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('mimo_fxnlms_error.png')
    plt.show()

    # 绘制滤波器系数
    plt.figure(figsize=(8, 4))
    for l in range(num_spk):
        plt.plot(model.Wc[l].detach().numpy(), label=f'Speaker {l+1}')
    plt.xlabel('Tap index')
    plt.ylabel('Coefficient value')
    plt.title('Trained control filters')
    plt.legend()
    plt.grid(True)
    plt.savefig('mimo_filters.png')
    plt.show()

    print("训练完成。误差曲线和滤波器系数已保存为图片。")

    # 可选：保存模型系数
    np.save('mimo_control_filters.npy', model.Wc.detach().numpy())
    print("滤波器系数已保存至 mimo_control_filters.npy")

if __name__ == "__main__":
    run_mimo_fxnlms_demo()