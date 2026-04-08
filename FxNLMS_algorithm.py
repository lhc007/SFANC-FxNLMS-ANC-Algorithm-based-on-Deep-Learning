import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar

#------------------------------------------------------------------------------
# 类: FxNLMS 算法
# 描述: 实现滤波-x归一化最小均方算法，用于主动噪声控制
#------------------------------------------------------------------------------
class FxNLMS():
    
    def __init__(self, Len):
        """
        初始化FxNLMS算法
        
        参数:
        - Len: 滤波器长度
        """
        # 控制滤波器系数
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)
        # 输入信号延迟线
        self.Xd = torch.zeros(1, Len, dtype= torch.float)
    
    def feedforward(self, Xf):
        """
        前向传播
        
        参数:
        - Xf: 滤波后的参考信号
        
        返回:
        - yt: 控制信号
        - power: 输入信号功率
        """
        # 更新延迟线
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf
        # 计算控制信号
        yt = self.Wc @ self.Xd.t()
        # 计算输入信号功率（FxNLMS与FxLMS的不同之处）
        power = self.Xd @ self.Xd.t()
        return yt, power
    
    def LossFunction(self, y, d, power):
        """
        损失函数
        
        参数:
        - y: 控制信号
        - d: 干扰信号
        - power: 输入信号功率
        
        返回:
        - loss: 损失值
        - e: 误差信号
        """
        # 计算误差信号（干扰信号-控制信号）
        e = d - y
        # 计算归一化损失
        return e**2 / (2 * power), e
    
    def _get_coeff_(self):
        """
        获取滤波器系数
        
        返回:
        - Wc: 滤波器系数
        """
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# 类: MIMOFxNLMS (多通道版本)
# 描述: 实现多输入多输出滤波-x归一化最小均方算法，用于多扬声器、多误差麦克风的ANC系统
#------------------------------------------------------------------------------
class MIMOFxNLMS():
    
    def __init__(self, filter_len, num_spk, num_error, dtype=torch.float):
        """
        初始化MIMO FxNLMS算法
        
        参数:
        - filter_len: 每个控制滤波器的长度
        - num_spk: 扬声器数量（输出通道数）
        - num_error: 误差麦克风数量（输入误差通道数）
        - dtype: 数据类型
        """
        self.filter_len = filter_len
        self.num_spk = num_spk
        self.num_error = num_error
        self.dtype = dtype
        
        # 控制滤波器系数矩阵: [num_spk, filter_len]
        # 每个扬声器对应一个滤波器，需要梯度用于离线训练
        self.Wc = torch.zeros(num_spk, filter_len, requires_grad=True, dtype=dtype)
        
        # 原始参考信号延迟线: [num_spk, filter_len]
        # 每个扬声器独立（虽然输入相同，但为并行计算保留副本）
        self.Xd = torch.zeros(num_spk, filter_len, dtype=dtype)
        
        # 滤波后参考信号延迟线: [num_error, num_spk, filter_len]
        # 每个(误差麦克风, 扬声器)对独立
        self.Xfilt_delay = torch.zeros(num_error, num_spk, filter_len, dtype=dtype)
    
    def feedforward(self, x_ref, x_filt_matrix):
        """
        前向传播：更新延迟线，计算控制信号和功率矩阵
        
        参数:
        - x_ref: 当前时刻的原始参考信号（标量）
        - x_filt_matrix: 当前时刻的滤波后参考信号矩阵，形状 [num_error, num_spk]
        
        返回:
        - y: 控制信号向量，形状 [num_spk]
        - power: 每个扬声器的归一化功率（所有误差通道功率和），形状 [num_spk]
        """
        # 1. 更新原始参考延迟线（每个扬声器独立，内容相同）
        self.Xd = torch.roll(self.Xd, 1, dims=1)
        self.Xd[:, 0] = x_ref
        
        # 2. 更新滤波后参考延迟线（每个误差-扬声器对）
        self.Xfilt_delay = torch.roll(self.Xfilt_delay, 1, dims=2)
        self.Xfilt_delay[:, :, 0] = x_filt_matrix
        
        # 3. 计算控制信号 y = Wc @ Xd^T (逐扬声器)
        # 对于每个扬声器 l: y[l] = sum_{i=0}^{filter_len-1} Wc[l, i] * Xd[l, i]
        y = torch.einsum('li,li->l', self.Wc, self.Xd)   # [num_spk]
        
        # 4. 计算每个扬声器的归一化功率（所有误差通道的滤波-x信号功率之和）
        # power[l] = sum_{m=0}^{num_error-1} sum_{i=0}^{filter_len-1} (Xfilt_delay[m,l,i])^2
        power = torch.einsum('mli->l', self.Xfilt_delay ** 2)   # [num_spk]
        
        return y, power
    
    def LossFunction(self, y, d, power):
        """
        损失函数：计算误差向量和归一化损失（每个扬声器独立）
        
        参数:
        - y: 控制信号向量，形状 [num_spk]
        - d: 误差信号向量，形状 [num_error]
        - power: 每个扬声器的功率，形状 [num_spk]
        
        返回:
        - loss: 总损失（标量，用于自动微分训练）
        - e: 误差向量，形状 [num_error]
        """
        # 注意：在多通道系统中，误差是直接测量得到的，不是 y 和 d 的简单减法。
        # 实际物理过程：d = 期望信号 - 次级路径作用后的 y。
        # 但在算法内部，我们直接用测量到的误差 e（由回调提供）进行更新。
        # 这里的 d 参数实际应该是误差信号 e（为了与原接口一致，保留参数名 d）。
        # 因此直接 e = d（即输入的误差信号）。
        e = d   # 形状 [num_error]
        
        # 损失定义为所有扬声器归一化误差平方和（用于自动微分训练）
        # 注意：功率 power 是每个扬声器的标量，需要合理广播
        # 通常每个扬声器的更新独立，但损失可设为 sum_l (||e||^2 / (2*power[l]))
        loss = torch.sum(e**2) / (2 * torch.sum(power) + 1e-8)
        return loss, e
    
    def _get_coeff_(self):
        """返回控制滤波器系数矩阵，形状 [num_spk, filter_len]"""
        return self.Wc.detach().numpy()
    
    def update_coeff_manual(self, e, x_filt_delay, power, mu):
        """
        手动更新滤波器系数（不使用自动微分），用于实时系统。
        
        参数:
        - e: 误差信号向量，形状 [num_error]
        - x_filt_delay: 滤波后参考延迟线，形状 [num_error, num_spk, filter_len]
        - power: 每个扬声器的功率（已加 delta），形状 [num_spk]
        - mu: 步长
        """
        # 计算每个扬声器的梯度: grad[l] = sum_m e[m] * x_filt_delay[m, l, :]
        grad = torch.einsum('m,ml...->l...', e, x_filt_delay)   # [num_spk, filter_len]
        # 更新系数
        self.Wc.data += mu * grad / power.unsqueeze(1)   # 广播除法

#------------------------------------------------------------------------------
# 函数: train_fxnlms_algorithm()
# 描述: 训练FxNLMS算法
#------------------------------------------------------------------------------
def train_fxnlms_algorithm(Model, Ref, Disturbance, Stepsize=0.0001):
    """
    训练FxNLMS算法
    
    参数:
    - Model: FxNLMS模型实例
    - Ref: 参考信号
    - Disturbance: 干扰信号
    - Stepsize: 学习率
    
    返回:
    - Erro_signal: 误差信号
    """
    # 创建进度条
    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    # 初始化优化器
    optimizer = optim.SGD([Model.Wc], lr=Stepsize)
    
    # 开始训练
    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    
    for itera in range(len_data):
        # 前向传播
        xin = Ref[itera]
        dis = Disturbance[itera]
        y, power = Model.feedforward(xin)
        loss, e = Model.LossFunction(y, dis, power)
        
        # 显示进度
        bar.update(2*itera + 1)
            
        # 反向传播
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())
        
        # 显示进度
        bar.update(2*itera + 2)
    
    bar.finish()
    return Erro_signal

#------------------------------------------------------------------------------
# 函数: train_mimo_fxnlms_algorithm()
# 描述: 离线训练 MIMO FxNLMS 算法（使用自动微分和优化器）
#------------------------------------------------------------------------------
def train_mimo_fxnlms_algorithm(model, ref_signal, error_signal, secondary_path_matrix, 
                                mu=0.01, delta=0.01):
    """
    训练 MIMO FxNLMS 算法（离线仿真）
    
    参数:
    - model: MIMOFxNLMS 实例
    - ref_signal: 原始参考信号，形状 [N] 或 [N, 1]
    - error_signal: 误差信号（期望的残余噪声），形状 [N, num_error]
    - secondary_path_matrix: 次级路径脉冲响应矩阵，形状 [num_error, num_spk, filter_len]
    - mu: 步长
    - delta: 正则化项（加到功率上）
    
    返回:
    - error_history: 每个样本的误差向量历史（可选）
    """
    # 确保输入为 torch 张量
    if not isinstance(ref_signal, torch.Tensor):
        ref_signal = torch.from_numpy(ref_signal).float()
    if not isinstance(error_signal, torch.Tensor):
        error_signal = torch.from_numpy(error_signal).float()
    if not isinstance(secondary_path_matrix, torch.Tensor):
        secondary_path_matrix = torch.from_numpy(secondary_path_matrix).float()
    
    N = ref_signal.shape[0]
    num_error = model.num_error
    num_spk = model.num_spk
    filt_len = model.filter_len
    
    # 准备滤波后的参考信号（离线卷积）
    # 为简化，使用线性卷积逐样本生成滤波-x信号
    # 实际可预先计算全部，但为演示动态更新，采用滑动卷积
    
    # 优化器
    optimizer = optim.SGD([model.Wc], lr=mu)
    
    # 延迟线状态
    conv_buffers = [np.zeros(filt_len - 1) for _ in range(num_error * num_spk)]
    
    error_history = []
    
    bar = progressbar.ProgressBar(maxval=N, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for t in range(N):
        x_ref = ref_signal[t]
        # 计算当前时刻的滤波-x矩阵
        x_filt_matrix = torch.zeros(num_error, num_spk)
        for m in range(num_error):
            for l in range(num_spk):
                # 用次级路径卷积当前参考信号（简化：每次重新卷积，效率低，仅用于演示）
                # 实际训练应预计算
                idx = m * num_spk + l
                # 更新缓冲区（这里演示使用 numpy 卷积，但为了梯度，需保持 torch）
                # 为保持自动微分，建议预先计算滤波-x信号并存储
                pass
        # 由于离线训练时通常预先计算所有滤波-x信号，这里简化：假设已经作为输入提供
        # 实际实现中，可预先计算 filtered_ref 矩阵 [N, num_error, num_spk]
        raise NotImplementedError("离线训练需要预先计算滤波后的参考信号矩阵")
    
    bar.finish()
    return error_history

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
# 函数: Generating_boardband_noise_wavefrom_tensor()
# 描述: 生成测试用的宽带噪声
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    """
    生成测试用的宽带噪声
    
    参数:
    - Wc_F: 截止频率
    - Seconds: 噪声长度（秒）
    - fs: 采样率
    
    返回:
    - yout: 生成的噪声张量
    """
    filter_len = 1024 
    # 设计带通滤波器
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window='hamming', fs=fs) 
    # 计算信号长度
    N = filter_len + Seconds * fs
    # 生成随机信号
    xin = np.random.randn(N)
    # 应用滤波器
    y = signal.lfilter(bandpass_filter, 1, xin)
    # 去除滤波器的 transient 响应
    yout = y[filter_len:]
    # 标准化
    yout = yout / np.sqrt(np.var(yout))
    # 返回形状为 [1 x 采样率] 的张量
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)


#------------------------------------------------------------
## 测试用函数
#------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy import signal

# 假设用户已有上述 MIMOFxNLMS 及相关函数（从原代码中复制）
# 为完整，这里重新导入或定义（实际使用时直接 import 原模块即可）
# 此处省略原代码中的类定义，假设它们已在当前环境中

def run_mimo_fxnlms_demo():
    # ------------------------------
    # 1. 参数设置
    # ------------------------------
    fs = 16000                 # 采样率 (Hz)
    duration = 5             # 信号时长 (秒)
    N = fs * duration        # 样本点数
    num_spk = 2              # 扬声器数量
    num_error = 2            # 误差麦克风数量
    filter_len = 64          # 控制滤波器长度
    mu = 0.05                # 步长
    delta = 0.01             # 正则化项

    # 次级路径长度（模拟）
    sec_path_len = 32

    # ------------------------------
    # 2. 生成参考信号（宽带噪声）
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
    # 3. 生成次级路径脉冲响应（随机）
    # ------------------------------
    # 次级路径矩阵形状 [num_error, num_spk, sec_path_len]
    secondary_path = np.random.randn(num_error, num_spk, sec_path_len) * 0.1
    # 添加指数衰减使其更像物理系统
    for m in range(num_error):
        for l in range(num_spk):
            decay = np.exp(-np.arange(sec_path_len) / (sec_path_len/3))
            secondary_path[m, l] *= decay
    secondary_path_tensor = torch.from_numpy(secondary_path).float()

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
    # 6. 创建 MIMO FxNLMS 模型
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