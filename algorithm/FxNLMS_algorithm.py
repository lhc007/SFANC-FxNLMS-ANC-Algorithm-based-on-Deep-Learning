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
# 函数: train_fxnlms_algorithm()
# 描述: 在线训练FxNLMS算法
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

#------------------------------------------------------------
# 函数: Generating_broadband_noise_wavefrom_tensor()
# 描述: 生成测试用的宽带噪声
#------------------------------------------------------------
def Generating_broadband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
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


