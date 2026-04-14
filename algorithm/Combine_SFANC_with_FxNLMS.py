# Combine_SFANC_with_FxNLMS.py
import numpy as np
from scipy.io import loadmat
import torch
import torch.optim as optim

class FxNLMS():
    """
    PyTorch 实现的 FxNLMS
    """
    def __init__(self, Len, Ws):
        self.Wc = torch.tensor(Ws, requires_grad=True) # Ws: 初始系数
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
    
    def feedforward(self,Xf):
        self.Xd = torch.roll(self.Xd,1,1)
        self.Xd[0,0] = Xf 
        yt = self.Wc @ self.Xd.t()
        power = self.Xd @ self.Xd.t() # 与 FxLMS 不同
        return yt, power
    
    def LossFunction(self, y, d, power):
        e = d-y
        return e**2 / (2*power + 1e-12), e
    
    def set_coeffs(self, new_Ws):
        """
        替换滤波器系数，但保留延迟线（不重置 Xd）
        new_Ws: 新系数（数组或张量）
        """
        self.Wc.data = torch.tensor(new_Ws, dtype=torch.float32)
        
    def _get_coeff_(self):
        return self.Wc.detach().numpy()

# class FxNLMS:
#   """
#       NumPy 方式实现的 FANC-FxNLMS 混合主动噪声控制算法
#   """
    
#     def __init__(self, filter_len, initial_coeffs=None, eps=1e-6):
#         self.filter_len = filter_len
#         self.eps = eps
#         self.w = np.zeros(filter_len) if initial_coeffs is None else initial_coeffs.copy()
#         self.x_buffer = np.zeros(filter_len)
    
#     def update(self, x_new, desired, step_size):
#         """单步更新，返回误差"""
#         # 更新延迟线
#         self.x_buffer[1:] = self.x_buffer[:-1]
#         self.x_buffer[0] = x_new
        
#         # 滤波输出
#         y = np.dot(self.w, self.x_buffer)
        
#         # 误差
#         e = desired - y
        
#         # 归一化因子
#         norm = np.dot(self.x_buffer, self.x_buffer) + self.eps
        
#         # 系数更新
#         self.w += step_size * e * self.x_buffer / norm
        
#         return e
    
#     def reset_buffer(self):
#         """重置延迟线（切换滤波器时可能需要）"""
#         self.x_buffer.fill(0)
    
#     def set_coeffs(self, new_coeffs):
#         """直接替换系数（保持延迟线不变）"""
#         self.w[:] = new_coeffs

class SFANC_FxNLMS:
    """
    SFANC-FxNLMS 混合主动噪声控制算法类
    
    结合预训练的固定控制滤波器（选择性固定滤波器，SFANC）与 FxNLMS 自适应算法。
    根据每个时间段的噪声类型选择对应的固定滤波器作为初始控制滤波器，
    然后利用 FxNLMS 算法对残余噪声进行自适应消除。
    """
    
    def __init__(self, MAT_FILE, fs=16000, filter_len=1024, eps=1e-6):
        """
        初始化 SFANC-FxNLMS 控制器
        
        参数:
            MAT_FILE: 预训练固定控制滤波器的 .mat 文件路径
            fs: 采样率，默认为 16000 Hz
            filter_len: 滤波器长度，默认为 1024（应与预训练滤波器长度一致）
            eps: 归一化步长中的正则化常数，防止除零
        """
        self.fs = fs
        # self.filter_len = filter_len
        self.eps = eps
        self.control_filters = self.Load_Pretrained_filters_to_tensor(MAT_FILE) # 子控制滤波器 torch.Size([15, 1024])
        self.num_filters = self.control_filters.shape[0] #滤波器数量
        self.filter_len = self.control_filters.shape[1] #滤波器长度
        print(f"加载了 {self.num_filters} 个固定控制滤波器，每个长度 {self.filter_len}")

    # def noise_cancellation(self, Dis, Fx, filter_index, Stepsize):
    #   """
    #       NumPy 方式实现的 FANC-FxNLMS 混合主动噪声控制算法
    #   """

    #     N = len(Dis)
    #     error_signal = np.zeros(N)
        
    #     # 创建 FxNLMS 实例
    #     fx_nlms = FxNLMS(self.filter_len, eps=self.eps)
        
    #     # 初始滤波器系数
    #     current_index = filter_index[0]
    #     fx_nlms.set_coeffs(self.control_filters[current_index, :])
        
    #     for n in range(N):
    #         # 单步更新，FxNLMS.update 内部自动维护延迟线
    #         e = fx_nlms.update(Fx[n], Dis[n], Stepsize)
    #         error_signal[n] = e
            
    #         # 每秒切换固定滤波器
    #         if (n + 1) % self.fs == 0:
    #             next_sec = (n + 1) // self.fs
    #             if next_sec < len(filter_index):
    #                 new_index = filter_index[next_sec]
    #                 if new_index != current_index:
    #                     # 切换系数，同时重置延迟线（避免历史信号影响）
    #                     fx_nlms.set_coeffs(self.control_filters[new_index, :])
    #                     # fx_nlms.reset_buffer()
    #                     current_index = new_index
    #     return error_signal

    def noise_cancellation(self, Dis, Fx, filter_index, Stepsize):
        """
        PyTorch 实现的 FxNLMS 自适应滤波器
        参数:
            Dis: 干扰信号，一维 PyTorch 张量 (float32)
            Fx:  滤波参考信号，一维 PyTorch 张量 (float32)
            filter_index: 每秒对应的固定滤波器索引列表
            Stepsize: 学习率（用于 SGD 优化器）
        返回:
            error_signal: 误差信号列表（Python 列表，包含每个样本的误差）
        """

        Dis = Dis.flatten()
        Fx = Fx.flatten()
        N = len(Dis)

        error_signal = []
        
        # 初始滤波器系数（注意 control_filters 是 NumPy 数组，形状 (num_filters, filter_len)）
        current_index = filter_index[0]
        initial_coeffs = self.control_filters[current_index, :]   # NumPy 数组
        
        # 创建 FxNLMS 实例
        model = FxNLMS(self.filter_len, initial_coeffs)
        optimizer = optim.SGD([model.Wc], lr=Stepsize)
        
        for ii in range(N):
            x_f = Fx[ii]      # 标量张量
            d   = Dis[ii]     # 标量张量
            
            y, power = model.feedforward(x_f)
            loss, e = model.LossFunction(y, d, power)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            error_signal.append(e.item())   # 存储 Python 数值
            
            # 每秒结束时检查是否切换滤波器
            if (ii + 1) % self.fs == 0:
                next_sec = (ii + 1) // self.fs
                if next_sec < len(filter_index):
                    new_index = filter_index[next_sec]
                    if new_index != current_index:
                        # 仅替换系数，保留延迟线 Xd
                        new_coeffs = self.control_filters[new_index, :]
                        model.set_coeffs(new_coeffs)
                        # 由于系数改变，优化器仍指向同一个 Parameter，无需重建
                        # 如果步长需要变化，可以重建 optimizer（此处保持不变）
                        current_index = new_index
        
        return error_signal   # 返回误差列表，也可转为 np.array(error_signal)

    def Load_Pretrained_filters_to_tensor(self, MAT_FILE): # 从 mat 文件加载预训练控制滤波器
        mat_contents = loadmat(MAT_FILE)
        print(f"文件中包含的所有键 (Keys): {list(mat_contents.keys())}")        
        Wc_vectors = mat_contents['Wc_v'][:15, :] # !!! 15 个子控制滤波器
        print(f"滤波器原始形状: {Wc_vectors.shape}")   # 输出 (num_filters, filter_len)
        return torch.from_numpy(Wc_vectors).type(torch.float)