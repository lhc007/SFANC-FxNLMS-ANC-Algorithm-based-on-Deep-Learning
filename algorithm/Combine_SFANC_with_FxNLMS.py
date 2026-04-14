# Combine_SFANC_with_FxNLMS.py
import numpy as np
from scipy.io import loadmat
import torch
import torch.optim as optim

class FxNLMS:
    """NumPy 实现的 FxNLMS 自适应滤波器"""
    
    def __init__(self, filter_len, initial_coeffs=None, eps=1e-6):
        self.filter_len = filter_len
        self.eps = eps
        self.w = np.zeros(filter_len) if initial_coeffs is None else initial_coeffs.copy()
        self.x_buffer = np.zeros(filter_len)
    
    def update(self, x_new, desired, step_size):
        """单步更新，返回误差"""
        # 更新延迟线
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_new
        
        # 滤波输出
        y = np.dot(self.w, self.x_buffer)
        
        # 误差
        e = desired - y
        
        # 归一化因子
        norm = np.dot(self.x_buffer, self.x_buffer) + self.eps
        
        # 系数更新
        self.w += step_size * e * self.x_buffer / norm
        
        return e
    
    def reset_buffer(self):
        """重置延迟线（切换滤波器时可能需要）"""
        self.x_buffer.fill(0)
    
    def set_coeffs(self, new_coeffs):
        """直接替换系数（保持延迟线不变）"""
        self.w[:] = new_coeffs

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
        self.filter_len = filter_len
        self.eps = eps
        
        # 加载预训练的固定控制滤波器组
        mat_data = loadmat(MAT_FILE)
        # 尝试找到包含滤波器矩阵的变量名
        # 常见变量名：'Control_filters', 'Filters', 'W', 'control_filters'
        possible_keys = ['Control_filters', 'Filters', 'W', 'control_filters']
        self.control_filters = None
        for key in possible_keys:
            if key in mat_data:
                self.control_filters = mat_data[key]
                break
        if self.control_filters is None:
            # 如果没找到，尝试取第一个非特殊变量
            for key in mat_data:
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    self.control_filters = mat_data[key]
                    break
        if self.control_filters is None:
            raise ValueError(f"无法从 {MAT_FILE} 中找到控制滤波器矩阵，请检查文件内容。")
        
        # 确保滤波器形状正确：假设形状为 (filter_len, num_filters)
        if self.control_filters.shape[0] != filter_len:
            # 可能形状是 (num_filters, filter_len)，则转置
            if self.control_filters.shape[1] == filter_len:
                self.control_filters = self.control_filters.T
            else:
                raise ValueError(f"滤波器长度不匹配：期望 {filter_len}，实际为 {self.control_filters.shape[0]} 或 {self.control_filters.shape[1]}")
        
        self.num_filters = self.control_filters.shape[1]
        print(f"加载了 {self.num_filters} 个固定控制滤波器，每个长度 {self.filter_len}")

    def noise_cancellation(self, Dis, Fx, filter_index, Stepsize):
        N = len(Dis)
        error_signal = np.zeros(N)
        
        # 创建 FxNLMS 实例
        fx_nlms = FxNLMS(self.filter_len, eps=self.eps)
        
        # 初始滤波器系数
        current_index = filter_index[0]
        fx_nlms.set_coeffs(self.control_filters[:, current_index])
        
        for n in range(N):
            # 单步更新，FxNLMS.update 内部自动维护延迟线
            e = fx_nlms.update(Fx[n], Dis[n], Stepsize)
            error_signal[n] = e
            
            # 每秒切换固定滤波器
            if (n + 1) % self.fs == 0:
                next_sec = (n + 1) // self.fs
                if next_sec < len(filter_index):
                    new_index = filter_index[next_sec]
                    if new_index != current_index:
                        # 切换系数，同时重置延迟线（避免历史信号影响）
                        fx_nlms.set_coeffs(self.control_filters[:, new_index])
                        # fx_nlms.reset_buffer()
                        current_index = new_index
        return error_signal


#------------------------------------------------------------------------------
# 类: FxNLMS 算法，初始系数由 GFANC 确定
#------------------------------------------------------------------------------
class FxNLMS2():
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
    
    def _get_coeff_(self):
        return self.Wc.detach().numpy()

class SFANC_FxNLMS2:
    def __init__(self, MAT_FILE, fs=16000, filter_len=1024):
        self.fs = fs
        self.filter_len = filter_len

        # 加载固定滤波器组（形状应为 (filter_len, num_filters) 或 (num_filters, filter_len)）
        mat_data = loadmat(MAT_FILE)
        possible_keys = ['Control_filters', 'Filters', 'W', 'control_filters']
        control_filters = None
        for key in possible_keys:
            if key in mat_data:
                control_filters = mat_data[key]
                break
        if control_filters is None:
            for key in mat_data:
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    control_filters = mat_data[key]
                    break
        if control_filters is None:
            raise ValueError(f"无法从 {MAT_FILE} 中找到控制滤波器矩阵。")

        # 统一形状为 (num_filters, filter_len)
        if control_filters.shape[0] == filter_len:
            control_filters = control_filters.T
        elif control_filters.shape[1] != filter_len:
            raise ValueError(f"滤波器长度不匹配：期望 {filter_len}，实际为 {control_filters.shape[1]}")

        self.control_filters = torch.from_numpy(control_filters).float()  # (num_filters, filter_len)
        self.num_filters = self.control_filters.shape[0]
        print(f"加载了 {self.num_filters} 个固定控制滤波器，每个长度 {self.filter_len}")

    def noise_cancellation(self, Dis, Fx, filter_index, Stepsize):
        """
        参数:
            Dis: 干扰信号，一维数组或张量
            Fx:  滤波参考信号，一维数组或张量
            filter_index: 每秒对应的滤波器索引，长度等于信号秒数（向上取整）
            Stepsize: 学习率
        返回:
            Error: 误差信号列表
        """
        # 统一转换为 numpy 数组以便索引，但保留标量值用于 PyTorch
        if torch.is_tensor(Dis):
            Dis = Dis.numpy().flatten()
        if torch.is_tensor(Fx):
            Fx = Fx.numpy().flatten()
        Dis = np.asarray(Dis, dtype=np.float32).flatten()
        Fx  = np.asarray(Fx,  dtype=np.float32).flatten()

        N = len(Dis)
        assert len(Fx) == N, "Dis 和 Fx 长度必须相等"

        Error = []
        j = 0  # 已过去的秒数计数器

        # 初始滤波器：使用 filter_index[0] 指定的固定滤波器
        init_idx = filter_index[0]
        current_filter = self.control_filters[init_idx]  # shape (filter_len,)
        model = FxNLMS2(Len=self.filter_len, Ws=current_filter)
        optimizer = optim.SGD([model.Wc], lr=Stepsize)

        for ii in range(N):
            # 当前样本
            x_f = torch.tensor(Fx[ii], dtype=torch.float)
            d   = torch.tensor(Dis[ii], dtype=torch.float)

            y, power = model.feedforward(x_f)
            loss, e = model.LossFunction(y, d, power)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Error.append(e.item())

            # 每秒结束时检查是否切换滤波器
            if (ii + 1) % self.fs == 0:
                next_sec = (ii + 1) // self.fs
                if next_sec < len(filter_index):
                    new_idx = filter_index[next_sec]
                    if new_idx != init_idx:
                        print(f'第 {next_sec} 秒：切换控制滤波器，索引 {init_idx} -> {new_idx}')
                        init_idx = new_idx
                        new_filter = self.control_filters[new_idx]
                        model = FxNLMS2(Len=self.filter_len, Ws=new_filter)
                        optimizer = optim.SGD([model.Wc], lr=Stepsize)
                j += 1

        return Error