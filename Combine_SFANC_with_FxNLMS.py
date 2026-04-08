# Combine_SFANC_with_FxNLMS.py
import numpy as np
from scipy.io import loadmat

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
        """
        执行 SFANC-FxNLMS 噪声消除
        
        参数:
            Dis: 干扰信号（期望信号），一维 numpy 数组
            Fx: 滤波参考信号（x_filtered），一维 numpy 数组
            filter_index: 每秒钟对应的固定滤波器索引列表，长度等于信号总秒数（向上取整）
            Stepsize: FxNLMS 算法的步长（学习率）
        
        返回:
            error_signal: 消除后的误差信号，一维 numpy 数组
        """
        N = len(Dis)
        # 确保 Dis 和 Fx 长度一致
        assert len(Fx) == N, "Dis 和 Fx 长度必须相等"
        
        # 初始化输出误差数组
        error_signal = np.zeros(N)
        
        # 滤波器状态：延迟线（存储最近的 filter_len 个 Fx 样本）
        x_buffer = np.zeros(self.filter_len)
        
        # 初始滤波器系数：使用第一秒对应的固定滤波器
        current_index = filter_index[0]
        w = self.control_filters[:, current_index].copy()
        
        # 逐样本处理
        for n in range(N):
            # 更新延迟线
            x_buffer[1:] = x_buffer[:-1]
            x_buffer[0] = Fx[n]
            
            # 计算滤波输出
            y = np.dot(w, x_buffer)
            
            # 计算误差
            e = Dis[n] - y
            error_signal[n] = e
            
            # 计算归一化因子
            norm = np.dot(x_buffer, x_buffer) + self.eps
            
            # 更新滤波器系数
            w += Stepsize * e * x_buffer / norm
            
            # 检查是否需要切换固定滤波器（每秒切换一次）
            # 注意：索引从0开始，所以第 k 秒对应的样本范围为 [k*fs, (k+1)*fs-1]
            # 当处理完第 (n+1) 个样本时，如果 (n+1) 是 fs 的整数倍，则下一秒开始
            if (n + 1) % self.fs == 0:
                next_sec = (n + 1) // self.fs
                if next_sec < len(filter_index):
                    new_index = filter_index[next_sec]
                    if new_index != current_index:
                        # 切换滤波器系数
                        w = self.control_filters[:, new_index].copy()
                        current_index = new_index
        
        return error_signal