# Combine_SFANC_with_FxNLMS.py
import numpy as np
from scipy.io import loadmat
import torch
import torch.optim as optim

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class MIMO_FxNLMS:
    """
    多通道 FxNLMS (多输入多输出)
    支持: I 个参考输入通道, J 个次级源输出通道, K 个误差传感器通道
    手动更新滤波器系数，不依赖自动求导。
    在单通道 (I=J=K=1, 次级路径单位增益) 时与原始 FxNLMS 完全一致。
    """
    def __init__(self, filter_len, num_inputs, num_outputs, num_errors,
                 secondary_path=None, mu=0.1, eps=1e-6, initial_coeffs=None):
        """
        参数:
            filter_len: FIR 滤波器长度 L
            num_inputs : 参考信号通道数 I
            num_outputs: 次级源通道数 J
            num_errors : 误差传感器通道数 K
            secondary_path: 次级路径脉冲响应，形状 (J, K, sec_len) 或 None
                            若为 None，则假设次级路径为单位脉冲 (直接路径)
            mu: 步长 (固定学习率)
            eps: 归一化正则化常数
            initial_coeffs: 初始滤波器系数，形状 (J, I, L) 或 None（默认零）
        """
        self.L = filter_len
        self.I = num_inputs
        self.J = num_outputs
        self.K = num_errors
        self.mu = mu
        self.eps = eps

        # 初始化滤波器系数 (J, I, L)
        if initial_coeffs is None:
            self.Wc = torch.zeros((self.J, self.I, self.L), dtype=torch.float32)
        else:
            self.Wc = torch.tensor(initial_coeffs, dtype=torch.float32)

        # 延迟线: 每个输入通道一条 (I, L) 存储参考信号历史
        self.Xd = torch.zeros((self.I, self.L), dtype=torch.float32)

        # 延迟线: 每个输出通道一条 (J, max_sec_len) 存储控制输出历史，用于次级路径滤波
        self.Yd = None  # 延迟初始化

        # 次级路径 (用于生成滤波参考信号 和 误差信号中的次级路径滤波)
        if secondary_path is None:
            # 默认单位脉冲: 每个输出只影响对应误差通道 (需 J == K)
            assert self.J == self.K, "默认次级路径要求 J == K"
            self.S = torch.eye(self.J, self.K).unsqueeze(-1)  # (J, K, 1)
        else:
            # self.S = torch.tensor(secondary_path, dtype=torch.float32)
            S_raw = torch.as_tensor(secondary_path, dtype=torch.float32).clone()
            # 直接截断至滤波器长度（如果更长）
            if S_raw.shape[2] > self.L:
                self.S = S_raw[:, :, :self.L]
            else:
                self.S = S_raw
        self.sec_len = self.S.shape[2]

        # 滤波参考信号延迟线 (J, I, K, L)
        self.Rd = torch.zeros((self.J, self.I, self.K, self.L), dtype=torch.float32)

        # 为次级路径滤波初始化 Yd 延迟线
        if self.sec_len > 1:
            self.Yd = torch.zeros((self.J, self.sec_len), dtype=torch.float32)

    def _apply_secondary_path(self, y):
        """
        将控制输出 y (J,) 通过次级路径滤波，得到误差传感器处的信号 (K,)
        使用当前 Yd 延迟线（自动维护）
        """
        if self.sec_len == 1:
            # 标量增益情况
            return torch.einsum('jk,j->k', self.S[:, :, 0], y)
        else:
            # FIR 滤波: 需要 y 的历史值，self.Yd 已存储
            # 更新 Yd 延迟线
            self.Yd = torch.roll(self.Yd, shifts=1, dims=1)
            self.Yd[:, 0] = y
            # 对每个误差通道 k，计算 sum_j (s_jk * y_j)
            y_filtered = torch.zeros(self.K, dtype=torch.float32)
            for k in range(self.K):
                for j in range(self.J):
                    # 卷积: sum_{m=0}^{sec_len-1} S[j,k,m] * Yd[j, m]
                    y_filtered[k] += torch.dot(self.S[j, k], self.Yd[j])
            return y_filtered

    def _update_filtered_ref(self, x):
        """
        更新滤波参考信号延迟线 Rd
        输入: x (I,) 当前时刻参考信号
        """
        # 先滚动 Rd 所有延迟线
        self.Rd = torch.roll(self.Rd, shifts=1, dims=3)  # 在最后一维（L）上滚动
        # 计算当前时刻的滤波参考信号值 r_{ijk}[0] = sum_{m=0}^{sec_len-1} S[j,k,m] * Xd[i, m]
        # 注意: Xd[i, m] 中 m=0 是最新样本，m=1 是前一时刻，与 S 的索引一致
        for j in range(self.J):
            for i in range(self.I):
                for k in range(self.K):
                    # 点积: S[j,k] 与 Xd[i, 0:sec_len]
                    r_val = torch.dot(self.S[j, k], self.Xd[i, :self.sec_len])
                    self.Rd[j, i, k, 0] = r_val

    def feedforward(self, x):
        """
        前向计算控制输出 y
        参数:
            x: 当前时刻所有参考信号，形状 (I,)
        返回:
            y: 控制输出，形状 (J,)
            power: 每个滤波器使用的归一化功率（基于滤波参考信号），形状 (J, I)
        """
        # 1. 更新参考信号延迟线 Xd
        self.Xd = torch.roll(self.Xd, shifts=1, dims=1)
        self.Xd[:, 0] = x

        # 2. 更新滤波参考信号延迟线 Rd
        self._update_filtered_ref(x)

        # 3. 计算控制输出 y_j = sum_i dot(Wc[j,i], Xd[i])
        y = torch.zeros(self.J, dtype=torch.float32)
        for j in range(self.J):
            for i in range(self.I):
                y[j] += torch.dot(self.Wc[j, i], self.Xd[i])

        # 4. 计算每个滤波器 (j,i) 的归一化功率 (基于 Rd 向量)
        power = torch.zeros((self.J, self.I), dtype=torch.float32)
        for j in range(self.J):
            for i in range(self.I):
                # 对该滤波器所有误差通道的 Rd 向量求平方和
                sum_sq = 0.0
                for k in range(self.K):
                    sum_sq += torch.dot(self.Rd[j, i, k], self.Rd[j, i, k])
                power[j, i] = sum_sq + self.eps

        return y, power

    def LossFunction(self, y, d, filtered_ref=None):
        """
        计算瞬时误差 e 和损失（仅用于监控，不参与梯度）
        参数:
            y: 控制输出，形状 (J,)
            d: 期望信号（误差传感器处信号），形状 (K,)
            filtered_ref: 未使用，保留接口
        返回:
            loss: 标量损失 (0.5 * e^T e)
            e: 误差向量 (K,)
        """
        # 计算误差 e = d - S * y
        y_filtered = self._apply_secondary_path(y)
        e = d - y_filtered
        loss = 0.5 * torch.sum(e ** 2)
        return loss, e

    def update(self, x, d, y=None, filtered_ref=None):
        """
        执行 MIMO FxNLMS 系数更新（手动更新）
        参数:
            x: 当前时刻参考信号 (I,)
            d: 期望信号 (K,)
            y: 控制输出 (J,)（如果为 None，则自动调用 feedforward 计算）
            filtered_ref: 未使用，保留接口
        返回:
            e: 误差信号 (K,)
        """
        # 如果没有提供 y，则前向计算
        if y is None:
            y, _ = self.feedforward(x)
        else:
            # 仍然需要更新内部状态（延迟线等），但用户已提供 y
            # 为了保持内部状态一致，仍需调用 feedforward（它会更新 Xd, Rd）
            # 注意：feedforward 会重新计算 y，可能与传入的 y 不一致
            # 因此这里要求调用前已经执行过 feedforward，或者直接使用内部计算
            # 为安全起见，我们强制调用 feedforward 并使用其 y
            y, _ = self.feedforward(x)

        # 1. 计算误差 e = d - S * y
        y_filtered = self._apply_secondary_path(y)
        e = d - y_filtered

        # 2. 更新每个滤波器 W[j,i]
        for j in range(self.J):
            for i in range(self.I):
                # 分子: sum_k e_k * Rd[j,i,k]
                grad_vec = torch.zeros(self.L, dtype=torch.float32)
                power = 0.0
                for k in range(self.K):
                    grad_vec += e[k] * self.Rd[j, i, k]
                    power += torch.dot(self.Rd[j, i, k], self.Rd[j, i, k])
                power += self.eps
                # 更新系数
                self.Wc[j, i] += self.mu * grad_vec / power

        return e

    def set_coeffs(self, new_Ws):
        """
        替换滤波器系数，保留延迟线
        new_Ws: 形状 (J, I, L) 的张量或数组
        """
        self.Wc.data = torch.tensor(new_Ws, dtype=torch.float32)

    def _get_coeff_(self):
        return self.Wc.detach().numpy()

    def reset(self):
        """重置所有延迟线（不重置滤波器系数）"""
        self.Xd.fill_(0)
        self.Rd.fill_(0)
        if self.Yd is not None:
            self.Yd.fill_(0)
            
class MIMO_SFANC_FxNLMS:
    """
    多通道 SFANC-FxNLMS 混合主动噪声控制算法类
    
    结合预训练的固定控制滤波器（选择性固定滤波器，SFANC）与 MIMO FxNLMS 自适应算法。
    根据每个时间段的噪声类型选择对应的固定滤波器作为初始控制滤波器，
    然后利用 MIMO FxNLMS 算法对残余噪声进行自适应消除。
    """
    
    def __init__(self, MAT_FILE, num_inputs, num_outputs, num_errors,
                 secondary_path, fs=16000, eps=1e-6):
        """
        初始化多通道 SFANC-FxNLMS 控制器
        
        参数:
            MAT_FILE: 预训练固定控制滤波器的 .mat 文件路径
                      文件中应包含变量 'Wc_v'，形状为 (num_filters, J, I, L)
            num_inputs : 参考信号通道数 I
            num_outputs: 次级源通道数 J
            num_errors : 误差传感器通道数 K
            secondary_path: 次级路径脉冲响应，形状 (J, K, sec_len)
            fs: 采样率，默认为 16000 Hz
            eps: 归一化步长中的正则化常数，防止除零
        """
        self.fs = fs
        self.eps = eps
        self.I = num_inputs
        self.J = num_outputs
        self.K = num_errors
        self.secondary_path = secondary_path  # 形状 (J, K, sec_len)
        
        if secondary_path is None:
            # 如果没有提供次级路径，默认使用单位矩阵
            self.S = torch.eye(self.J, self.K).unsqueeze(-1)  # (J, K, 1)
        else:
            # 如果提供了次级路径，确保其形状为 (J, K, sec_len)
            # 并将其转换为 torch 张量 (float32)
            self.S = torch.tensor(secondary_path, dtype=torch.float32)
            # 确保至少 3 维
            if self.S.dim() == 1:
                # 假设 (L_sec,) -> (1, 1, L_sec)
                self.S = self.S.reshape(1, 1, -1)
            elif self.S.dim() == 2:
                # 假设 (J, K) 或 (K, L_sec) 等，这里按 (J, K) 处理并增加长度维度
                # 更安全的方式：如果形状为 (J, K)，则扩展为 (J, K, 1)
                if self.S.shape[0] == self.J and self.S.shape[1] == self.K:
                    self.S = self.S.unsqueeze(-1)  # (J, K, 1)
                else:
                    # 否则尝试 reshape 为 (1, K, L_sec) 或类似，但简单起见报错提示
                    raise ValueError(f"无法自动推断次级路径形状 {self.S.shape}，期望 (J, K) 或 (J, K, L_sec)")
            # 如果已经是三维，保持不变
        self.secondary_path = self.S   # 确保传入 MIMO_FxNLMS 的是三维张量

        # 加载预训练的固定控制滤波器，形状应为 (num_filters, J, I, L)
        self.control_filters = self.Load_Pretrained_filters_to_tensor(MAT_FILE)
        # 期望形状: (num_filters, J, I, L)
        assert self.control_filters.dim() == 4, "预训练滤波器必须是4维张量 (num_filters, J, I, L)"
        self.num_filters = self.control_filters.shape[0]
        self.filter_len = self.control_filters.shape[3]
        print(f"加载了 {self.num_filters} 个固定控制滤波器，每个形状 ({self.J},{self.I},{self.filter_len})")

    def noise_cancellation(self, Dis, Re, filter_index, Stepsize):
        """
        多通道噪声消除主函数
        
        参数:
            Re: Reference signal 参考信号，形状 (I, N) 或 (N, I) — 每个时刻 I 个通道
            Dis: Desired signal 期望信号 （误差传感器处期望信号），形状 (K, N) 或 (N, K)
            filter_index: 每秒对应的固定滤波器索引列表，长度等于总秒数
            Stepsize: 步长 (μ)，标量
        返回:
            error_signal: 误差信号列表，形状 (K, N) 的列表形式（每个样本为 K 维向量）
        """
        # 统一转换为 (I, N) 和 (K, N) 并确保为 torch.float32
        Ref = torch.as_tensor(Re, dtype=torch.float32)
        Des = torch.as_tensor(Dis, dtype=torch.float32)
        if Ref.dim() == 1:
            Ref = Ref.unsqueeze(0)  # (1, N)
        if Ref.shape[0] != self.I:
            Ref = Ref.T  # 转置为 (I, N)
        if Des.dim() == 1:
            Des = Des.unsqueeze(0)  # (1, N)
        if Des.shape[0] != self.K:
            Des = Des.T  # 转置为 (K, N)
        
        N = Ref.shape[1]  # 总样本数
        assert Des.shape[1] == N, "参考信号和期望信号长度不匹配"
        
        # 初始固定滤波器系数
        current_index = filter_index[0]
        initial_coeffs = self.control_filters[current_index]  # (J, I, L)
        
        # 创建 MIMO_FxNLMS 实例
        model = MIMO_FxNLMS(
            filter_len=self.filter_len,
            num_inputs=self.I,
            num_outputs=self.J,
            num_errors=self.K,
            secondary_path=self.secondary_path,
            mu=Stepsize,
            eps=self.eps,
            initial_coeffs=initial_coeffs
        )
        
        error_history = []  # 存储每个样本的误差向量 (K,)
        
        for n in range(N):
            # 获取当前时刻的参考信号 (I,) 和期望信号 (K,)
            x_n = Ref[:, n]   # (I,)
            d_n = Des[:, n]   # (K,)
            
            # 执行一步更新，返回误差向量 (K,)
            e_n = model.update(x_n, d_n)
            error_history.append(e_n.clone().detach().numpy())
            
            # 每秒结束时检查是否切换固定滤波器
            if (n + 1) % self.fs == 0:
                sec_idx = (n + 1) // self.fs
                if sec_idx < len(filter_index):
                    new_index = filter_index[sec_idx]
                    if new_index != current_index:
                        new_coeffs = self.control_filters[new_index]  # (J, I, L)
                        model.set_coeffs(new_coeffs)
                        current_index = new_index
                        # 注意：此处保留了所有延迟线（Xd, Rd, Yd），不重置
                        # 若需要重置延迟线，可调用 model.reset()
        
        # 将误差历史转换为 numpy 数组，形状 (N, K) 或 (K, N)，根据需求返回
        error_signal = np.array(error_history)  # (N, K)
        return error_signal  # 每行对应一个时刻，每列对应一个误差通道

    def Load_Pretrained_filters_to_tensor(self, MAT_FILE):
            """
            从 .mat 文件加载预训练的多通道固定控制滤波器
            期望变量名: 'Wc_v'，形状 (num_filters, J, I, L)
            若文件中为旧格式 (num_filters, L)，则自动扩展为 (num_filters, 1, 1, L)
            """
            mat_contents = loadmat(MAT_FILE)
            print(f"文件中包含的所有键 (Keys): {list(mat_contents.keys())}")
            Wc_vectors = mat_contents['Wc_v']
            print(f"原始滤波器形状: {Wc_vectors.shape}")
            
            # 如果导入的是2维 (num_filters, L)，则扩展为 (num_filters, J, I, L)
            if Wc_vectors.ndim == 2:
                print("检测到2维滤波器，自动扩展为4维 (num_filters, J, I, L)，其中 J=I=1")
                Wc_vectors = Wc_vectors[:, np.newaxis, np.newaxis, :]  # (num_filters, 1, 1, L)
                # 检查 J 和 I 是否均为1，否则报错
                if self.J != 1 or self.I != 1:
                    raise ValueError(f"预训练滤波器为单通道，但 MIMO 配置要求 J={self.J}, I={self.I}")
            elif Wc_vectors.ndim == 4:
                # 期望形状 (num_filters, J, I, L)
                if Wc_vectors.shape[1] != self.J or Wc_vectors.shape[2] != self.I:
                    raise ValueError(f"滤波器形状不匹配: 期望 (?, {self.J}, {self.I}, ?)，实际 {Wc_vectors.shape}")
            else:
                raise ValueError(f"不支持的滤波器维度: {Wc_vectors.ndim}，需要2或4维")
            
            return torch.from_numpy(Wc_vectors).float()


##================================================================================
# class MIMO_FxNLMS:
#     """
#     多通道 FxNLMS (滤波-x 归一化最小均方) 算法，支持任意参考通道数(R)、输出通道数(O)、误差通道数(E)。
#     假设 E == O，每个输出通道对应一个误差通道。
#     """
#     def __init__(self, Len, R, O, E, Ws):
#         """
#         参数:
#             Len: 滤波器长度
#             R:   参考通道数
#             O:   输出通道数
#             E:   误差通道数 (应与 O 相等)
#             Ws:  初始滤波器系数，形状 (O, R, Len) 的张量或可转换为张量的数组
#         """
#         assert O == E, "多通道 FxNLMS 要求输出通道数等于误差通道数"
#         self.Wc = torch.tensor(Ws, dtype=torch.float32, requires_grad=True)  # (O, R, Len)
#         self.R = R
#         self.O = O
#         self.E = E
#         self.Len = Len
#         # 延迟线：每个参考通道一个，存储滤波参考信号的历史样本
#         self.Xd = torch.zeros(R, Len, dtype=torch.float32)

#     def feedforward(self, Xf):
#         """
#         前向计算：更新延迟线，计算控制输出 y 和归一化功率 power。
#         参数:
#             Xf: 当前时刻的滤波参考信号，形状 (R,)
#         返回:
#             y:     控制输出信号，形状 (O,)
#             power: 归一化功率（标量），等于所有参考通道延迟线的能量之和
#         """
#         # 更新延迟线：每个参考通道滚动并放入新样本
#         self.Xd = torch.roll(self.Xd, 1, dims=1)
#         self.Xd[:, 0] = Xf  # Xf 形状 (R,)
#         # 计算输出：每个输出通道 o = sum_r (Wc[o,r] · Xd[r])
#         y = torch.einsum('orl,rl->o', self.Wc, self.Xd)  # (O,)
#         # 计算功率：所有参考通道延迟线样本的能量之和（标量）
#         power = torch.sum(self.Xd ** 2)
#         return y, power

#     def LossFunction(self, y, d, power):
#         """
#         计算损失和误差信号。
#         参数:
#             y:     控制输出，形状 (O,)
#             d:     期望信号（误差麦克风信号），形状 (E,)  (E == O)
#             power: 归一化功率，标量
#         返回:
#             loss: 损失值 (标量)
#             e:    误差信号，形状 (O,)
#         """
#         e = d - y                     # (O,)
#         loss = torch.sum(e ** 2) / (2 * power + 1e-12)
#         return loss, e

#     def set_coeffs(self, new_Ws):
#         """替换滤波器系数，保留延迟线"""
#         self.Wc.data = torch.tensor(new_Ws, dtype=torch.float32)

#     def _get_coeff_(self):
#         """返回当前系数 (numpy 数组)"""
#         return self.Wc.detach().numpy()


# class MIMO_SFANC_FxNLMS:
#     """
#     SFANC-FxNLMS 混合主动噪声控制算法（多通道版本）
    
#     根据每秒钟的噪声类型选择预训练的固定控制滤波器作为初始值，
#     然后使用多通道 FxNLMS 算法进行自适应消除。
#     """
#     def __init__(self, MAT_FILE, fs=16000, filter_len=1024, eps=1e-6):
#         """
#         参数:
#             MAT_FILE:   预训练固定控制滤波器的 .mat 文件路径
#             fs:         采样率 (Hz)
#             filter_len: 滤波器长度（应与预训练滤波器长度一致）
#             eps:        正则化常数
#         """
#         self.fs = fs
#         self.eps = eps
#         # 加载预训练滤波器，形状 (num_filters, O, R, Len)
#         self.control_filters = self._load_pretrained_filters(MAT_FILE)
#         self.num_filters = self.control_filters.shape[0]
#         self.filter_len = self.control_filters.shape[3]
#         self.R = self.control_filters.shape[2]   # 参考通道数
#         self.O = self.control_filters.shape[1]   # 输出通道数
#         self.E = self.O                          # 假设误差通道数等于输出通道数
#         print(f"加载了 {self.num_filters} 个固定控制滤波器，每个形状 (O={self.O}, R={self.R}, Len={self.filter_len})")

#     def _load_pretrained_filters(self, MAT_FILE):
#         """从 .mat 文件加载预训练滤波器，转换为 torch.Tensor"""
#         mat_contents = loadmat(MAT_FILE)
#         print(f"文件中包含的所有键: {list(mat_contents.keys())}")
#         # 假设 mat 文件中变量名为 'Wc_v'，形状 (num_filters, O*R*Len) 或 (num_filters, O, R, Len)
#         Wc_vectors = mat_contents['Wc_v']
#         # 如果是一维展开的，需要根据 O,R,Len 进行重塑；这里要求文件已存储为多维数组
#         # 为通用性，直接假设 Wc_vectors 形状为 (num_filters, O, R, Len)
#         if Wc_vectors.ndim == 2:
#             # 若为 (num_filters, -1)，尝试按 O,R,Len 重塑（需要外部指定 O,R 或从文件名推断）
#             # 这里抛出提示，要求用户确保数据格式正确
#             raise ValueError("预训练滤波器应为 4 维数组 (num_filters, O, R, Len)，请检查 MAT_FILE 内容")
#         return torch.from_numpy(Wc_vectors).float()

#     def noise_cancellation(self, Dis, Fx, filter_index, Stepsize):
#         """
#         多通道噪声消除主函数。
#         参数:
#             Dis:         期望信号（误差麦克风信号），形状 (N, E)，N 为样本点数
#             Fx:          滤波参考信号，形状 (N, R)
#             filter_index: 每秒对应的固定滤波器索引列表，长度为 ceil(N/fs)
#             Stepsize:    学习率（标量）
#         返回:
#             error_signal: 误差信号列表，形状 (N, E) 的 numpy 数组
#         """
#         Dis = torch.tensor(Dis, dtype=torch.float32)  # (N, E)
#         Fx  = torch.tensor(Fx,  dtype=torch.float32)  # (N, R)
#         N = Dis.shape[0]
#         E = Dis.shape[1]
#         assert E == self.E, f"期望信号通道数 {E} 与初始化时 {self.E} 不一致"
#         assert Fx.shape[1] == self.R, f"滤波参考通道数 {Fx.shape[1]} 与初始化时 {self.R} 不一致"

#         # 初始滤波器系数（第一秒对应的固定滤波器）
#         current_index = filter_index[0]
#         initial_coeffs = self.control_filters[current_index]  # (O, R, Len)

#         # 创建多通道 FxNLMS 实例
#         model = MIMO_FxNLMS(self.filter_len, self.R, self.O, self.E, initial_coeffs)
#         optimizer = optim.SGD([model.Wc], lr=Stepsize)

#         error_signal = []  # 存储每个样本所有误差通道的误差

#         for ii in range(N):
#             x_f = Fx[ii]       # (R,)
#             d   = Dis[ii]      # (E,)

#             y, power = model.feedforward(x_f)   # y: (O,), power: 标量
#             loss, e = model.LossFunction(y, d, power)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             error_signal.append(e.detach().numpy())  # 存储 (E,) 数组

#             # 每秒结束时检查是否切换固定滤波器
#             if (ii + 1) % self.fs == 0:
#                 next_sec = (ii + 1) // self.fs
#                 if next_sec < len(filter_index):
#                     new_index = filter_index[next_sec]
#                     if new_index != current_index:
#                         new_coeffs = self.control_filters[new_index]  # (O, R, Len)
#                         model.set_coeffs(new_coeffs)
#                         current_index = new_index

#         # 将列表转换为 (N, E) 的 numpy 数组
#         return np.array(error_signal)