import numpy as np
import torch
import sounddevice as sd
from scipy.io import loadmat
from Network import m6_res                # 预训练 CNN 模型

"""
实时 SFANC-FxNLMS 算法运行函数
"""

# 配置文件
try:
    from config import anc_config
except ImportError:
    anc_config = {
        'fs': 16000,
        'frameSize': 1024,
        'micNumChannels': 6,               # 总输入通道数（1参考 + 5误差）
        'micChannels': {
            'reference': [1],              # 参考麦克风通道（1-based）
            'error': [2, 3, 4, 5, 6]       # 误差麦克风通道（1-based）
        },
        'spkChannels': [1, 2],             # 扬声器通道（1-based）
        'classify_interval': 16,           # 每 16 帧重新分类一次（约 1 秒）
        'mu': 0.01,                        # 步长
        'delta': 0.01                      # 正则化项
    }

class RealtimeMIMOANCFxNLMS:
    def __init__(self):
        # 基本参数
        self.fs = anc_config['fs']
        self.frame_size = anc_config['frameSize']
        self.mic_channels = anc_config['micChannels']
        self.spk_channels = anc_config['spkChannels']
        self.num_ref = len(self.mic_channels['reference'])       # 1
        self.num_error = len(self.mic_channels['error'])         # 5
        self.num_spk = len(self.spk_channels)                    # 2
        self.classify_interval = anc_config['classify_interval']
        self.mu = anc_config['mu']
        self.delta = anc_config['delta']

        # 加载预训练 CNN 模型并设为评估模式
        self.model = m6_res
        self.model.eval()

        # 1. 先加载次级路径矩阵，确定滤波器长度
        self.secondary_path = self.load_secondary_path_matrix(
            'Primary and Secondary Path/secondary_path_5mic_4spk.npy'
        )
        self.filter_len = self.secondary_path.shape[2]

        # 2. 加载预训练控制滤波器
        self.control_filters = self.load_control_filters(
            'Trained models/Pretrained_Control_filters.mat'
        )
        # control_filters 形状应为 (filter_len, num_spk, num_filters)
        self.num_filters = self.control_filters.shape[2]

        # 初始化控制滤波器系数（每个扬声器一个滤波器）
        self.W = np.zeros((self.num_spk, self.filter_len), dtype=np.float32)
        self.current_filter_idx = 0
        self.W[:, :] = self.control_filters[:, :, 0].T   # 默认第一个类别

        # 延迟线：原始参考信号（每个扬声器独立）
        self.raw_delay = np.zeros((self.num_spk, self.filter_len), dtype=np.float32)
        # 滤波后参考信号延迟线：每个(误差, 扬声器)对独立
        self.filtered_delay = np.zeros((self.num_error, self.num_spk, self.filter_len), dtype=np.float32)

        # 用于卷积的状态保持（每个次级路径单独保持）
        self.conv_buffers = np.zeros((self.num_error, self.num_spk, self.filter_len - 1), dtype=np.float32)

        # 分类相关变量
        self.frame_count = 0
        self.buffer_for_classify = np.array([], dtype=np.float32)

        # 音频设备设置
        self.setup_audio_devices()

    def load_control_filters(self, mat_file):
        """
        加载预训练控制滤波器
        设计形状: [filter_len, num_spk, num_filters]
        """
        print(f"加载控制滤波器: {mat_file}")
        data = loadmat(mat_file)
        Wc = data['Wc_v'].astype(np.float32)
        print(f"控制滤波器形状: {Wc.shape} (len, spk, categories)")
        return Wc

    def load_secondary_path_matrix(self, mat_file):
        """
        加载次级路径矩阵
        设计形状: [num_error, num_spk, filter_len] 或 [filter_len, num_error, num_spk]
        返回形状: [num_error, num_spk, filter_len]
        """
        print(f"加载次级路径矩阵: {mat_file}")
        data = np.load(mat_file)
        S = data.astype(np.float32)
        print(f"次级路径矩阵形状: {S.shape} (error, spk, filter_len)")
        return S

    # def load_control_filters(self, mat_file):
    #     print(f"加载控制滤波器: {mat_file}")
    #     data = loadmat(mat_file)
    #     Wc = data['Wc_v'].astype(np.float32)
    #     print(f"原始控制滤波器形状: {Wc.shape}")
    #     # 假设形状为 (num_filters, filter_len) 或 (filter_len, num_filters)
    #     if Wc.ndim == 2:
    #         # 判断哪一维可能是滤波器长度（应该等于 self.filter_len 或更大）
    #         if Wc.shape[1] == self.filter_len:
    #             # (num_filters, filter_len) -> (filter_len, 1, num_filters)
    #             Wc = Wc.T.reshape(self.filter_len, 1, -1)
    #         elif Wc.shape[0] == self.filter_len:
    #             # (filter_len, num_filters) -> (filter_len, 1, num_filters)
    #             Wc = Wc.reshape(self.filter_len, 1, -1)
    #         else:
    #             # 都不匹配，尝试将较长的维度作为 filter_len
    #             if Wc.shape[0] > Wc.shape[1]:
    #                 filter_len_candidate = Wc.shape[0]
    #                 Wc = Wc.reshape(filter_len_candidate, 1, -1)
    #             else:
    #                 filter_len_candidate = Wc.shape[1]
    #                 Wc = Wc.T.reshape(filter_len_candidate, 1, -1)
    #             print(f"警告：滤波器长度 {filter_len_candidate} 与次级路径长度 {self.filter_len} 不一致")
    #             # 这里可以选择截断或补零，见下文
    #     elif Wc.ndim == 3:
    #         # 已经是预期形状，但需验证维度顺序
    #         pass
    #     else:
    #         raise ValueError(f"不支持的控制滤波器维度: {Wc.ndim}")
    #     print(f"重塑后控制滤波器形状: {Wc.shape} (filter_len, num_spk, num_filters)")
    #     return Wc

    # def load_secondary_path_matrix(self, mat_file):
    #     print(f"加载次级路径矩阵: {mat_file}")
    #     data = np.load(mat_file)
    #     # 尝试获取次级路径数据，常见变量名 'S', 'S_primary', 'secondary' 等
    #     S = None
    #     for key in ['S', 'secondary', 'sec_path', 'S_est']:
    #         if key in data:
    #             S = data[key]
    #             break
    #     if S is None:
    #         # 如果都没有，取第一个非__的变量
    #         for k, v in data.items():
    #             if not k.startswith('__'):
    #                 S = v
    #                 break
    #     S = S.astype(np.float32)
    #     print(f"原始次级路径形状: {S.shape}")
        
    #     # 如果是一维或二维 (filter_len, 1) 则视为 SISO
    #     if S.ndim == 1:
    #         filter_len = S.shape[0]
    #         S = S.reshape(1, 1, filter_len)
    #     elif S.ndim == 2 and S.shape[1] == 1:
    #         filter_len = S.shape[0]
    #         S = S.reshape(1, 1, filter_len)
    #     elif S.ndim == 3:
    #         # 假设形状可能是 (num_error, num_spk, filter_len) 或 (filter_len, num_error, num_spk)
    #         # 检查哪个维度长度匹配 num_error/num_spk
    #         if S.shape[0] == self.num_error and S.shape[1] == self.num_spk:
    #             pass  # 已经是正确顺序
    #         elif S.shape[2] == self.num_error and S.shape[1] == self.num_spk:
    #             S = np.transpose(S, (2, 1, 0))  # (filter_len, num_spk, num_error) -> (num_error, num_spk, filter_len)
    #         elif S.shape[0] == self.num_spk and S.shape[1] == self.num_error:
    #             S = np.transpose(S, (1, 0, 2))
    #         else:
    #             raise ValueError(f"无法自动推断次级路径维度，形状 {S.shape}，期望 (num_error={self.num_error}, num_spk={self.num_spk}, filter_len)")
    #     else:
    #         raise ValueError(f"次级路径维度异常: {S.ndim}D, 形状 {S.shape}")
        
    #     print(f"重塑后次级路径形状: {S.shape} (error, spk, filter_len)")
    #     return S

    def setup_audio_devices(self):
        """查找并验证音频设备"""
        devices = sd.query_devices()
        print("可用的音频设备:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (输入通道: {dev['max_input_channels']}, 输出通道: {dev['max_output_channels']})")

        required_input_channels = self.num_ref + self.num_error
        self.input_device = sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device
        required_output_channels = self.num_spk
        self.output_device = sd.default.device[1] if isinstance(sd.default.device, tuple) else sd.default.device

        try:
            input_info = sd.query_devices(self.input_device)
            output_info = sd.query_devices(self.output_device)
            if input_info['max_input_channels'] < required_input_channels:
                print(f"警告: 输入设备 {input_info['name']} 仅有 {input_info['max_input_channels']} 通道，需要 {required_input_channels}")
            if output_info['max_output_channels'] < required_output_channels:
                print(f"警告: 输出设备 {output_info['name']} 仅有 {output_info['max_output_channels']} 通道，需要 {required_output_channels}")
            print(f"当前输入设备: {input_info['name']} (通道数: {input_info['max_input_channels']})")
            print(f"当前输出设备: {output_info['name']} (通道数: {output_info['max_output_channels']})")
        except Exception as e:
            print(f"设备验证失败: {e}")
            raise

    def classify_noise(self, audio_segment):
        """使用 CNN 对一段音频进行分类，返回滤波器索引"""
        input_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            idx = torch.argmax(output).item()
        return idx

    def update_filter_if_needed(self, new_idx):
        """切换所有扬声器的滤波器系数"""
        if new_idx != self.current_filter_idx:
            self.current_filter_idx = new_idx
            self.W[:, :] = self.control_filters[:, :, new_idx].T
            print(f"滤波器切换到索引 {new_idx}")

    def process_frame(self, ref_frame, err_frame):
        """
        处理一帧音频
        :param ref_frame: 参考麦克风信号，形状 (frame_size,)
        :param err_frame: 误差麦克风信号，形状 (frame_size, num_error)
        :return: 控制信号，形状 (frame_size, num_spk)
        """
        # 1. 定期进行分类
        self.frame_count += 1
        if self.frame_count % self.classify_interval == 0:
            self.buffer_for_classify = np.concatenate([self.buffer_for_classify, ref_frame])
            if len(self.buffer_for_classify) >= self.fs:
                segment = self.buffer_for_classify[-self.fs:]
                new_idx = self.classify_noise(segment)
                self.update_filter_if_needed(new_idx)
                self.buffer_for_classify = np.array([], dtype=np.float32)

        # 2. 生成滤波-x信号矩阵
        filtered_ref = np.zeros((self.num_error, self.num_spk, self.frame_size), dtype=np.float32)
        for m in range(self.num_error):
            for l in range(self.num_spk):
                input_with_buffer = np.concatenate([self.conv_buffers[m, l], ref_frame])
                conv_full = np.convolve(input_with_buffer, self.secondary_path[m, l], mode='full')
                filtered_ref[m, l, :] = conv_full[len(self.conv_buffers[m, l]):len(self.conv_buffers[m, l]) + self.frame_size]
                self.conv_buffers[m, l] = conv_full[-(self.filter_len - 1):]

        # 3. 逐样本 MIMO FxNLMS 更新
        output_frame = np.zeros((self.frame_size, self.num_spk), dtype=np.float32)

        W = self.W
        raw_delay = self.raw_delay
        filtered_delay = self.filtered_delay

        for i in range(self.frame_size):
            x_raw = ref_frame[i]
            x_filt_slice = filtered_ref[:, :, i]
            d = err_frame[i, :]

            # 计算控制信号
            y = np.zeros(self.num_spk, dtype=np.float32)
            for l in range(self.num_spk):
                raw_delay[l] = np.roll(raw_delay[l], 1)
                raw_delay[l, 0] = x_raw
                y[l] = np.dot(W[l], raw_delay[l])

            e = d  # 直接使用测量到的误差信号

            # 更新滤波器
            for l in range(self.num_spk):
                power_sum = 0.0
                grad = np.zeros(self.filter_len, dtype=np.float32)
                for m in range(self.num_error):
                    filtered_delay[m, l] = np.roll(filtered_delay[m, l], 1)
                    filtered_delay[m, l, 0] = x_filt_slice[m, l]
                    grad += e[m] * filtered_delay[m, l]
                    power_sum += np.dot(filtered_delay[m, l], filtered_delay[m, l])
                power = power_sum + self.delta
                W[l] += self.mu / power * grad

            output_frame[i, :] = y

        self.W = W
        self.raw_delay = raw_delay
        self.filtered_delay = filtered_delay

        return output_frame

    def callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"音频状态: {status}")

        ref_frame = indata[:, 0].copy()
        err_frame = indata[:, 1:1+self.num_error].copy()
        output_frame = self.process_frame(ref_frame, err_frame)

        outdata[:, 0] = output_frame[:, 0]
        if self.num_spk > 1:
            outdata[:, 1] = output_frame[:, 1]

    def start(self):
        print("开始多通道实时噪声控制...")
        print(f"采样率: {self.fs} Hz, 帧大小: {self.frame_size}")
        print(f"扬声器数: {self.num_spk}, 误差麦克风数: {self.num_error}")
        print("按 Enter 停止...")

        try:
            with sd.Stream(
                callback=self.callback,
                channels=(self.num_ref + self.num_error, self.num_spk),
                samplerate=self.fs,
                blocksize=self.frame_size,
                device=[self.input_device, self.output_device]
            ):
                input()
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            print("实时噪声控制已停止")

if __name__ == "__main__":
    processor = RealtimeMIMOANCFxNLMS()
    processor.start()