import torch

from Modified_ShufflenetV2 import Modified_ShufflenetV2
from Loading_real_wave_noise_2D import waveform_to_spectorgram


def load_weigth_for_model(model, pretrained_path):
    """
    为模型加载预训练权重
    
    参数:
        model: 需要加载权重的模型
        pretrained_path: 预训练权重文件路径
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location="cpu")
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]    
    model.load_state_dict(model_dict)


def minmaxscaler(data):
    """
    最小-最大归一化
    
    参数:
        data: 输入数据
        
    返回:
        归一化后的数据
    """
    min = data.min()
    max = data.max()    
    return (data)/(max-min)


def Casting_multiple_time_length_of_primary_noise(primary_noise, fs):
    """
    调整主噪声的长度，使其为采样率的整数倍
    
    参数:
        primary_noise: 主噪声数据，形状应为 [1 x 采样点数]
        fs: 采样率
        
    返回:
        调整长度后的主噪声
    """
    assert  primary_noise.shape[0] == 1, '主噪声的维度应为 [1 x 采样点数] !!!'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1]%fs
    return primary_noise[:,:cast_len] # 使主噪声的长度为采样率的整数倍


#-------------------------------------------------------------
# 类 : Control_filter_Index_predictor
#-------------------------------------------------------------
class Control_filter_Index_predictor():
    """
    控制滤波器索引预测器类，使用预训练的CNN模型预测噪声类型
    """
    
    def __init__(self, MODEL_PATH, device, fs):
        """
        初始化预测器
        
        参数:
            MODEL_PATH: 预训练模型路径
            device: 计算设备（CPU或GPU）
            fs: 采样率
        """
        self.device = device
        # 设置模型
        model = Modified_ShufflenetV2(num_classes=7)
        model = model.to(self.device)
        # 加载模型权重
        load_weigth_for_model(model, MODEL_PATH)
        model.eval()
        
        
        self.model = model
        self.fs = fs
    
    def predic_ID(self, noise):
        """
        预测单段噪声的类别索引
        
        参数:
            noise: 输入的噪声数据
            
        返回:
            预测的类别索引
        """
        spectorgram = waveform_to_spectorgram(noise) # 转换为频谱图，形状为 [1, 64, 32]
        spectorgram = spectorgram.to(self.device)
        spectorgram = spectorgram.unsqueeze(0) # 添加批次维度，形状为 [1, 1, 64, 32]
        prediction = self.model(spectorgram) # 预测结果，形状为 [7]
        pred = torch.argmax(prediction).item()
        return pred
    
    def predic_ID_vector(self, primary_noise):
        """
        对每一秒的噪声帧进行分类预测，返回预测索引向量
        
        参数:
            primary_noise: 主噪声数据
            
        返回:
            ID_vector: 每一帧的预测索引列表
        """
        # 检查主噪声的长度
        assert  primary_noise.shape[0] == 1, '主噪声的维度应为 [1 x 采样点数] !!!'
        assert  primary_noise.shape[1] % self.fs == 0, '主噪声的长度不是采样率的整数倍。'
        
        # 计算主噪声包含多少秒
        Time_len = int(primary_noise.shape[1]/self.fs)
        
        # 构建主噪声矩阵 [时间帧数 x 1 x 采样率]
        primary_noise_vectors = primary_noise.reshape(Time_len, self.fs).unsqueeze(1)
        
        # 对每一帧（长度为1秒）进行噪声分类
        ID_vector = []
        for ii in range(Time_len):
            ID_vector.append(self.predic_ID(primary_noise_vectors[ii]))
        return ID_vector


def Control_filter_selection(fs=16000, Primary_noise=None):
    """
    控制滤波器选择函数，根据输入噪声选择合适的控制滤波器索引
    
    参数:
        fs: 采样率，默认16000Hz
        Primary_noise: 输入的主噪声数据
        
    返回:
        Id_vector: 预测的控制滤波器索引向量
    """
    # 预训练CNN模型路径
    MODEL_PATH = 'Trained models/ShuffleNetV2_Synthetic.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Pre_trained_control_filter_ID_pridector = Control_filter_Index_predictor(MODEL_PATH=MODEL_PATH, device=device, fs=fs)
    
    Primary_noise = Casting_multiple_time_length_of_primary_noise(Primary_noise, fs=fs)
    
    Id_vector = Pre_trained_control_filter_ID_pridector.predic_ID_vector(Primary_noise)
    
    return Id_vector