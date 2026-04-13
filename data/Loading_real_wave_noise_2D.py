import os 
import torch
import torchaudio
import torchaudio.transforms as T
import librosa

def minmaxscaler(data):
    """
    最小-最大归一化函数
    
    参数:
        data: 输入数据
        
    返回:
        归一化后的数据
    """
    min = data.min()
    max = data.max()  
    return (data)/(max-min)


def resample_wav(waveform, sample_rate, resample_rate):
    """
    音频重采样函数
    
    参数:
        waveform: 原始音频波形数据
        sample_rate: 原始采样率
        resample_rate: 目标采样率
        
    返回:
        重采样后的音频波形
    """
    resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform


class transforms_construction():
    """
    音频变换构造类，用于构建不同类型的频谱图变换
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mel=64, TwoD_nfft=256, TwoD_Hop=128):
        """
        初始化变换参数
        
        参数:
            sample_rate: 采样率，默认16000Hz
            n_fft: FFT窗口大小，默认1024
            hop_length: 帧移，默认512
            n_mel: Mel滤波器数量，默认64
            TwoD_nfft: 二维频谱图的FFT大小，默认256
            TwoD_Hop: 二维频谱图的帧移，默认128
        """
        self.Sample_Rate = sample_rate
        self.N_FFT = n_fft
        self.Hop_Num = hop_length
        self.Mel_Num = n_mel
        self.TwoD_FFT = TwoD_nfft
        self.TwoD_Hop = TwoD_Hop
    
    def __transformation__(self, Type = 'Mel' ):
        """
        创建音频变换
        
        参数:
            Type: 变换类型，'Mel'为Mel频谱图，'Spec'为普通频谱图
            
        返回:
            音频变换对象
        """
        if Type == 'Mel':
            transformation = torchaudio.transforms.MelSpectrogram(sample_rate=self.Sample_Rate, n_fft=self.N_FFT, hop_length=self.Hop_Num, n_mels=self.Mel_Num) # 输出形状 [1, 64, 32]
        elif Type == 'Spec':
            transformation = torchaudio.transforms.Spectrogram(n_fft=self.TwoD_FFT, hop_length=self.TwoD_Hop, power=2, center=False, onesided=True) # 输出形状 [1, 129, 124]
        else:
            transformation = None
        return transformation
    

def loading_real_wave_noise(folde_name, sound_name):
    """
    加载真实波形噪声文件并重采样到16000Hz
    
    参数:
        folde_name: 文件夹名称
        sound_name: 音频文件名
        
    返回:
        waveform: 加载并重采样后的波形数据
        resample_rate: 重采样后的采样率（16000Hz）
    """
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folde_name, sound_name)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate)
    return waveform, resample_rate


def waveform_to_spectorgram(waveform):
    """
    将波形转换为Mel频谱图
    
    参数:
        waveform: 输入音频波形
        
    返回:
        spectorgram: 转换后的Mel频谱图（dB单位）
    """
    waveform = minmaxscaler(waveform) # 最小-最大归一化
    trasformation = transforms_construction().__transformation__(Type='Mel')
    spectorgram = trasformation(waveform)
    spectorgram = librosa.core.power_to_db(spectorgram) # 转换为分贝（dB）单位
    spectorgram = torch.from_numpy(spectorgram)
    return spectorgram