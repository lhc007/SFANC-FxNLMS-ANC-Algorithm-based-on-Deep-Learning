import os 
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

#--------------------------------------------------------------
# 函数: print_stats()
# 描述: 打印音频文件的统计信息
#--------------------------------------------------------------
def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
        print("Shape:", tuple(waveform.shape))
        print("Dtype:", waveform.dtype)
        print(f" - Max:     {waveform.max().item():6.3f}")
        print(f" - Min:     {waveform.min().item():6.3f}")
        print(f" - Mean:    {waveform.mean().item():6.3f}")
        print(f" - Std Dev: {waveform.std().item():6.3f}")
        print()
        print(waveform)
        print()

#-------------------------------------------------------------- 
# 函数: plot_waveform()
# 描述: 绘制音频文件的波形图
#--------------------------------------------------------------
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=True)

#--------------------------------------------------------------
# 函数: plot_specgram()
# 描述: 绘制音频文件的功率谱图
#-------------------------------------------------------------
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=True)

#--------------------------------------------------------------
# 函数: resample_wav()
# 描述: 使用重采样率对波形进行重采样
#--------------------------------------------------------------
def resample_wav(waveform, sample_rate, resample_rate):
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform

#--------------------------------------------------------------
# 函数: loading_real_wave_noise()
# 描述: 加载原始飞机噪声并进行重采样
#--------------------------------------------------------------
def loading_real_wave_noise(folde_name, sound_name):
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folde_name, sound_name)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH, backend="soundfile")
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate)
    return waveform, resample_rate

#----------------------------------------------------------------
# 主函数: 加载并分析音频文件
def main():
    folde_name = 'Real_noise'
    sound_name = 'Aircraft.wav'
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folde_name, sound_name)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
    waveform = resample_wav(waveform, sample_rate, 16000)
    sample_rate = 16000
    print_stats(waveform, sample_rate=sample_rate)
    plot_waveform(waveform, sample_rate)
    plot_specgram(waveform, sample_rate)

if __name__ == "__main__":
    main()