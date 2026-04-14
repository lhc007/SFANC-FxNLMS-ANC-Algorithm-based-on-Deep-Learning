import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import repeat 
from scipy import signal, misc
import torch

#-------------------------------------------------------------
# 函数: Disturbance_reference_generation()
# 描述: 使用默认参数生成干扰信号和参考信号
#-------------------------------------------------------------
def Disturbance_reference_generation():
    # 定义ANC系统的配置
    fs = 16000 # 采样率
    T = 5 # 模拟时间
    t = np.arange(0,T,1/fs).reshape(-1,1)
    f0 = 500
    
    # 生成参考信号
    Re = np.random.randn(len(t))
    
    # 定义低通滤波器
    f_cutoff = 2000 
    N_fc = f_cutoff/fs 
    b1, b2 = signal.firwin(128, N_fc), signal.firwin(128, 2*N_fc)
    
    # 构建主路径
    Pri_path = signal.convolve(b1,b2)
    w1, h1 = signal.freqz(Pri_path)
    w2, h2 = signal.freqz(b2)

    # 绘制低通滤波器的频谱响应图
    plt.title('Digital filter frequency response')
    plt.plot(w1, 20*np.log10(np.abs(h1)),'b')
    plt.plot(w2, 20*np.log10(np.abs(h2)),'r')
    plt.ylabel('Amplitude Response (dB)')
    plt.xlabel('Frequency (rad/sample)')
    plt.grid()
    plt.show()

    # 绘制主路径的脉冲响应图
    plt.title('The response of the primary path')
    plt.plot(b1)
    plt.ylabel('Amplitude')
    plt.xlabel('Length (taps)')
    plt.grid()
    plt.show()
    
    # 构建所需信号
    Dir, Fx = signal.lfilter(Pri_path, 1, Re), signal.lfilter(b2, 1, Re)
    print(Fx[1])
    print(Dir.shape, Fx.shape)

    # 绘制干扰信号的频谱图
    f, Pper_spec = signal.periodogram(Dir, fs, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pper_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()

    # 绘制参考信号的频谱图
    f, Pper_spec = signal.periodogram(Fx, fs, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pper_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)

#-------------------------------------------------------------
# 函数: Disturbance_reference_generation_from_Fvector()
# 描述: 根据定义的频率向量生成干扰信号和参考信号
#-------------------------------------------------------------
def Disturbance_reference_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    # Pri_path 和 Sec_path 是一维数组
    # 构建带通滤波器
    # 生成随机噪声
    t = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    b2 = signal.firwin(len_f, [f_vector[0],f_vector[1]], pass_zero='bandpass', window ='hamming', fs=fs)
    
    xin = np.random.randn(len(t))
    Re = signal.lfilter(b2,1,xin) # 随机噪声通过带通滤波器，得到干扰信号
    Noise = Re[len_f-1:]
    
    # 构建所需信号
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)

#-------------------------------------------------------------
# 函数: Disturbance_generation_from_real_noise()
# 描述: 从原始波形生成干扰信号和滤波后的参考信号
#-------------------------------------------------------------
def Disturbance_generation_from_real_noise(fs, Repet, wave_form, Pri_path, Sec_path):
    wave = wave_form[0,:].numpy()
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec,wave),axis=0) # 通过重复增加波形的长度
    pass

    # 构建所需信号
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec), signal.lfilter(Sec_path, 1, wavec)
    
    N = len(Dir)
    N_z = N//fs
    Dir, Fx = Dir[0:N_z*fs], Fx[0:N_z*fs]
    
    # torch.from_numpy(Dir)：将 NumPy 数组 Dir 转换为 PyTorch 张量（Tensor），两者共享内存（零拷贝）。
    # .type(torch.float)：将张量的数据类型设置为 torch.float，即 32 位浮点数（float32）。
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(wavec).type(torch.float)

#-------------------------------------------------------------
# 函数: Varied_distrubance_reference_generation_from_Fvector()
# 描述: 根据变化的频率向量生成干扰信号和参考信号
#-------------------------------------------------------------
def Varied_distrubance_reference_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    t = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    for ii in range(len(f_vector)):
        b2 = signal.firwin(len_f, f_vector[ii], pass_zero='bandpass', window ='hamming',fs=fs)
        xin = np.random.randn(len(t))
        Re = signal.lfilter(b2,1,xin)
        if ii == 0:
            Noise = Re[fs:]
        else:
            if ii ==2 :
                Noise = np.concatenate((Noise, 4*Re[fs:]),axis=0)
            else: 
                Noise = np.concatenate((Noise, Re[fs:]),axis=0)
        
    # 构建所需信号
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(Noise).type(torch.float)