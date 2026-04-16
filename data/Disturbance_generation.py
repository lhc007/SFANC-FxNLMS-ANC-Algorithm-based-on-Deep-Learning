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
    #从 wave_form 中取出第 0 行、所有列的数据，转换为 NumPy 数组
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
    Dis = torch.from_numpy(Dir).type(torch.float)
    Fx = torch.from_numpy(Fx).type(torch.float)
    Re = torch.from_numpy(wavec).type(torch.float)

    return Dis, Fx, Re

def Disturbance_generation_from_real_noise_MIMO(fs, Repet, wave_form, Pri_path, Sec_path):
    """
    生成多通道干扰信号和原始参考信号（逻辑与单通道一致），返回值直接适配 MIMO_SFANC_FxNLMS

    参数:
        fs          : 采样率 (Hz)
        Repet       : 重复次数，用于扩展信号长度
        wave_form   : 原始噪声波形，形状 (I, N) 或 (N,)，I 为参考通道数
        Pri_path    : 主路径脉冲响应，形状 (I, K, L_pri) 或 (K, L_pri)（I=1）
        Sec_path    : 次级路径脉冲响应，形状 (J, K, L_sec)

    返回:
        Dis         : 干扰信号（主路径输出），形状 (K, N_out)
        Re          : 原始重复波形，形状 (I, N_out)
    """
    # 1. 将 wave_form 转换为 numpy 数组并展平（若为单通道）
    if torch.is_tensor(wave_form):
        wave = wave_form.cpu().numpy()
    else:
        wave = np.asarray(wave_form)

    # 若 wave 是一维，则转为 (1, N) 以便统一处理
    if wave.ndim == 1:
        print("wave is 1D, reshape to (1, N)")
        wave = wave.reshape(1, -1)
    I, N_orig = wave.shape

    # 2. 通过重复增加信号长度
    wavec = wave
    for _ in range(Repet):
        wavec = np.concatenate((wavec, wave), axis=1)   # 沿时间轴拼接
    # wavec 形状: (I, N_total)

    # 3. 解析主路径和次级路径的维度
    # 主路径 Pri_path: 期望形状 (I, K, L_pri)
    if Pri_path.ndim == 1:
        Pri_path = Pri_path.reshape(1, 1, -1)
    elif Pri_path.ndim == 2:
        if Pri_path.shape[0] == I:
            Pri_path = Pri_path.reshape(I, 1, -1)
        else:
            K = Pri_path.shape[0]
            Pri_path = Pri_path.reshape(1, K, -1)
    I_pri, K_pri, L_pri = Pri_path.shape
    assert I_pri == I, f"主路径参考通道数 {I_pri} 与 wave_form 通道数 {I} 不一致"

    # 次级路径 Sec_path: 形状 (J, K, L_sec)
    if Sec_path.ndim == 1:
        # 单通道次级路径，形状 (L_sec,) -> (1, 1, L_sec)
        Sec_path = Sec_path.reshape(1, 1, -1)
    elif Sec_path.ndim == 2:
        Sec_path = Sec_path.reshape(1, Sec_path.shape[0], -1)
    J, K_sec, L_sec = Sec_path.shape
    assert K_pri == K_sec, f"主路径误差通道数 {K_pri} 与次级路径误差通道数 {K_sec} 不一致"
    K = K_pri

    # 4. 计算干扰信号 Dis（主路径输出）
    N_total = wavec.shape[1]
    Dis = np.zeros((K, N_total))
    for i in range(I):
        for k in range(K):
            filtered = signal.lfilter(Pri_path[i, k], 1.0, wavec[i])
            Dis[k] += filtered

    # 5. （可选）计算滤波参考信号 Fx，但不再返回，因为算法内部会基于 Re 和 Sec_path 生成
    Fx = np.zeros((J, N_total))
    for j in range(J):
        for k in range(K):
            for i in range(I):
                filtered = signal.lfilter(Sec_path[j, k], 1.0, wavec[i])
                Fx[j] += filtered

    # 6. 截断到整秒数
    N_z = N_total // fs
    Dis = Dis[:, :N_z * fs]
    Re  = wavec[:, :N_z * fs]   # 原始重复波形，形状 (I, N_out)

    # 7. 转换为 torch 张量 (float32)
    Dis_t = torch.from_numpy(Dis.T).float()
    Re_t  = torch.from_numpy(Re.T).float()
    Fx_t = torch.from_numpy(Fx.T).float()
    
    return Dis_t, Re_t, Fx_t



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