#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFANC-FxNLMS 主动噪声控制算法运行脚本

本脚本整合了SFANC-FxNLMS for ANC_zh.ipynb中的所有代码，
实现了选择性固定滤波器主动噪声控制（SFANC）与滤波-x归一化最小均方（FxNLMS）算法的混合实现。

作者: 基于原始笔记本代码
日期: 2024年
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat
import math
import os
import sys

# 添加自定义模块路径
sys.path.append(os.path.dirname(__file__))

# 导入自定义模块
try:
    from data.loading_real_wave_noise import loading_real_wave_noise
    from Reading_path_test import loading_paths_from_MAT
    from algorithm.FxNLMS_algorithm import FxNLMS, train_fxnlms_algorithm
    from algorithm.Control_filter_selection import Control_filter_selection
    from data.Disturbance_generation import Disturbance_generation_from_real_noise
    from algorithm.Combine_SFANC_with_FxNLMS import SFANC_FxNLMS
except ImportError as e:
    print(f"导入模块错误: {e}")
    print("请确保项目结构正确，模块文件存在")
    sys.exit(1)

def check_gpu():
    """检查GPU是否可用"""
    gpu_available = torch.cuda.is_available()
    print(f'GPU可用性: {gpu_available}')
    return gpu_available

def load_and_preprocess_data():
    """数据加载与预处理"""
    print("=== 数据加载与预处理 ===")
    
    # 参数设置
    fs = 16000  # 采样率：16kHz
    StepSize = 0.0001  # FxNLMS算法的学习率
    sound_name = 'Traffic'  # 噪声类型：交通噪声
    
    print(f"采样率: {fs} Hz")
    print(f"学习率: {StepSize}")
    print(f"噪声类型: {sound_name}")
    
    # 从WAV文件加载噪声波形数据
    print("加载噪声数据...")
    waveform, resample_rate = loading_real_wave_noise(
        folde_name='Real Noise Examples/',  # 噪声文件所在目录
        sound_name=sound_name+'.wav'  # 噪声文件名
    )
    
    # 加载主路径和次级路径
    print("加载路径数据...")
    Pri_path, Secon_path = loading_paths_from_MAT(
        folder='Primary and Secondary Path',  # 路径文件所在主目录
        subfolder='',  # 子目录
        Pri_path_file_name='Primary_path.mat',  # 主路径文件名
        Sec_path_file_name='Secondary_path.mat'  # 次级路径文件名
        # Sec_path_file_name='secondary_path_5mic_4spk.npy'  # 次级路径文件名
    )
    
    # 从真实噪声生成干扰信号
    print("生成干扰信号...")
    Dis, Fx, Re = Disturbance_generation_from_real_noise(
        fs=fs,  # 采样率
        Repet=0,  # 重复次数
        wave_form=waveform,  # 输入波形
        Pri_path=Pri_path,  # 主路径
        Sec_path=Secon_path  # 次级路径
    )
    
    # 打印数据形状，验证数据加载正确性
    print('原始波形形状:', waveform.shape)
    print('重复波形形状:', Re.shape)
    print('干扰信号形状:', Dis.shape)
    
    # 设置matplotlib参数，处理大数据集绘图
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000  # 增加路径块大小以避免绘图错误
    
    return fs, StepSize, Re, Dis, Fx

def run_fxnlms_algorithm(fs, StepSize, Dis, Fx):
    """运行FxNLMS算法"""
    print("\n=== FxNLMS 算法实现 ===")
    
    # 初始化FxNLMS控制器，滤波器长度为1024
    print("初始化FxNLMS控制器...")
    controller = FxNLMS(Len=1024)
    
    # 训练FxNLMS算法
    print("训练FxNLMS算法...")
    ErrorFxNLMS = train_fxnlms_algorithm(
        Model=controller,  # FxNLMS模型
        Ref=Fx,  # 参考信号（filtered-x信号）
        Disturbance=Dis,  # 干扰信号
        Stepsize=StepSize  # 学习率
    )
    
    # 创建时间轴用于绘图
    Time = np.arange(len(Dis)) / fs
    
    # 中文显示设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制FxNLMS算法结果
    print("绘制FxNLMS算法结果...")
    plt.figure(figsize=(10, 6))
    plt.title('FxNLMS算法性能对比')
    plt.plot(Time, Dis, color='blue', label='ANC关闭', linewidth=0.5)
    plt.plot(Time, ErrorFxNLMS, color='green', label='ANC开启', linewidth=0.5)
    plt.ylabel('幅度')
    plt.xlabel('时间（秒）')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 确保pdf目录存在
    os.makedirs('pdf', exist_ok=True)
    plt.savefig('atlas/FxNLMS.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return ErrorFxNLMS, Time


def select_control_filter(fs, Re):
    """控制滤波器选择"""
    print("\n=== 控制滤波器选择 ===")
    
    # 使用CNN模型预测最适合当前噪声的控制滤波器索引
    print("使用CNN模型选择控制滤波器...")
    id_vector = Control_filter_selection(
        fs=16000,  # 采样率
        Primary_noise=Re.unsqueeze(0)  # 主噪声，增加批次维度
    )
    
    print('选择的控制滤波器索引:', id_vector)
    return id_vector


def run_sfanc_fxnlms_hybrid(fs, StepSize, Dis, Fx, id_vector):
    """运行SFANC-FxNLMS混合算法"""
    print("\n=== SFANC-FxNLMS 混合算法 ===")
    
    # 预训练控制滤波器文件路径
    FILE_NAME_PATH = 'Trained models/Pretrained_Control_filters.mat'
    
    # 创建SFANC-FxNLMS混合算法实例
    print("创建SFANC-FxNLMS混合算法实例...")
    SFANC_FxNLMS_Cancellation = SFANC_FxNLMS(
        MAT_FILE=FILE_NAME_PATH,  # 预训练滤波器文件
        fs=16000  # 采样率
    )
    
    # 执行噪声消除
    print("执行噪声消除...")
    Error_SFANC_FxNLMS = SFANC_FxNLMS_Cancellation.noise_cancellation(
        Dis=Dis,  # 干扰信号
        Fx=Fx,  # filtered-x信号
        filter_index=id_vector,  # 选择的滤波器索引
        Stepsize=StepSize  # 学习率
    )
    
    # 绘制混合算法结果
    print("绘制混合算法结果...")
    Time = np.arange(len(Dis)) / fs
    
    # 中文显示设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))
    plt.title('SFANC-FxNLMS混合算法性能对比')
    plt.plot(Time, Dis, color='blue', label='ANC关闭', linewidth=0.5)
    plt.plot(Time, Error_SFANC_FxNLMS, color='red', label='ANC开启', linewidth=0.5)
    plt.ylabel('幅度')
    plt.xlabel('时间（秒）')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig('atlas/SFANC_FxNLMS.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return Error_SFANC_FxNLMS


def performance_comparison(Dis, ErrorFxNLMS, Error_SFANC_FxNLMS, fs):
    """性能对比分析"""
    print("\n=== 性能对比分析 ===")
    
    # 确保数据为 numpy 数组
    ErrorFxNLMS = np.array(ErrorFxNLMS)
    Error_SFANC_FxNLMS = np.array(Error_SFANC_FxNLMS)
    Dis = np.asarray(Dis)
    
    # 计算均方根误差（RMSE）
    rmse_original = np.sqrt(np.mean(Dis**2))
    rmse_fxnlms = np.sqrt(np.mean(ErrorFxNLMS**2))
    rmse_hybrid = np.sqrt(np.mean(Error_SFANC_FxNLMS**2))
    
    # 计算信噪比改善（SNR Improvement）
    snr_original = 10 * np.log10(np.var(Dis) / np.var(Dis))  # 原始SNR
    snr_fxnlms = 10 * np.log10(np.var(Dis) / np.var(ErrorFxNLMS))
    snr_hybrid = 10 * np.log10(np.var(Dis) / np.var(Error_SFANC_FxNLMS))
    
    print('=== 性能对比结果 ===')
    print(f'原始信号RMSE: {rmse_original:.4f}')
    print(f'FxNLMS算法RMSE: {rmse_fxnlms:.4f}')
    print(f'混合算法RMSE: {rmse_hybrid:.4f}')
    print(f'FxNLMS SNR改善: {snr_fxnlms:.2f} dB')
    print(f'混合算法SNR改善: {snr_hybrid:.2f} dB')
    
    # 绘制三种情况的对比图
    Time = np.arange(len(Dis)) / fs
    
    # 中文显示设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(Time, Dis, color='black', label='原始信号', linewidth=0.5)
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('噪声控制算法性能对比')
    
    plt.subplot(3, 1, 2)
    plt.plot(Time, ErrorFxNLMS, color='green', label='FxNLMS', linewidth=0.5)
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(Time, Error_SFANC_FxNLMS, color='red', label='SFANC-FxNLMS', linewidth=0.5)
    plt.ylabel('幅度')
    plt.xlabel('时间（秒）')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pdf/Comparison.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return {
        'rmse_original': rmse_original,
        'rmse_fxnlms': rmse_fxnlms,
        'rmse_hybrid': rmse_hybrid,
        'snr_fxnlms': snr_fxnlms,
        'snr_hybrid': snr_hybrid
    }


def save_results(results):
    """保存结果到文件"""
    print("\n=== 保存结果 ===")
    
    # 保存性能结果到文本文件
    with open('atlas/performance_results.txt', 'w', encoding='utf-8') as f:
        f.write("SFANC-FxNLMS 算法性能结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"原始信号RMSE: {results['rmse_original']:.4f}\n")
        f.write(f"FxNLMS算法RMSE: {results['rmse_fxnlms']:.4f}\n")
        f.write(f"混合算法RMSE: {results['rmse_hybrid']:.4f}\n")
        f.write(f"FxNLMS SNR改善: {results['snr_fxnlms']:.2f} dB\n")
        f.write(f"混合算法SNR改善: {results['snr_hybrid']:.2f} dB\n")
    
    print("性能结果已保存到 atlas/performance_results.txt")
    print("图表已保存到 atlas/ 目录")

def main():
    """主函数"""
    print("SFANC-FxNLMS 主动噪声控制算法运行脚本")
    print("=" * 50)
    
    try:
         # 1. 检查GPU
        check_gpu()
        
        # 2. 数据加载与预处理
        fs, StepSize, Re, Dis, Fx = load_and_preprocess_data()
        
        # 3. 运行FxNLMS算法
        ErrorFxNLMS, Time = run_fxnlms_algorithm(fs, StepSize, Dis, Fx)
        
        # 4. 控制滤波器选择
        id_vector = select_control_filter(fs, Re)
        
        # 5. 运行SFANC-FxNLMS混合算法
        Error_SFANC_FxNLMS = run_sfanc_fxnlms_hybrid(fs, StepSize, Dis, Fx, id_vector)
        
        # 6. 性能对比分析
        results = performance_comparison(Dis, ErrorFxNLMS, Error_SFANC_FxNLMS, fs)
        
        # 7. 保存结果
        save_results(results)

        print("\n=== 算法运行完成 ===")
        print("所有处理步骤已完成，结果文件已生成")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("算法运行失败，请检查文件路径和依赖项")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()