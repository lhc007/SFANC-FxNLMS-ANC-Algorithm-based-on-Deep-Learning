# 混合SFANC-FxNLMS算法

描述：
这是SPL论文"基于深度学习的混合SFANC-FxNLMS主动噪声控制算法"的代码。
您可以在 <https://arxiv.org/pdf/2208.08082.pdf> 或IEEE Xplore上找到该论文。

该论文提出了一种混合SFANC-FxNLMS方法，以克服自适应算法收敛速度慢的问题，并提供比SFANC方法更好的噪声 reduction水平。设计了一个轻量级的一维卷积神经网络（1D CNN），用于自动为每帧主噪声选择最合适的预训练控制滤波器。同时，FxNLMS算法继续以采样率更新所选预训练控制滤波器的系数。
!\[Fig1\_00]\(https\://user-images.githubusercontent.com/95018034/163777818-985cac62-74fb-4585-84d4-c4d9b29fc0e6.png null)

平台：NVIDIA-SMI 466.47，驱动版本：466.47，CUDA版本：11.3

环境：Jupyter Notebook 6.4.5，Python 3.9.7，Pytorch 1.10.1

## 项目结构

```
SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning/
├── Primary and Secondary Path/     # 主路径和次级路径数据
│   ├── Primary_path.mat             # 主路径数据
│   └── Secondary_path.mat           # 次级路径数据
├── Real Noise Examples/             # 真实噪声示例
│   ├── Aircraft.wav                 # 飞机噪声
│   ├── Connect_Aircraft_Traffic.wav # 连接的飞机和交通噪声
│   ├── Mix_Aircraft_Traffic.wav     # 混合的飞机和交通噪声
│   └── Traffic.wav                  # 交通噪声
├── Trained models/                  # 预训练模型
│   ├── Pretrained_Control_filters.mat # 预训练控制滤波器
│   └── model.pth                    # 预训练的1D CNN模型
├── Bcolors.py                       # 颜色输出工具
├── Disturbance_generation.py        # 干扰信号生成
├── FxNLMS_algorithm.py              # FxNLMS算法实现
├── Network.py                       # 神经网络模型定义
├── README.md                        # 英文说明文档
├── README_zh.md                     # 中文说明文档
├── Reading_path_test.py             # 路径读取测试
├── SFANC-FxNLMS for ANC.ipynb       # Jupyter Notebook示例
├── loading_real_wave_noise.py       # 加载真实波形噪声
└── realtime_sfanc_fxnlms.py         # 实时噪声控制实现
```

## 模块功能介绍

### 1. Network.py

- **功能**：定义用于噪声分类的卷积神经网络模型
- **主要类**：
  - `CNN`：基础卷积神经网络模型
  - `ResBlock`：残差连接块
  - `CNNRes`：带残差连接的卷积神经网络
- **用途**：通过1D CNN对输入噪声进行分类，选择合适的预训练控制滤波器

### 2. FxNLMS\_algorithm.py

- **功能**：实现滤波-x归一化最小均方算法
- **主要类和函数**：
  - `FxNLMS`：FxNLMS算法类
  - `train_fxnlms_algorithm`：训练FxNLMS算法
  - `Generating_boardband_noise_wavefrom_tensor`：生成宽带噪声
- **用途**：更新控制滤波器系数，实现主动噪声控制

### 3. realtime\_sfanc\_fxnlms.py

- **功能**：实现实时噪声控制
- **主要类**：
  - `RealtimeSFANCFxNLMS`：实时噪声控制类
- **用途**：结合CNN噪声分类和FxNLMS算法，实现实时噪声控制

### 4. Disturbance\_generation.py

- **功能**：生成干扰信号
- **用途**：为训练和测试生成各种噪声信号

### 5. SFANC-FxNLMS for ANC.ipynb

- **功能**：Jupyter Notebook示例
- **用途**：展示如何使用混合SFANC-FxNLMS算法进行主动噪声控制

## 安装和环境配置

### 1. 安装依赖

```bash
pip install torch numpy scipy sounddevice progressbar
```

### 2. 配置环境

- Python 3.9.7或更高版本
- PyTorch 1.10.1或更高版本
- CUDA 11.3或更高版本（用于GPU加速）

## 使用方式

### 1. 使用预训练模型进行噪声控制

1. **运行Jupyter Notebook示例**：
   ```bash
   jupyter notebook "SFANC-FxNLMS for ANC.ipynb"
   ```
2. **运行实时噪声控制**：
   ```bash
   python realtime_sfanc_fxnlms.py
   ```

### 2. 训练自己的模型

1. **准备数据集**：
   - 生成或下载噪声数据集
   - 按照80,000:2,000:2,000的比例划分训练、验证和测试集
2. **训练模型**：
   - 使用`Network.py`中定义的模型
   - 训练完成后保存模型到`Trained models/model.pth`
3. **训练控制滤波器**：
   - 使用`FxNLMS_algorithm.py`中的`train_fxnlms_algorithm`函数
   - 保存训练好的控制滤波器到`Trained models/Pretrained_Control_filters.mat`

## 常见问题和解决方案

### 1. 缺少依赖库

**问题**：运行时出现`ModuleNotFoundError`

**解决方案**：安装缺少的依赖库

```bash
pip install <缺失的库>
```

### 2. MAT文件变量名错误

**问题**：运行时出现`KeyError`，提示找不到某个变量

**解决方案**：检查MAT文件中的变量名，确保代码中使用的变量名与MAT文件中的一致

### 3. 音频设备错误

**问题**：运行`realtime_sfanc_fxnlms.py`时出现音频设备错误

**解决方案**：

- 检查音频设备是否正常工作
- 在`realtime_sfanc_fxnlms.py`中修改`mic_device_name`和`spk_device_name`为您的实际设备名称

### 4. 实时性能问题

**问题**：实时噪声控制时出现卡顿

**解决方案**：

- 降低采样率或帧大小
- 使用更高效的硬件
- 优化代码性能

## 运行说明：

为了训练1D CNN模型，我们随机生成了80,000个具有各种频段、幅度和背景噪声水平的宽带噪声轨道。每个轨道持续1秒。合成噪声数据集被分为三个子集：80,000个噪声轨道用于训练，2,000个噪声轨道用于验证，2,000个噪声轨道用于测试。整个数据集可在 <https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/ETJWLU> 获取。

如果您不想训练模型，可以直接使用存储在"Trained models/model.pth"中的预训练1D模型。

基于所提出的混合SFANC-FxNLMS算法对真实记录噪声进行主动噪声控制。您可以轻松运行"SFANC-FxNLMS for ANC.ipynb"
真实噪声在"Real Noise Examples/"中提供。

引用：
如果您发现混合SFANC-FxNLMS算法在您的研究中有用，请考虑引用：
@ARTICLE{9761749,
author={Luo, Zhengding and Shi, Dongyuan and Gan, Woon-Seng},
journal={IEEE Signal Processing Letters},
title={A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control Based on Deep Learning},
year={2022},
volume={29},
pages={1102-1106},
doi={10.1109/LSP.2022.3169428}}

## 相关出版物

1. **基于无延迟CNN的选择性固定滤波器主动噪声控制的实时实现和可解释AI分析**\
   *期刊*: Mechanical Systems and Signal Processing, 2024, 214: 111364.\
   *论文链接*: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327024002620)   *代码链接*: [GitHub](https://github.com/Luo-Zhengding/SFANC-Window)
2. **基于多任务学习的频率方向感知多通道选择性固定滤波器主动噪声控制**\
   *期刊*: IEEE Transactions on Audio, Speech and Language Processing, 2025, 33: 3137-3147.
   *论文链接*: [IEEE](https://ieeexplore.ieee.org/document/11082568)   *代码链接*: [GitHub](https://github.com/Luo-Zhengding/Frequency-Direction-MCSFANC)
3. **基于不同卷积神经网络的选择性固定滤波器主动噪声控制性能评估**\
   *会议*: The 51st International Congress and Exposition on Noise Control Engineering (Inter-Noise 2022)
   *论文链接*: [arXiv](https://arxiv.org/pdf/2208.08440)

**如果您对我们的工作感兴趣，请考虑引用我们的论文。谢谢！祝您有愉快的一天！**
