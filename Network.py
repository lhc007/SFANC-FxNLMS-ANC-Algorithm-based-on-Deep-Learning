import torch

#------------------------------------------------------------------------------
# 类: CNN - 基础卷积神经网络模型
# 描述: 用于噪声分类的一维卷积神经网络
#------------------------------------------------------------------------------
class CNN(torch.nn.Module):
    
    def __init__(self, channels, conv_kernels, conv_strides, conv_padding, pool_padding, num_classes=15):
        """
        初始化CNN模型
        
        参数:
        - channels: 各层卷积通道数
        - conv_kernels: 卷积核大小
        - conv_strides: 卷积步长
        - conv_padding: 卷积填充
        - pool_padding: 池化填充
        - num_classes: 分类类别数
        """
        assert len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding)
        super(CNN, self).__init__()
        
        # 创建卷积块
        self.conv_blocks = torch.nn.ModuleList()
        prev_channel = 1
        
        for i in range(len(channels)):
            # 添加堆叠卷积层
            block = []
            for j, conv_channel in enumerate(channels[i]):
                block.append(torch.nn.Conv1d(in_channels=prev_channel, out_channels=conv_channel, kernel_size=conv_kernels[i], stride=conv_strides[i], padding=conv_padding[i]))
                prev_channel = conv_channel
                # 添加批归一化层
                block.append(torch.nn.BatchNorm1d(prev_channel))
                # 添加ReLU激活函数
                block.append(torch.nn.ReLU())
            self.conv_blocks.append(torch.nn.Sequential(*block))

        # 创建池化块
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(len(pool_padding)):
            # 添加最大池化（维度减少4倍）
            self.pool_blocks.append(torch.nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[i]))

        # 全局池化
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(prev_channel, num_classes)


    def forward(self, inwav):
        """
        前向传播
        
        参数:
        - inwav: 输入音频数据
        
        返回:
        - out: 分类结果
        """
        for i in range(len(self.conv_blocks)):
            # 应用卷积层
            inwav = self.conv_blocks[i](inwav)
            # 应用池化层
            if i < len(self.pool_blocks): inwav = self.pool_blocks[i](inwav)
        # 应用全局池化
        out = self.global_pool(inwav).squeeze() # [batch_size, 256, 1] -> [batch_size, 256]
        out = self.linear(out) #[batch_size, 15]
        return out
    
#------------------------------------------------------------------------------
# 类: ResBlock - 残差块
# 描述: 用于CNNRes模型中的残差连接块
#------------------------------------------------------------------------------
class ResBlock(torch.nn.Module):
    
    def __init__(self, prev_channel, channel, conv_kernel, conv_stride, conv_pad):
        """
        初始化残差块
        
        参数:
        - prev_channel: 输入通道数
        - channel: 输出通道数
        - conv_kernel: 卷积核大小
        - conv_stride: 卷积步长
        - conv_pad: 卷积填充
        """
        super(ResBlock, self).__init__()
        self.res = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels = prev_channel, out_channels=channel, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad),
            torch.nn.BatchNorm1d(channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad),
            torch.nn.BatchNorm1d(channel),
        )
        self.bn = torch.nn.BatchNorm1d(channel)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入数据
        
        返回:
        - x: 输出数据
        """
        identity = x
        x = self.res(x)
        if x.shape[1] == identity.shape[1]:
            x += identity
        # 重复较小的块直到达到较大块的大小
        elif x.shape[1] > identity.shape[1]:
            if x.shape[1] % identity.shape[1] == 0:
                x += identity.repeat(1, x.shape[1]//identity.shape[1], 1)
            else:
                raise RuntimeError("ResBlock中的维度需要能被前一个维度整除！")
        else:
            if identity.shape[1] % x.shape[1] == 0:
                identity += x.repeat(1, identity.shape[1]//x.shape[1], 1)
            else:
                raise RuntimeError("ResBlock中的维度需要能被前一个维度整除！")
            x = identity
        x = self.bn(x)
        x = self.relu(x)
        return x
    
#------------------------------------------------------------------------------
# 类: CNNRes - 带残差连接的卷积神经网络
# 描述: 用于噪声分类的带残差连接的一维卷积神经网络
#------------------------------------------------------------------------------
class CNNRes(torch.nn.Module):       
        
    def __init__(self, channels, conv_kernels, conv_strides, conv_padding, pool_padding, num_classes=15):
        """
        初始化CNNRes模型
        
        参数:
        - channels: 各层卷积通道数
        - conv_kernels: 卷积核大小
        - conv_strides: 卷积步长
        - conv_padding: 卷积填充
        - pool_padding: 池化填充
        - num_classes: 分类类别数
        """
        assert len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding)
        super(CNNRes, self).__init__()
        
        # 创建卷积块
        prev_channel = 1
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=prev_channel, out_channels=channels[0][0], kernel_size=conv_kernels[0], stride=conv_strides[0], padding=conv_padding[0]),
            # 添加批归一化层
            torch.nn.BatchNorm1d(channels[0][0]),
            # 添加ReLU激活函数
            torch.nn.ReLU(),
            # 添加最大池化
            torch.nn.MaxPool1d(kernel_size = 4, stride = 4, padding = pool_padding[0]),
        )
        
        # 创建残差块
        prev_channel = channels[0][0]
        self.res_blocks = torch.nn.ModuleList()
        for i in range(1, len(channels)):
            # 添加堆叠残差层
            block = []
            for j, conv_channel in enumerate(channels[i]):
                block.append(ResBlock(prev_channel, conv_channel, conv_kernels[i], conv_strides[i], conv_padding[i]))
                prev_channel = conv_channel
            self.res_blocks.append(torch.nn.Sequential(*block))

        # 创建池化块
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(1, len(pool_padding)):
            # 添加最大池化（维度减少4倍）
            self.pool_blocks.append( torch.nn.MaxPool1d(kernel_size = 4, stride = 4, padding = pool_padding[i]) )

        # 全局池化
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(prev_channel, num_classes)


    def forward(self, inwav):
        """
        前向传播
        
        参数:
        - inwav: 输入音频数据
        
        返回:
        - out: 分类结果
        """
        inwav = self.conv_block(inwav)
        for i in range(len(self.res_blocks)):
            # 应用残差层
            inwav = self.res_blocks[i](inwav)
            # 应用池化层
            if i < len(self.pool_blocks): inwav = self.pool_blocks[i](inwav)
        # 应用全局池化
        out = self.global_pool(inwav).squeeze()
        out = self.linear(out)
        return out


#------------------------------------------------------------------------------
# 模型实例化
# 描述: 创建一个CNNRes模型实例，用于噪声分类
# 参数:
# - channels: 各层卷积通道数，[[128], [128, 128]]
# - conv_kernels: 卷积核大小，[80, 3]
# - conv_strides: 卷积步长，[4, 1]
# - conv_padding: 卷积填充，[38, 1]
# - pool_padding: 池化填充，[0, 0]
#------------------------------------------------------------------------------
m6_res = CNNRes(channels = [[128], [128]*2],
         conv_kernels = [80, 3],
         conv_strides = [4, 1],
         conv_padding = [38, 1],
         pool_padding = [0, 0])
