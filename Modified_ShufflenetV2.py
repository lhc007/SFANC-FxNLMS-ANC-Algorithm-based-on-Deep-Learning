import torchvision.models as models
import torch.nn as nn


class Modified_ShufflenetV2(nn.Module):
    """
    修改版的ShuffleNetV2网络，用于噪声分类
    将单通道的频谱图转换为3通道，然后使用预训练的ShuffleNetV2进行特征提取和分类
    """

    def __init__(self, num_classes):
        """
        初始化修改版ShuffleNetV2网络
        
        参数:
            num_classes: 分类类别数量（即控制滤波器的数量）
        """
        super().__init__()

        # 单通道到3通道的转换层（将频谱图的单通道转换为3通道，适配预训练模型）
        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),  # 对单通道输入进行批归一化
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),  # 1x1卷积，1通道→10通道
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())  # 1x1卷积，10通道→3通道

        # 加载在ImageNet上预训练的ShuffleNetV2 x0.5模型
        self.mv2 = models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
        # 修改最后一个卷积层，将输出通道从1024改为512
        self.mv2.conv5 = nn.Sequential(
            nn.Conv2d(192, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),)

        # 修改全连接层，将输出维度改为指定的类别数
        self.mv2.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入数据（单通道频谱图）
            
        返回:
            分类预测结果
        """
        x = self.bw2col(x)  # 先将单通道转换为3通道
        x = self.mv2(x)      # 通过ShuffleNetV2进行特征提取和分类
        return x