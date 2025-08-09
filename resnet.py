#encoding=utf-8
#◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Author: Xiaoyuan Yao
# GitHub: https://github.com/yaoxiaoyuan/mystRL/
# Contact: yaoxiaoyuan1990@gmail.com
# Created: Sat Jun 14 15:08:00 2025
# License: MIT
# Version: 0.1.0
#
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, n_filters):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            3,
            n_filters, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))    


class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super(ResidualBlock, self).__init__()   
        self.conv_1 = nn.Conv2d(
            n_filters,
            n_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn_1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(
            n_filters,
            n_filters,
            kernel_size=3,                                                                                
            stride=1,
            padding=1,
        )
        self.bn_2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        output = self.relu(self.bn_1(self.conv_1(x)))
        output = self.bn_2(self.conv_2(output))
        return self.relu(output + x)
 
class PolicyHead(nn.Module):
    def __init__(self, 
                 input_shape,
                 n_class):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(
            input_shape[0],
            2,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2*input_shape[1]*input_shape[2], n_class)

    def forward(self, x): 
        output = self.relu(self.bn(self.conv(x)))
        logits = self.linear(self.flatten(output))
        return logits  

class ValueHead(nn.Module):
    def __init__(self,                                                                                    
                 input_shape,
                 hidden_size):                                                                               
        super(ValueHead, self).__init__()                                                                
        self.conv = nn.Conv2d(                                                                            
            input_shape[0],
            1,
            kernel_size=1,                                                                                
            stride=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(input_shape[1]*input_shape[2], hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.relu(self.bn(self.conv(x)))
        output = self.relu(self.linear_1(self.flatten(output)))
        output = self.tanh(self.linear_2(output))
        return output.squeeze(-1)

class ResNet(nn.Module):
    def __init__(self, 
                 input_shape, 
                 n_filters,
                 n_res_blocks, 
                 hidden_size,
                 n_class):
        super(ResNet, self).__init__()

        input_channels = input_shape[0]

        self.conv_block = ConvBlock(input_channels, n_filters)
 
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(n_filters) for _ in range(n_res_blocks)]
        )       
        
        res_output_shape = [n_filters, input_shape[1], input_shape[2]]
        self.policy_head = PolicyHead(res_output_shape, n_class) 
        self.value_head = ValueHead(res_output_shape, hidden_size)


    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value


