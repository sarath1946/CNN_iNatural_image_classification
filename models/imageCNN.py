from .conv_block import ConvBlock

import torch
import torch.nn as nn
# import torch.nn.Module
# import torch.nn.Functional


# kernel_size=3,
#                  filter_strategy="constant",
#                  activation="relu",
#                  dense_neurons=1024):

class imageCNN(nn.Module):
    def __init__(self,input_channels,num_classes,num_blocks=5,base_filters=32,
    kernal_size=3,filter_strategy="halving",activation="relu",dense_neurons=1024):

        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        out_ch = base_filters

        
        for i in range(num_blocks):
            self.blocks.append(ConvBlock
            (in_ch,out_ch,kernal_size,
            use_bn=True,activation=activation,
            pool_kernal=2,dropout=0.2))

            in_ch=out_ch
            # if filter_strategy == "doubling":
            #     out_ch *= 2
            # elif filter_strategy == "halving":
            #     out_ch = max(out_ch // 2, 1)

        self.flatten = nn.Flatten()
        self.feature_dim = self._get_conv_output(224)
        self.fc1 = nn.Linear(self.feature_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def _get_conv_output(self, image_size):

        x = torch.zeros(1, 3, image_size, image_size)

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)

        return x.shape[1]
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            print(x.shape)
        print("conv block output")
        print(x.shape)
        x = self.flatten(x)

    # global average pooling alternative (safer)
    # x = x.mean(dim=(2, 3))

        x = self.fc1(x)
        x = self.fc2(x)
        return x

