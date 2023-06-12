import torch.nn as nn
import torch
from backbone.modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
    PositionAttentionModule,
    ChannelAttentionModule,
    DualAttention
)

# from modules import (
#     ResidualConv,
#     ASPP,
#     AttentionBlock,
#     Upsample_,
#     Squeeze_Excite_Block,
#     PositionAttentionModule,
#     ChannelAttentionModule,
#     DualAttention
# )

class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()

        # First block 
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        # Giải thích câu hỏi tại sao không cho trực tiếp input để skip 
        # Vì output sau khi cho qua "input_layer" có shape đã khác input đầu vào 
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        
        # self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.dual_attention1 = DualAttention(filters[0])
        
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        # self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.dual_attention2 = DualAttention(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.dual_attention3 = DualAttention(filters[2])
    
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        # x2 = self.squeeze_excite1(x1)
 
        x2 = self.dual_attention1(x1)
        # print(f"with dual_attention block, x2.shape = {x2.shape}")
        x2 = self.residual_conv1(x2)
        # print(f"self.residual_conv1(x2) = {x2.size()}")
        # x3 = self.squeeze_excite2(x2)
        # print(f"self.squeeze_excite(x2) = {x3.size()}")
        x3 = self.dual_attention2(x2)
        x3 = self.residual_conv2(x3)

        # x4 = self.squeeze_excite3(x3)
        x4 = self.dual_attention3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out
    
