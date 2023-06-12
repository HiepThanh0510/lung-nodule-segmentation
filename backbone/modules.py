import torch.nn as nn
import torch
import torch.nn.functional as F 
# Input: b, c, stride, padding
# Output: c, c', string, padding
# ResidualConv(input_dim=c, output_dim=c')
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)
        # return self.conv_block(x) + x
# Input: b, c, h, w
# Output: b, c, h, w
# Squeeze_Excite_Block(c)
class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Input: b, in_dims(c), height, width 
# Output: b, out_dims(c'), height, width 
# ASPP(in_dims, out_dims)
class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# Input: b, c, h, w
# Output: b, c, h * 2, w * 2
class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


'''
Explain interpolate parameters align_corners: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/13
'''
class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1) # Nếu out_channels chỗ này vẫn giữa là như in_channels thì vẫn đúng mà ta ? 
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1) # Có cơ hội hãy trả lời câu này. 
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1) # Vi du input cua softmax la tensor co shape la (a, b, c) -> có a ma trận kích thước (bxc)
                                          # dim = 0 -> theo logic là vị trí theo a -> lấy theo chiều của kênh                                         
                                          # dim = 1 -> theo logic là vị trí theo b -> mà trong spatial b là số hàng -> tức là lấy dọc theo CỘT của ma trận bxc 
                                          # dim = -1 -> theo logic là lấy vị trí cuối cùng -> c -> mà trong spatial c là số cột -> tức là lấy dọc theo HÀNG của ma trận bxc 
        self.down_sample = nn.MaxPool2d((8, 8))
        self.up_sample = nn.UpsamplingBilinear2d((512, 512))
        # self.up_sample = nn.UpsamplingBilinear2d((256, 256))
    def forward(self, x):
        # print(f"In PositionAttention_forward(), height = {x.size()[2]}, width = {x.size()[3]}")
        _, _, height_up, width_up = x.size()
        x = self.down_sample(x)
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1) # purmute kiểu này nếu print shape ra sẽ hiểu cái này đang transpose để có thể nhân với C 
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c)) # 1, 128 * 128, 128 * 128 
        feat_d = self.conv_d(x).view(batch_size, -1, height * width) # 1, 4, 512 * 512 
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        # x = self.up_sample(x)
        x = F.interpolate(x, size=(height_up, width_up), mode='bilinear', align_corners=True)
        # feat_e = self.up_sample(feat_e) 
        feat_e = F.interpolate(feat_e, size=(height_up, width_up), mode='bilinear', align_corners=True)
        out = self.alpha * feat_e + x
        # print(f"position shape = {out.shape}")
        return out
class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new) 

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out    

class DualAttention(nn.Module):
    def __init__(self, channel):
        super(DualAttention, self).__init__()
        self.position = PositionAttentionModule(channel)
        self.channel = ChannelAttentionModule()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
    def forward(self, x):
        p = self.position(x)
        c = self.channel(x)
        final_feat = self.conv(p+c)
        return final_feat 
    

        
        
        
        
        