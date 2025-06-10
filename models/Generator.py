import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import warnings
from pytorch_wavelets import DWTForward
import sys, os

sys.path.insert(0, os.getcwd())
from models.loss import SSIM,VGGLoss
# TODO? 下采样模块
from Bottleneck import AKConv, MFFA, WTConv
from Bottleneck import PConv
# TODO? 频域模块
from Neck import ffc, FreBlock

warnings.filterwarnings('ignore')


class CBA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel = 3, stride = 1):
        super(CBA, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel, stride = stride,
                      padding = kernel // 2, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return (self.layer(x))


class res_basic(nn.Module):
    def __init__(self, inplanes, outplanes, ksize = 1, stride = 1):
        super(res_basic, self).__init__()

        self.conv1 = CBA(in_planes = inplanes, out_planes = outplanes,
                         kernel = ksize, stride = stride)

        self.relu = nn.ReLU(inplace = True)

        self.bn = nn.BatchNorm2d(outplanes)

        self.conv2 = nn.Conv2d(in_channels = outplanes,
                               out_channels = outplanes,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)

        self.bn1 = nn.BatchNorm2d(outplanes)

        self.conv3 = nn.Conv2d(in_channels = inplanes,
                               out_channels = outplanes,
                               kernel_size = 1,
                               stride = stride,
                               padding = 0)

        self.bn2 = nn.BatchNorm2d(outplanes)

        self.out = nn.Conv2d(in_channels = 2 * outplanes,
                             out_channels = outplanes,
                             kernel_size = 1,
                             stride = 1,
                             padding = 0)

    def forward(self, x):
        x1 = x  #
        x = self.conv1(x)
        x = self.bn1(self.conv2(x))
        x1 = self.relu(self.bn2(self.conv3(x1)))

        out = torch.cat((x1, x), dim = 1)

        return self.out(out)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan1, in_chan2, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = res_basic(in_chan1 + in_chan2, out_chan, 3, stride = 1)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([ fsp, fcp ], dim = 1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[ 2: ])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)

        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction = 4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels // reduction, channels, kernel_size = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale + x


def histogram_loss(pred, target, bins = 256, range = (0., 1.)):
    # 计算直方图

    hist_pred = torch.histc(pred, bins = bins, min = range[ 0 ], \
                            max = range[ 1 ])
    hist_target = torch.histc(target, bins = bins, min = range[ 0 ], \
                              max = range[ 1 ])

    total_pixel = pred.numel()
    pred_cdf = (hist_pred) / (total_pixel)

    target_cdf = (hist_target) / (total_pixel)
    pred_cdf = (hist_pred) / (total_pixel)

    # pred_cdf = hist_pred.cumsum(dim = 0)
    # target_cdf = hist_target.cumsum(dim = 0)

    # KL散度衡量分布差异
    # pred_cdf = (pred_cdf - pred_cdf.min()) / (pred_cdf.max() - pred_cdf.min())
    # target_cdf = (target_cdf - target_cdf.min()) / (target_cdf.max() - target_cdf.min())

    loss = torch.sum(target_cdf * torch.log(target_cdf / (pred_cdf + 1e-6)))
    return loss # 权重设为0.1


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义Sobel算子（水平和垂直方向）
        kernel_x = torch.tensor([ [ [ [ -1, 0, 1 ], [ -2, 0, 2 ], [ -1, 0, 1 ] ] ] ], dtype = torch.float32,
                                device = torch.device('cuda'))
        kernel_y = torch.tensor([ [ [ [ -1, -2, -1 ], [ 0, 0, 0 ], [ 1, 2, 1 ] ] ] ], dtype = torch.float32,
                                device = torch.device('cuda'))

        self.register_parameter('kernel_x', torch.nn.Parameter(kernel_x, requires_grad = True))
        self.register_parameter('kernel_y', torch.nn.Parameter(kernel_y, requires_grad = True))

    def forward(self, pred, target):
        # 确保输入为4D张量（Batch, Channel, Height, Width）
        assert pred.dim() == 4 and target.dim() == 4, "Input must be 4D (B, C, H, W)"

        # 对每个通道计算梯度
        grad_x_pred = self._conv2d(pred, self.kernel_x)
        grad_y_pred = self._conv2d(pred, self.kernel_y)
        grad_x_target = self._conv2d(target, self.kernel_x)
        grad_y_target = self._conv2d(target, self.kernel_y)

        # 计算L1损失（平均绝对值差）
        loss_x = torch.mean(torch.abs(grad_x_pred - grad_x_target))
        loss_y = torch.mean(torch.abs(grad_y_pred - grad_y_target))

        return loss_x + loss_y

    def _conv2d(self, x, kernel):
        # 处理多通道输入：每个通道独立卷积后求和
        if x.size(1) > 1:
            kernel = kernel.repeat(x.size(1), 1, 1, 1)  # 扩展至与输入通道数匹配
            padding = (kernel.size(2) // 2, kernel.size(3) // 2)  # 保持输出尺寸不变
            return torch.nn.functional.conv2d(x, kernel, padding = padding, groups = x.size(1))
        else:
            return torch.nn.functional.conv2d(x, kernel, padding = 1)


# TODO? 判别器
class PatchDiscriminator(nn.Module):
    def __init__(self, in_dim = 3, ndf = 64, n_layers = 3):
        super(PatchDiscriminator, self).__init__()
        self.train_iters = 0

        # TODO? downsample
        layers = [ nn.Conv2d(in_dim, ndf, kernel_size = 4, padding = 1, stride = 2),
                   nn.LeakyReLU(0.2, inplace = True) ]

        in_channels = ndf
        out_channels = ndf * 2
        for i in range(1, n_layers + 1):
            stride = 2 if i < n_layers else 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 4, padding = 1, stride = stride))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            layers.append(ResnetBlock22(out_channels, padding_type = 'reflect'))

            in_channels = out_channels
            out_channels = out_channels * (2 if i < 3 else 1)

        layers.append(nn.Conv2d(out_channels, 1, kernel_size = 4, padding = 1, stride = 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


from kornia.color import rgb_to_lab


def lab_loss(pred = None,
             gt = None):
    device = pred.device
    pred = rgb_to_lab(pred)
    target = rgb_to_lab(gt)

    # TODO? convert tensor
    l_pred = torch.tensor(pred[ ..., 0 ]) / 100.
    l_target = torch.tensor(target[ ..., 0 ]) / 100.
    pred = torch.tensor(pred[ ..., 1:3 ]).to(device) / 127.
    target = torch.tensor(target[ ..., 1:3 ]).to(device) / 127.

    # l1 loss
    loss = torch.mean(torch.abs(pred - target)) + 1.5 * torch.mean(torch.abs(l_pred - l_target))

    # TODO? hsv

    # pre_hsv = rgb_to_hsv(pred)
    # target_hsv = rgb_to_hsv(gt)

    # loss_hsv = 1.0 - torch.cos((pre_hsv[ ..., 0 ] - target_hsv[ ..., 0 ]) * 2 * np.pi).mean()

    return loss  # + loss_hsv


def cross_attention(q = None,
                    k = None,
                    v = None,
                    h = None,
                    w = None):
    b, d, c = q.shape

    # -----------------------------------------------#
    #   TODO? 单一维度容易显存占用过大
    # -----------------------------------------------#
    chunk = d // 8  # TODO? 拆分数量
    assert chunk != 0, "operator error!!!"

    feats = [ ]
    for ind in range(0, d, chunk):
        sta = ind
        end = sta + chunk

        attention = F.softmax((q[ :, sta:end, : ] @ k[ :, sta:end, : ].permute((0, 2, 1))) * (8 ** (-0.5)), dim = 1)

        attention = attention @ v[ :, sta:end, : ]
        feats.append(attention)

    # h = int(d**0.5)

    attention = torch.cat(feats, dim = 1)

    # attention = attention.contiguous().view((b,h,w,c)).permute((0,3,1,2))

    return attention


class FFN(nn.Module):
    def __init__(self,
                 in_planes = None,
                 out_planes = None):
        super(FFN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_planes),
            nn.Linear(in_features = in_planes,
                      out_features = out_planes),
            nn.GELU(),

        )

        self.div = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.GELU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(out_planes, out_planes),
            nn.GELU()
        )

        self.layer3 = nn.Sequential(
            nn.LayerNorm(in_planes),
            nn.Linear(in_features = in_planes,
                      out_features = out_planes),
            nn.GELU(),
            nn.Linear(out_planes, out_planes),

        )

        self.norm = nn.LayerNorm(out_planes)
        self.act = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.Sigmoid())
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = (self.div(x) - x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x)

        gate = self.act(x)

        return gate * self.drop(self.norm(x2 + x3))


class Down(nn.Module):
    def __init__(self,
                 in_planes = None,
                 out_planes = None):
        super(Down, self).__init__()

        # +++++++++++++++++++++++++++++++++++++++++++#
        #      TODO? 分解频率域
        # +++++++++++++++++++++++++++++++++++++++++++#
        self.dwt = DWTForward(J = 1, wave = 'haar')

        # TODO?
        self.conv1 = nn.Conv2d(in_channels = 2 * in_planes,
                               out_channels = out_planes // 2, kernel_size = 1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = in_planes,
                      out_channels = out_planes // 2,
                      kernel_size = 4, padding = 1, stride = 2),

            nn.ReLU(inplace = True)
        )

        self.ffn = FFN(out_planes // 2, out_planes // 2)

    def forward(self, x):
        dw = self.layer1(x)

        yL, yH = self.dwt(x)

        y_HL = yH[ 0 ][ :, :, 0, :: ]
        y_LH = yH[ 0 ][ :, :, 1, :: ]

        y_HH = yH[ 0 ][ :, :, 2, :: ]
        #  print(y_HL.shape,y_HL.shape)

        y = self.conv1(torch.cat((y_HL, y_LH), dim = 1))

        # ------------------------------------------#
        #   TODO? 转为
        # ------------------------------------------#
        b, c, h, w = y.shape

        y = y.permute((0, 2, 3, 1)).contiguous().view((b, -1, c))
        yL = yL.permute((0, 2, 3, 1)).contiguous().view((b, -1, c))
        y_HH = y_HH.permute((0, 2, 3, 1)).contiguous().view((b, -1, c))
        # dw = dw.permute((0,2,3,1)).contiguous().view((b,-1,c))
        # print(yL.shape,y.shape,y_HH.shape)

        # TODO? 1
        attention = self.ffn(cross_attention(yL, y, y_HH, h, w))
        attention = attention.contiguous().view((b, h, w, c)).permute((0, 3, 1, 2))
        y = y.contiguous().view((b, h, w, c)).permute((0, 3, 1, 2))
        # TODO? 2
        # attention = cross_attention(dw,y,yL,h,w)

        # TODO? 3
        # attention = cross_attention(dw,y,yL,h,w)
        # attention = self.ffn(attention)
        # attention = attention.contiguous().view((b,h,w,c)).permute((0,3,1,2))

        # TODO? 4
        # attention = cross_attention(dw,y,y,h,w)
        # attention = self.ffn(attention)
        # attention = attention.contiguous().view((b,h,w,c)).permute((0,3,1,2))

        # TODO? 5
        # attention = cross_attention(dw,y,dw,h,w)
        # attention = self.ffn(attention)
        # attention = attention.contiguous().view((b,h,w,c)).permute((0,3,1,2))

        down_2x = F.interpolate(x, size = attention.shape[ 2: ], mode = 'bilinear',
                                align_corners = True)

        attention = attention * (down_2x - dw)

        return attention + y


class DilateConv(nn.Module):
    def __init__(self, inc, filter = [ 3, 5, 7, 9 ]):
        super(DilateConv, self).__init__()

        self.filter = filter

        self.conv = nn.ModuleList()

        self.conv.extend([ nn.Sequential(nn.Conv2d(in_channels = inc,
                                                   out_channels = inc,
                                                   kernel_size = 3,
                                                   padding = (3 + (ind - 1) * 2) // 2,
                                                   dilation = ind),
                                         nn.BatchNorm2d(inc),
                                         nn.ReLU(inplace = True)) for ind in filter ])
        self.fin = nn.Conv2d(in_channels = inc * 4, out_channels = inc,
                             kernel_size = 3, padding = 1)

        self.out = nn.Conv2d(in_channels = inc, out_channels = inc,
                             kernel_size = 1)

    def forward(self, x):
        pred = x

        feats = [ ]
        for layer in self.conv:
            x = layer(x)
            feats.append(x)

        feats = torch.cat(feats, dim = 1)

        x = self.fin(feats)
        x1 = self.out(x * F.sigmoid(x)) + pred

        return x1


class MixFreFeature(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()

        # 可学习参数 (初始化接近中心区域)
        self.h_radius_raw = nn.Parameter(torch.tensor(0.1))  # 高频保留范围
        self.l_radius_raw = nn.Parameter(torch.tensor(0.3))  # 低频保留范围
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 高低频混合系数

        # 动态融合权重
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace = True)
        )

        self.fin = nn.Conv2d(2 * inplanes, outplanes, 1)

    def generate_gaussian_mask(self, x, radius, is_high_freq = True):
        """
        生成可微分高斯掩码
        Args:
            x: 输入特征图 (用于获取尺寸信息)
            radius: 控制高斯衰减的半径 (浮点数)
            is_high_freq: 是否为高频掩码
        """
        B, C, H, W = x.shape
        device = x.device

        # 生成距离矩阵 (归一化到[-1,1])
        y_coord = torch.linspace(-1, 1, H, device = device).view(1, H, 1)
        x_coord = torch.linspace(-1, 1, W, device = device).view(1, 1, W)
        dist = torch.sqrt(y_coord ** 2 + x_coord ** 2)  # [1, H, W]

        # 计算高斯衰减
        sigma = torch.clamp(radius / H, 0.1, 1.0)
        mask = torch.exp(-dist ** 2 / (2 * sigma ** 2 + 1e-6))

        # 高频保留周围，低频保留中心
        if is_high_freq:
            mask = 1.0 - mask  # 反转掩码

        return mask.unsqueeze(1)  # [1, 1, H, W]

    def get_radius(self, x, radius):
        return (torch.sigmoid(x) * radius)

    def forward(self, x):
        # 傅里叶变换
        f = fft.rfft2(x, norm = 'backward')

        # 高频掩码（保留周围细节）
        h_radius = self.get_radius(self.h_radius_raw, f.shape[ -1 ])
        mask_high = self.generate_gaussian_mask(f, h_radius, True)

        # 低频掩码（保留中心光照）
        l_radius = self.get_radius(self.l_radius_raw, f.shape[ -1 ])
        mask_low = self.generate_gaussian_mask(f, l_radius, False)

        # 频域分离
        f_high = f * mask_high

        f_low = f * mask_low

        # 逆变换回空间域
        x_high = fft.irfft2(f_high, s = x.shape[ -2: ], norm = 'backward')
        x_low = fft.irfft2(f_low, s = x.shape[ -2: ], norm = 'backward')

        mixed = self.alpha.sigmoid() * x_high + (1. - self.alpha.sigmoid()) * x_low
        mixed = self.conv1(mixed)

        out = torch.cat((mixed, x), dim = 1)

        return self.fin(out)


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc,
                 output_nc, ngf = 64,
                 n_downsampling = 3,
                 n_blocks = 9,
                 padding_type = 'reflect',
                 opt = None):  # TODO? 可调控参数
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        # ------------------------------------------#
        # TODO? 损失函数
        # ------------------------------------------#

        self.pixel_loss = opt[ 'pixel_loss' ]
        self.ssim_loss = opt[ 'ssim_loss' ]
        self.color_loss = opt[ 'color_loss' ]
        self.gradient_loss = opt[ 'gradient_loss' ]
        self.histogram_loss = opt[ 'hist_loss' ]
        self.vgg_loss = opt[ 'vgg_loss' ]

        # ------------------------------------------#
        # TODO? 权重系数
        # ------------------------------------------#
        self.pixel_weight = opt[ 'pixel' ]
        self.ssim_weight = opt[ 'ssim' ]
        self.color_weight = opt[ 'color' ]
        self.gradient_weight = opt[ 'gradient' ]
        self.hist_weight = opt[ 'hist' ]
        self.vgg_weight = opt[ 'vgg' ]

        # ------------------------------------------#
        # TODO? 模块对比消融
        # 1. 使用正常下采样替换小波变换下采样(其他模块正常保留)
        # 2.
        # ------------------------------------------#

        model1 = [ nn.Conv2d(input_nc, ngf, kernel_size = 3, stride = 1, padding = 1), activation ]
        ### downsample
        mult = 2 ** 0

        down_type = opt[ 'down_type' ]

        if down_type == 'normal':
            model2 = [ nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size = 4, stride = 2, padding = 1), activation ]
        elif down_type == 'akconv':
            model2 = [ AKConv.LDConv(inc = ngf * mult, outc = ngf * mult * 2, num_param = 4, stride = 2), activation ]
        elif down_type == 'mffa':
            # TODO? 降采样
            model2 = [ MFFA.DWT_2D(inc = ngf * mult, outc = ngf * mult * 2), activation ]
        elif down_type == 'pconv':
            model2 = [ PConv.PConv(ngf * mult, ngf * mult * 2, 4, 2), activation ]
        elif down_type == 'wtconv':
            model2 = [ WTConv.WTConv2d(in_channels = ngf * mult,
                                       out_channels = ngf * mult * 2,stride = 2), activation ]
        elif down_type == 'ours':
            model2 = [ Down(in_planes = ngf * mult, out_planes = ngf * mult * 2),
                       nn.Conv2d(in_channels = ngf * mult, out_channels = ngf * mult * 2, kernel_size = 1),
                       activation ]
        else:
            raise NotImplementedError('down type error!!!')
        mult = 2 ** 1

        if down_type == 'normal':
            model3 = [ nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size = 4, stride = 2, padding = 1), activation ]

        elif down_type == 'akconv':
            model3 = [ AKConv.LDConv(inc = ngf * mult, outc = ngf * mult * 2, num_param = 4, stride = 2),
                       nn.Conv2d(in_channels = ngf * mult * 2, out_channels = ngf * mult * 2, kernel_size = 1),
                       activation ]
        elif down_type == 'mffa':
            # TODO? 降采样
            model3 = [ MFFA.DWT_2D(inc = ngf * mult, outc = ngf * mult * 2),
                       nn.Conv2d(in_channels = ngf * mult * 2, out_channels = ngf * mult * 2, kernel_size = 1), activation ]
        elif down_type == 'pconv':
            model3 = [ PConv.PConv(ngf * mult, ngf * mult * 2, 4, 2),
                       nn.Conv2d(in_channels = ngf * mult * 2, out_channels = ngf * mult * 2, kernel_size = 1), activation ]
        elif down_type == 'wtconv':
            model3 = [ WTConv.WTConv2d(in_channels = ngf * mult,
                                       out_channels = ngf * mult * 2,stride = 2), activation ]

        elif down_type == 'ours':
            model3 = [ Down(ngf * mult, ngf * mult * 2),
                       nn.Conv2d(in_channels = ngf * mult, out_channels = ngf * mult * 2, kernel_size = 1),
                       activation ]
        else:
            raise NotImplementedError('down type error!!!')

        self.is_use_dalte = opt[ 'is_use_dalte' ]  # TODO false

        if self.is_use_dalte:
            self.dalte1 = DilateConv(ngf * mult * 2)

        ### resnet blocks
        mult = 2 ** (n_downsampling - 1)
        model4 = [ ]
        for i in range(n_blocks):
            model4 += [ ResnetBlock22(ngf * mult, padding_type = padding_type, activation = activation),
                        DilateConv(ngf * mult) if self.is_use_dalte else nn.Identity() ]

        # TODO? feature fuse
        # self.fixuse1 = FeatureFusionModule(ngf * mult, ngf * mult, ngf * mult * 2)
        self.is_use_freq = opt[ 'is_use_freq' ]  # TODO? 0,1
        self.fre_type = opt[ 'fre_type' ]  # TODO str

        if self.is_use_freq:
            if self.fre_type == 'dm':
                self.model_mix4 = ffc.SpectralTransform(ngf * mult * 2, ngf * mult * 2)
            elif self.fre_type == 'four':
                self.model_mix4 = FreBlock.FreBlock(ngf * mult * 2)
            elif self.fre_type == 'ours':
                self.model_mix4 = MixFreFeature(ngf * mult * 2, ngf * mult * 2)
            else:
                raise NotImplementedError('frequency type error!!!')

        if self.is_use_dalte:
            self.dalte2 = DilateConv(ngf * mult * 2)

        mult = 2 ** (n_downsampling - 1)
        model5 = [ nn.Conv2d(ngf * mult * 2, int(ngf * mult / 2) * 4, kernel_size = 3, stride = 1, padding = 1) ]

        # TODO? 包括
        if self.is_use_freq:
            if self.fre_type == 'ours':
                model5.extend([ MixFreFeature(int(ngf * mult / 2) * 4, int(ngf * mult / 2)),
                                nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2) * 4, kernel_size = 1, stride = 1,
                                          padding = 0),
                                DilateConv(int(ngf * mult / 2) * 4) ])

            elif self.fre_type == 'dm':
                model5.extend([ ffc.SpectralTransform(ngf * mult * 2, ngf * mult * 2) ])
            elif self.fre_type == 'four':

                model5.extend([ FreBlock.FreBlock(ngf * mult * 2) ])

            else:
                raise NotImplementedError('frequency type error!!!')

        self.model5_act = activation
        mult = 2 ** (n_downsampling - 2)
        # self.fixuse2 = FeatureFusionModule(ngf * mult, ngf * mult, ngf * mult * 2)
        model6 = [ nn.Conv2d(ngf * mult * 2, int(ngf * mult / 2) * 4, kernel_size = 3, stride = 1, padding = 1) ]
        self.model6_act = activation

        model7 = [ ]
        model7 += [ nn.Conv2d(ngf * 2, ngf, kernel_size = 3, stride = 1, padding = 1) ]
        model7 += [ activation, nn.Conv2d(ngf, output_nc, kernel_size = 1, padding = 0) ]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.reset_params()

    @staticmethod
    def weight_init(m, init_type = 'kaiming', gain = 0.02, scale = 0.1):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain = gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain = 1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
                m.weight.data *= scale
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain = gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def compute_loss(self, recon = None,
                     gt = None, alpha = [ 0.85, 1., 0.45, 0.5, 0.8,0.75 ]):
        # TODO? 权重赋值
        alpha = [self.pixel_weight,
                 self.ssim_weight,
                 self.vgg_weight,
                 self.color_weight,
                 self.gradient_weight,
                 self.hist_weight]


        a1 = alpha[ 0 ]
        a2 = alpha[ 1 ]
        a3 = alpha[ 2 ]
        a4 = alpha[ 3 ]
        a5 = alpha[ 4 ]
        a6 = alpha[5]


        # TODO? 消融实验Baseline
        if self.pixel_weight:
            mseloss = a1 * torch.mean(torch.abs(recon - gt) ** 2)
        else:
            mseloss = torch.tensor([ 0. ], requires_grad = False).to(recon.device)

        if self.ssim_loss:
            ssimloss = (1. - SSIM()(recon, gt))
            ssimloss *= a2
        else:
            ssimloss = torch.tensor([ 0. ], requires_grad = False).to(recon.device)


        if self.vgg_loss:
         vggloss = a3 * VGGLoss()(recon, gt)

        else:
            vggloss = torch.tensor([ 0. ], requires_grad = False).to(recon.device)


        if self.color_loss:
         colorloss = a4 * lab_loss(recon, gt)

        else:
            colorloss = torch.tensor([ 0. ], requires_grad = False).to(recon.device)

        if self.gradient_loss:
            gradientloss = a5 * GradientLoss()(recon, gt)
        else:
            gradientloss = torch.tensor([ 0. ], requires_grad = False).to(recon.device)

        if self.histogram_loss:
            histogramloss = a6 * histogram_loss(recon, gt)
        else:
            histogramloss = torch.tensor([ 0. ], requires_grad = False).to(recon.device)


        total_loss = (mseloss + ssimloss + histogramloss + colorloss + gradientloss + vggloss)  #

        return mseloss, ssimloss, vggloss, colorloss, gradientloss, histogramloss, total_loss

    def forward(self,
                input = None,
                gt = None,
                is_traing = False):


        feature1 = self.model1(input)
        feature2 = self.model2(feature1)
        feature3 = self.model3(feature2)

        if self.is_use_dalte:
            feature3 = self.dalte1(feature3)

        feature4 = self.model4(feature3)
        feature4 = torch.cat([ feature4, feature3 ], dim = 1)

        if self.is_use_freq:
            feature4 = self.model_mix4(feature4)
        if self.is_use_dalte:
            feature4 = self.dalte2(feature4)

        feature5 = self.model5(feature4)
        feature5 = self.pixel_shuffle(feature5)
        feature5 = self.model5_act(feature5)
        feature5 = torch.cat([ feature5, feature2 ], dim = 1)

        feature6 = self.model6(feature5)
        feature6 = self.pixel_shuffle(feature6)
        feature6 = self.model6_act(feature6)
        feature6 = torch.cat([ feature6, feature1 ], dim = 1)

        output = self.model7(feature6)

        rec = {}

        output = output + input
        output = torch.clamp(output, min = 0.0, max = 1.0)

        # ---------------------------------------#

        # --------------------------------------#
        #   TODO? 图像重构
        # --------------------------------------#
        rec[ 'x_pred' ] = output

        if is_traing:
            # TODO?

            l_ycrbr = torch.tensor([ 0. ], requires_grad = False)
            mseloss, ssimloss, vggloss, colorloss, gradientloss, histogramloss, total_loss = self.compute_loss(output,
                                                                                                               gt)

            rec[ 'loss_dict' ] = (total_loss)
            rec[ 'rec_loss' ] = mseloss
            rec[ 'l_VGG_loss' ] = vggloss
            rec[ 'l_ssim_loss' ] = ssimloss
            rec[ 'l_color_loss' ] = colorloss
            rec[ 'l_gradient_loss' ] = gradientloss
            rec[ 'histogram_loss' ] = histogramloss
            rec[ 'l_ycrbr' ] = l_ycrbr

            return rec

        return rec


# Define a resnet block
class ResnetBlock22(nn.Module):
    def __init__(self, dim, padding_type, activation = nn.ReLU(True), use_dropout = False):
        super(ResnetBlock22, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, activation, use_dropout)
        self.se = SEBlock(dim)

    def build_conv_block(self, dim, padding_type, activation, use_dropout):
        conv_block = [ ]
        p = 0
        if padding_type == 'reflect':
            conv_block += [ nn.ReflectionPad2d(1) ]
        elif padding_type == 'replicate':
            conv_block += [ nn.ReplicationPad2d(1) ]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [ nn.Conv2d(dim, dim, kernel_size = 3, padding = p),
                        nn.BatchNorm2d(dim),
                        activation ]
        if use_dropout:
            conv_block += [ nn.Dropout(0.5) ]

        p = 0
        if padding_type == 'reflect':
            conv_block += [ nn.ReflectionPad2d(1) ]
        elif padding_type == 'replicate':
            conv_block += [ nn.ReplicationPad2d(1) ]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [ nn.Conv2d(dim, dim, kernel_size = 3, padding = p) ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = self.se(out) + x
        return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # TODO?
    parser.add_argument('--down_type', type = str, default = 'wtconv', help = 'normal,akconv,mffa,pconv,wtconv,ours')
    parser.add_argument('--is_use_dalte', action = 'store_true')
    parser.add_argument('--is_use_freq', action = 'store_true')
    parser.add_argument('--fre_type', type = str, default = 'ours', help = 'dm,four,ours')

    # TODO?
    parser.add_argument('--pixel', type = float, default = 0.85, help = 'pixel weight')
    parser.add_argument('--ssim', type = float, default = 1., help = 'ssim weight')
    parser.add_argument('--vgg', type = float, default = 0.45, help = 'vgg weight')
    parser.add_argument('--color', type = float, default = 0.5, help = 'color weight')
    parser.add_argument('--gradient', type = float, default = 0.8, help = 'gradient weight')
    parser.add_argument('--hist', type = float, default = 0.75, help = 'hist weight')

    # TODO?
    parser.add_argument('--pixel_loss', action = 'store_true')
    parser.add_argument('--ssim_loss', action = 'store_true')
    parser.add_argument('--color_loss', action = 'store_true')
    parser.add_argument('--gradient_loss', action = 'store_true')
    parser.add_argument('--hist_loss', action = 'store_true')
    parser.add_argument('--vgg_loss', action = 'store_true')

    opt = parser.parse_args()
    opt = vars(opt)


    # TODO? 参数配置
    print(opt)







    x = torch.randn((1, 128, 64, 64))

    # TODO?
    #model = Down(in_planes = 128,out_planes = 256)
    model = DilateConv(inc = 128)
    #model = MixFreFeature(inplanes = 128,outplanes = 256)



    # model = GlobalGenerator(input_nc = 3, output_nc = 3,opt = opt)
    # model = model.eval()
    # z = model(x)
    # print(z[ 'x_pred' ].shape)



    from thop import profile

    flops,params = profile(model,(x,),verbose = False)
    print(flops / 1e9,params/1e6)




