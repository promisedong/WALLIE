import torch
import torch.nn as nn
import torch.nn.functional as F

class PA(nn.Module):
    def __init__(self,in_planes = None):
        super(PA, self).__init__()

        self.layer = nn.Sequential(
                    nn.Conv2d(in_channels = in_planes,
                              out_channels = in_planes,
                              kernel_size = 3,
                              padding = 1,
                              stride = 1),
                    nn.BatchNorm2d(in_planes),
                    nn.LeakyReLU(inplace = True)
        )

    def forward(self,x = None,x1 = None):
        out = F.sigmoid(self.layer(x)) * x
        # TODO? DEBUG
        return out + x1

class CA(nn.Module):
    def __init__(self, in_planes = None,reduction = 4):
        super(CA, self).__init__()
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        hidden =  (in_planes // reduction) if (in_planes // reduction)!=0 else in_planes
        self.layer1 = nn.Linear(in_planes,hidden)
        self.layer2 = nn.Linear(hidden,in_planes)
    def forward(self,x):
        out = self.globalpool(x)
        # TODO?
        b,c,h,w = out.shape
        out1 = self.layer1(out.view(b,-1))
        out1 = F.sigmoid(self.layer2(out1))
        out1 = out1.view((b,c,h,w))
        return out1 * x

class SA(nn.Module):
    def __init__(self, in_planes = None):
        super(SA, self).__init__()
        self.layer = nn.Sequential(
                    nn.Conv2d(in_channels = in_planes,
                              out_channels = 1,
                              kernel_size = 3,
                              padding = 1,
                              stride = 1),
                    nn.LeakyReLU(inplace = True)
        )
    def forward(self,x):
        out = F.sigmoid(self.layer(x))
        return x * out

class MAFM(nn.Module):
    def __init__(self,in_planes = None,in_planes1 = None):
        super(MAFM, self).__init__()

        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels = in_planes1,out_channels = in_planes,
                              kernel_size = 3,padding = 1),
                    nn.LeakyReLU(inplace = True),
                    nn.Conv2d(in_channels = in_planes, out_channels = in_planes,
                      kernel_size = 1),
        )
        # TODO? 上分支
        self.ca = CA(in_planes = in_planes)
        self.upa = PA(in_planes = in_planes)
        # TODO? 下分支
        self.sa = SA(in_planes = in_planes)
        self.bpa = PA(in_planes = in_planes)
        self.beta = nn.Parameter(torch.randn(1,),requires_grad = True)
        self.lamba = nn.Parameter(torch.randn(1,),requires_grad = True)
    def forward(self,x = None,x1 = None):
        assert x.shape == x1.shape,"特征维度不匹配"
        x1 = self.conv(x1) #TODO?
        # TODO?
        out = x + x1
        out1 = F.sigmoid(self.upa(self.ca(out) + out,out))
        out2 = F.sigmoid(self.bpa(self.sa(out1) + out,out))
        W = self.beta * out1 + self.lamba * out2
        output = W * x + (1. - W) * x1 + out
        return output


def window_partition(x, window_szie = 8):
    b, c, h, w = x.shape

    pad = [ 0, 0, 0, 0 ]
    if h % window_szie != 0:
        pad[ 3 ] = (window_szie - h % window_szie)% window_szie

    if w % window_szie != 0:
        pad[ 1 ] = (window_szie - w % window_szie)% window_szie

    output = F.pad(x, pad = pad, mode = 'reflect')
    *_, H, W = output.shape

    output = output.contiguous().view(b, c, window_szie, H // window_szie, window_szie, W // window_szie)
    output = output.permute((0, 1, 2, 4, 3, 5)).contiguous().view(b, c * window_szie * window_szie, H // window_szie,
                                                                  W // window_szie)

    return output, h, w, c, H // window_szie


def window_reverse(x = None, inplanes = None, window_size = 8, patch = None):
    b, l, d = x.shape
    # TODO? BUG
    x = x.view((b, patch, l // patch, inplanes, window_size, window_size))
    x = x.contiguous().permute((0, 3, 4, 1, 5, 2))

    H = x.shape[ 2 ] * x.shape[ 3 ]
    x = x.contiguous().view(b, inplanes, H, -1)

    return x


class _cross_attention(nn.Module):
    def __init__(self,
                 inplanes, window_size = 8):
        super(_cross_attention, self).__init__()

        c = inplanes * window_size * window_size

        self.inplanes = c

    def forward(self,
                q = None,
                k = None,
                v = None):
        b, *_ = q.shape

        # TODO?
        q, qh, qw, qc, patch = window_partition(q, window_szie = 8)

        q = q.contiguous().permute((0, 2, 3, 1)).view(b, -1, self.inplanes)

        k, kh, kw, kc, _ = window_partition(k, window_szie = 8)
        k = k.contiguous().permute((0, 2, 3, 1)).view(b, -1, self.inplanes)

        v, vh, vw, vc, _ = window_partition(v, window_szie = 8)
        v = v.contiguous().permute((0, 2, 3, 1)).view(b, -1, self.inplanes)

        _q = F.normalize(q, p = 2)
        _k = F.normalize(k, p = 2)
        _v = F.normalize(v, p = 2)

        similarity = _q @ _k.transpose(-2, -1) * (self.inplanes ** (-0.5))

        sim = F.softmax(similarity, dim = -1) @ _v

        out = window_reverse(x = sim, inplanes = qc, patch = patch)

        return out[ :, :, :qh, :qw ]

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class CDEM(nn.Module):
    def __init__(self,in_planes = None,out_planes = None):
        super(CDEM, self).__init__()

        self.layer1 = nn.Conv2d(in_channels = in_planes,out_channels = in_planes,
                                kernel_size = 1)

        self.layer2 = nn.Conv2d(in_channels = in_planes,out_channels = in_planes,
                                kernel_size = 1)

        self.crossatt = _cross_attention(inplanes = in_planes)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels = in_planes, out_channels = in_planes // 4 ,
                      kernel_size = 3, padding = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels = in_planes // 4, out_channels = in_planes,
                      kernel_size = 1, padding = 0),
        )

        self.ffn = PreNorm(in_planes,FeedForward(dim = in_planes))
        self.beta = nn.Parameter(torch.ones(1,),requires_grad = True)
        self.lamba1 = nn.Parameter(torch.ones(1,),requires_grad = True)
        self.lamba2 = nn.Parameter(torch.ones(1, ), requires_grad = True)
        self.alpha = nn.Parameter(torch.ones(1,),requires_grad = True)

        self.fin = nn.Conv2d(in_channels = in_planes, out_channels = out_planes,
                                kernel_size = 3, padding = 1)
    def forward(self,x = None,x1 = None):
        out1 = self.layer1(x)
        out2 = self.layer2(x1)
        # TODO? eq.(6)
        out = self.fuse(self.crossatt(out1,out2,out2))
        z1 = self.alpha * out + self.beta * x1
        z2 = self.lamba1 * self.ffn(z1.permute(0,2,3,1)).permute((0,3,1,2)) + self.lamba2 * z1
        z2 = self.fin(z2) + x
        return z2


class SG(nn.Module):
    def __init__(self,in_planes = None,k = 2):
        super(SG, self).__init__()
        self.k = k

        self.layer = nn.Conv2d(in_channels = in_planes,out_channels = in_planes,
                               kernel_size = 3,padding = 1)

    def forward(self,x):
        logits = self.layer(x)
        noise = torch.randn_like(logits) * 0.1
        per = logits + noise
        top_v,indices = torch.topk(per,k = self.k,dim = 1)
        w = F.softmax(top_v,dim = 1)
        spare = torch.zeros_like(logits)
        spare.scatter_(1,indices,w)
        return spare

if __name__ == '__main__':
    x = torch.randn((1,8,64,64))
    x1 = torch.randn((1,8,64,64))

    ca = CA(8,4)
    out_ca = ca(x)
    print(out_ca.shape)

    sa = SA(8)
    out_sa = sa(x)
    print(out_sa.shape)

    # pa = PA(8)
    # out_pa = pa(x)
    # print(out_pa.shape)

    # TODO? aligned
    model = MAFM(in_planes = 8,in_planes1 = 8)
    output = model(x,x1)

    model = CDEM(in_planes = 8)
    out = model(x,x1)
    print(out.shape)





