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



class Net1(nn.Module):
    def __init__(self,in_planes = None,reduction = 4):
        super(Net1, self).__init__()
        self.layer1 = nn.Sequential(
                     nn.Conv2d(in_channels = in_planes,
                               out_channels = in_planes ,kernel_size = 1),
                     nn.Conv2d(in_channels = in_planes ,
                      out_channels = in_planes // reduction, kernel_size = 3,padding = 1),
                     nn.Conv2d(in_channels = in_planes // reduction,
                      out_channels = in_planes // reduction, kernel_size = 3, padding = 1),
                     nn.AdaptiveAvgPool2d((1,1))
        )

        self.layer2 = nn.Sequential(
                      nn.Conv2d(in_channels = in_planes // reduction,
                      out_channels = in_planes // reduction, kernel_size = 3, padding = 1),
                      nn.Conv2d(in_channels = in_planes // reduction,
                      out_channels = in_planes , kernel_size = 3, padding = 1),
                      nn.Conv2d(in_channels = in_planes ,
                      out_channels = in_planes , kernel_size = 3, padding = 1,groups = in_planes),
                      nn.Conv2d(in_channels = in_planes ,
                      out_channels = in_planes , kernel_size = 1),
        )

        self.fin = nn.Conv2d(in_channels = in_planes ,
                      out_channels = in_planes , kernel_size = 1)
    def forward(self,x):
        out = self.layer2(self.layer1(x)) + x
        out = F.sigmoid(self.fin(out)) * x
        return out

class Net2(nn.Module):
    def __init__(self,in_planes = None,reduction = 4):
        super(Net2, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_planes ,
                      out_channels = in_planes , kernel_size = 1)

        hidden = in_planes // reduction

        self.layer1 = nn.Conv2d(in_channels = in_planes ,
                      out_channels = hidden , kernel_size = 3,padding = 1)

        self.layer2 = nn.Conv2d(in_channels = in_planes + hidden ,
                      out_channels = in_planes , kernel_size = 3,padding = 1)
        self.middle = nn.Conv2d(in_channels = in_planes + hidden ,
                      out_channels = hidden , kernel_size = 3,padding = 1)
        self.layer3 = nn.Conv2d(in_channels = in_planes  ,
                      out_channels = hidden , kernel_size = 3,padding = 1)
        self.ca = CA(hidden * 3)
        self.sa = SA(hidden * 3)
        self.fin = nn.Conv2d(in_channels = hidden * 3  ,
                      out_channels = in_planes , kernel_size = 1)
    def forward(self,x):
        out = self.conv(x)
        out1a = self.layer1(out)#
        z = torch.cat((out,out1a),dim = 1)
        out1 = self.layer2(z) + out

        out2 = self.middle(z)#
        out3 = self.layer3(out1)#
        # TODO?
        out = torch.cat((out1a,out2,out3),dim = 1)
        ca = self.ca(out)
        sa = self.sa(out)
        output = self.fin(ca + sa) + x

        return output


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

class Net3(nn.Module):
    def __init__(self,in_planes = None):
        super(Net3, self).__init__()

        self.ln = nn.LayerNorm(in_planes)
        self.conv1x1 = nn.Conv2d(in_channels = in_planes  ,
                      out_channels = in_planes , kernel_size = 1)

        self.conv3x3 = nn.Conv2d(in_channels = in_planes,
                                 out_channels = in_planes, kernel_size = 3,padding = 1)

        self.conv5x5 = nn.Conv2d(in_channels = in_planes,
                                 out_channels = in_planes, kernel_size = 5,padding = 2)
        self.fuse = nn.Sequential(nn.Conv2d(in_channels = in_planes  ,
                      out_channels = in_planes , kernel_size = 1),
                                  nn.Conv2d(in_channels = in_planes,
                                            out_channels = in_planes, kernel_size = 3,padding = 1),
                                  )

        self.ga = SG(in_planes)
        self.ca = CA(in_planes)
        self.conv = nn.Conv2d(in_channels = in_planes  ,
                      out_channels = in_planes , kernel_size = 1)


        self.ln1 = nn.LayerNorm(in_planes)
        self.middle = nn.Sequential(nn.Conv2d(in_channels = in_planes  ,
                      out_channels = in_planes , kernel_size = 1),
                                    nn.Conv2d(in_channels = in_planes,
                                              out_channels = in_planes, kernel_size = 1),
                                    SG(in_planes),
                                    CA(in_planes),
                                    nn.Conv2d(in_channels = in_planes,
                                              out_channels = in_planes, kernel_size = 1)

        )
    def forward(self,x):

        out = self.ln(x.permute(0,2,3,1)).permute((0,3,1,2))
        out1 = self.conv1x1(out)
        out2 = self.conv3x3(out)
        out3 = self.conv5x5(out)

        out = out1 + out2 + out3
        out = self.ca(self.ga(self.fuse(out)))
        out = self.conv(out) + x

        output = self.ln(out.permute(0, 2, 3, 1)).permute((0, 3, 1, 2))
        output = self.middle(output) + out
        return output

class GMMOE(nn.Module):
    def __init__(self,in_planes = None,reduction = 4):
        super(GMMOE, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        hidden = in_planes // reduction
        self.mlp = nn.Sequential(nn.Linear(in_planes,hidden),
                                 nn.ReLU(inplace = True),
                                 nn.Linear(hidden,3),
                                 nn.Softmax(dim = -1)
                                 )

        self.net1 = Net1(in_planes,reduction)
        self.net2 = Net2(in_planes,reduction)
        self.net3 = Net3(in_planes)

    def forward(self,x):
        avgpool = self.avgpool(x)
        b,c,h,w = avgpool.shape
        avgpool = avgpool.view((b,-1))
        # TODO?
        alpha = self.mlp(avgpool)
        s1 = alpha[...,0]
        s2 = alpha[...,1]
        s3 = alpha[...,2]

        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)

        return s1[:,None,None,None] * out1 +\
               s2[:,None,None,None] * out2 + \
               s3[:,None,None,None] * out3

if __name__ == '__main__':
    x = torch.randn((1,8,64,64))
    x1 = torch.randn((1,8,64,64))

    ca = CA(8,4)
    out_ca = ca(x)
    print(out_ca.shape)

    sa = SA(8)
    out_sa = sa(x)
    print(out_sa.shape)

  


    model = GMMOE(in_planes = 8)
    out = model(x)
    print(f'GMMOE:{out.shape}')


