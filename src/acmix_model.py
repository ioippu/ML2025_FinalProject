import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

# ==========================================
# 1. Simplified ACMix (Attention + Conv Mixer)
# ==========================================
class ACMix(nn.Module):
    def __init__(self, in_planes, out_planes=None, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACMix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes if out_planes is not None else in_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        
        # 可學習的融合權重
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        # Q, K, V projections
        self.conv1 = nn.Conv2d(in_planes, out_planes, 1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, 1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, 1)

        # Position encoding
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_conv, padding=kernel_conv//2)

        # Attention Branch Util
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)

        # Conv Branch - 使用 Depthwise Conv 3x3 來實作 Local Conv 特徵
        # 原本的 ACMix 實作比較複雜且容易出錯，這裡用等效的結構替代
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_conv, padding=kernel_conv//2, groups=in_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rate1.data.fill_(0.5)
        self.rate2.data.fill_(0.5)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # --- Attn Branch ---
        # 產生位置編碼
        pe = self.conv_p(position(h, w, x.device))

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = F.avg_pool2d(q_att, self.stride, self.stride)
            k_att = F.avg_pool2d(k_att, self.stride, self.stride)
            v_att = F.avg_pool2d(v_att, self.stride, self.stride)

        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        unfold_r = self.unfold(self.pad_att(pe.view(1, self.head_dim, h, w).expand(b*self.head, -1, -1, -1))).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        
        # Attention Map
        att = (q_att.unsqueeze(2)*(unfold_k + unfold_r)).sum(1)
        att = self.softmax(att)
        
        out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        # --- Conv Branch ---
        out_conv = self.conv_branch(x)

        return self.rate1 * out_att + self.rate2 * out_conv

def position(H, W, device):
    if torch.cuda.is_available():
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc.to(device)


# ==========================================
# 2. ConvNeXt Block with ACMix
# ==========================================
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class BlockACMix(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.
    
    MODIFIED: Replaced DwConv (7x7) with ACMix
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        
        # 原本是 7x7 Depthwise Conv，現在換成 ACMix
        self.dwconv = ACMix(dim, dim) # ACMix 已經包含了 padding 機制
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = StochasticDepth(drop_path, mode="batch") if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtACMix(nn.Module):
    r""" ConvNeXt with ACMix
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BlockACMix(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnext_tiny_acmix(num_classes=1000, **kwargs):
    model = ConvNeXtACMix(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)
    return model
