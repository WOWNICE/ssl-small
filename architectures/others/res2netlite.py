import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['res2netlite']

# from . import get_registry
# from train.utils.export import get_exports_registry

# BACKBONES = get_registry()
# EXPORTS = get_exports_registry()

# EXPORTS.register_import_statement(item='import math', attach_to='res2netlite')
# EXPORTS.register_import_statement(item='import torch', attach_to='res2netlite')
# EXPORTS.register_import_statement(item='import torch.nn as nn', attach_to='res2netlite')
# EXPORTS.register_import_statement(item='import torch.utils.checkpoint as cp', attach_to='res2netlite')
# EXPORTS.register_import_statement(item='from torch.nn.modules.batchnorm import _BatchNorm', attach_to='res2netlite')

# @EXPORTS.register_function(attach_to='res2netlite')
def build_conv_layer(cfg, *args, **kwargs):
    cfg_ = {}
    layer = nn.Conv2d(*args, **kwargs, **cfg_)
    return layer


# @EXPORTS.register_function(attach_to='res2netlite')
def build_norm_layer(cfg, num_features, postfix=''):
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    norm_layer = nn.BatchNorm2d
    abbr = 'bn'
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


# @EXPORTS.register_function(attach_to='res2netlite')
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# @EXPORTS.register_class(attach_to='res2netlite')
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None
        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# @EXPORTS.register_class(attach_to='res2netlite')
class Res2LiteBottleneck(nn.Module):
    def __init__(self,
                 inplanes,
                 c,
                 outplanes):
        super(Res2LiteBottleneck, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, int(inplanes / 4), 1, 1, bias=False),  # /2或/4
                                   nn.BatchNorm2d(int(inplanes / 4)),
                                   nn.ReLU6(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(int(inplanes / 4), int(inplanes / 4), 3, 2, 1, groups=2, bias=False),
                                   nn.BatchNorm2d(int(inplanes / 4)),
                                   nn.ReLU6(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(int(inplanes / 4), outplanes, 1, 1, bias=False),
                                   nn.BatchNorm2d(outplanes),
                                   nn.ReLU6(inplace=True))

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# @EXPORTS.register_class(attach_to='res2netlite')
class Res2Block(Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 scales=4,
                 base_width=26,
                 base_channels=64,
                 stage_type='normal',
                 **kwargs):
        """Res2Block block for Res2NetLite.
        """
        super(Res2Block, self).__init__(inplanes, planes, **kwargs)
        # width = int(math.floor(self.planes * (base_width / base_channels)))
        channels = self.planes // 4
        width = channels // scales
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width * scales, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes, postfix=3)
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        convs = []
        bns = []
        for i in range(scales - 1):
            convs.append(
                build_conv_layer(
                    self.conv_cfg,
                    width,
                    width,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=1,
                    bias=False))
            bns.append(
                build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.planes,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            spx = torch.split(out, self.width, 1)
            sp = self.convs[0](spx[0].contiguous())
            sp = self.relu(self.bns[0](sp))
            out = sp
            for i in range(1, self.scales - 1):
                if self.stage_type == 'stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp.contiguous())
                sp = self.relu(self.bns[i](sp))
                out = torch.cat((out, sp), 1)

            out = torch.cat((out, spx[self.scales - 1]), 1)
            out = self.conv3(out)
            out = self.norm3(out)
            out += identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)
        return out


# @EXPORTS.register_class(attach_to='res2netlite')
class Res2LiteLayer(nn.Sequential):
    """Res2LiteLayer to build Res2NetLite style backbone.

    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 scales=4,
                 base_width=26,
                 **kwargs):
        self.block = block
        base_channels = kwargs['base_channels']
        downsample = None
        layers = []
        layers.append(
            block[0](inplanes, base_channels, planes))
        inplanes = planes
        for i in range(0, num_blocks):
            layers.append(
                block[1](
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scales=scales,
                    base_width=base_width,
                    **kwargs))
        super(Res2LiteLayer, self).__init__(*layers)


# @BACKBONES.register
class Res2NetLite(nn.Module):
    """Res2NetLite backbone.
    """

    arch_settings = {
        13: ((Res2LiteBottleneck, Res2Block), (3, 7, 3, 0, 0))
    }
    name = 'res2netlite'

    def __init__(self,
                 depth=13,
                 scales=4,
                 in_channels=3,
                 base_width=26,
                 num_stages=5,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 base_channels=72,
                 stem_channels=32,
                 # out_indices=(1, 2, 3, 4),
                 num_classes=1000,
                 feature_layer='classifier',
                 pretrained=False,
                 strides=(2, 2, 2, 0, 0),
                 **kwargs):
        super(Res2NetLite, self).__init__()

        self.scales = scales
        self.base_width = base_width
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 5
        self.strides = strides
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)  # 3 32

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            stage_plugins = None
            planes = base_channels * 2 ** (i + 2)  # 72*4 72*8 72*16
            if i == 3 or i == 4:
                planes = 512  # 512
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            self.feature_layer = feature_layer
        if feature_layer == 'last_conv':
            pass
        elif feature_layer == 'avgpool':
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif feature_layer == 'classifier':
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(512, num_classes)
        else:
            raise NotImplementedError
        self.init_weights()

    def make_res_layer(self, **kwargs):
        return Res2LiteLayer(
            scales=self.scales,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels=32):
        conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        _, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.stem = nn.Sequential(conv1, norm1, relu, maxpool)

    def _perform_kaiming_init_fc(self):
        torch.nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.fc.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.fc.bias, -bound, bound)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            if self.feature_layer == 'classifier':
                self._perform_kaiming_init_fc()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        if self.feature_layer == 'last_conv':
            pass
        elif self.feature_layer == 'avgpool':
            x = self.avgpool(x)
        elif self.feature_layer == 'classifier':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

def res2netlite(num_classes=1000):
    return Res2NetLite(depth=13, scales=4, base_width=26, norm_cfg=dict(type='BN', requires_grad=False),
                        num_classes=num_classes, feature_layer='classifier')

if __name__ == '__main__':

    model = Res2NetLite(depth=13, scales=4, base_width=26, norm_cfg=dict(type='BN', requires_grad=False),
                        num_classes=100, feature_layer='classifier')
    for k in model.state_dict():
        print(k)
    inputs = torch.rand(4, 3, 800, 800)
    level_outputs = model(inputs)
    print(level_outputs.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    ##feature_layer='last_conv'>>>[4, 512, 7, 7]
    ##feature_layer='avgpool'>>>[4, 512, 1, 1]
    ##feature_layer='classifier'>>>[4,num_classes]