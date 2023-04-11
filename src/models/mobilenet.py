import torch
import torch.nn as nn
import torch.nn as nn
import torchvision.models as tvmodels


# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]

#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup

#         layers = []
#         if expand_ratio != 1:
#             layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
#             layers.append(nn.BatchNorm2d(hidden_dim))
#             layers.append(nn.ReLU6(inplace=True))
#         layers.extend([
#             nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
#                       1, groups=hidden_dim, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU6(inplace=True),
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         ])
#         self.conv = nn.Sequential(*layers)

#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)


# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=20, width_mult=1.0,
#                  inverted_residual_setting=None, dropout_p=None,
#                  round_nearest=8, fc_dims=None,
#                  pretrained=True,  loss={"xent", "htri"},  pooling='avg',  **kwargs,):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         self.loss = loss
#         self.pooling = pooling
#         input_channel = 224
#         last_channel = 1280

#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 (1, 16, 1, 1),
#                 (6, 24, 2, 2),
#                 (6, 32, 3, 2),
#                 (6, 64, 4, 2),
#                 (6, 96, 3, 1),
#                 (6, 160, 3, 2),
#                 (6, 320, 1, 1),
#             ]

#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element tuple, got {}".format(inverted_residual_setting))

#         input_channel = _make_divisible(
#             input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(
#             last_channel * max(1.0, width_mult), round_nearest)
#         features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
#                     nn.BatchNorm2d(input_channel),
#                     nn.ReLU6(inplace=True)]
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(
#                     block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#         features.append(
#             nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False))
#         features.append(nn.BatchNorm2d(self.last_channel))
#         features.append(nn.ReLU6(inplace=True))
#         self.features = nn.Sequential(*features)
#         self.fc = self._construct_fc_layer(
#             fc_dims, last_channel, dropout_p)
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )
#         if pretrained:
#             self._load_pretrained_weights()
#         self._init_params()

#     def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
#         if fc_dims is None:
#             self.feature_dim = input_dim
#             return None

#         assert isinstance(
#             fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

#         layers = []
#         for dim in fc_dims:
#             layers.append(nn.Linear(input_dim, dim))
#             layers.append(nn.BatchNorm1d(dim))
#             layers.append(nn.ReLU(inplace=True))
#             input_dim = dim

#         self.feature_dim = fc_dims[-1]

#         return nn.Sequential(*layers)

#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def _load_pretrained_weights(self):
#         model = torch.hub.load('pytorch/vision:v0.10.0',
#                                'mobilenet_v2', pretrained=True)
#         state_dict = model.state_dict()
#         custom_state_dict = self.state_dict()

#         for k, v in state_dict.items():
#             if k in custom_state_dict and custom_state_dict[k].shape == v.shape:
#                 custom_state_dict[k] = v

#         self.load_state_dict(custom_state_dict)

#     def forward(self, x):
#         x = self.features(x)
#         # x = x.view(x.size(0), -1)

#         if self.fc is not None:
#             x = self.fc(x)

#         if not self.training:
#             return x

#         y = self.classifier(x)

#         if self.loss == {"xent"}:
#             return y
#         elif self.loss == {"xent", "htri"}:
#             return y, x
#         else:
#             raise KeyError(f"Unsupported loss: {self.loss}")


# def mobilenet_v2(num_classes, pooling='avg', loss={"xent", "htri"}, pretrained=True, **kwargs):
#     model = MobileNetV2(pretrained=pretrained,
#                         num_classes=num_classes, **kwargs)
#     return model


__all__ = ["mobilenet_v3_small", "vgg16"]


class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, fc_dims=None, dropout_p=None, **kwargs):
        super().__init__()

        self.loss = loss
        self.fc_dims = fc_dims
        self.backbone = tvmodels.__dict__[name](pretrained=pretrained)
        self.feature_dim = self.backbone.classifier[0].in_features
        self.fc = self._construct_fc_layer(
            self.fc_dims, self.feature_dim, dropout_p)

        # overwrite the classifier used for ImageNet pretrianing
        # nn.Identity() will do nothing, it's just a place-holder
        self.backbone.classifier = nn.Identity()
        if fc_dims is not None:
            self.classifier = nn.Linear(fc_dims, num_classes)
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def forward(self, x):
        v = self.backbone(x)
        if self.fc_dims is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)


def mobilenet_v3_small(num_classes, loss={"xent", "htri"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def mobilenet_v3_small_fc_512(num_classes, loss={"xent", "htri"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        fc_dims=[512],
        ** kwargs,
    )
    return model
