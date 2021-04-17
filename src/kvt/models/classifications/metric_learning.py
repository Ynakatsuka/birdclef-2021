# from torch import nn

# from .cnn import CnnModel
# from ..layers import MagrginLinear, SEBlock


# class CnnModelMetric(CnnModel):
#     def __init__(self, backbone, num_classes, pooling_config, pretrained="imagenet", n_extract=512):
#         super().__init__(backbone, num_classes, pooling_config, pretrained)
#         extractor = []
#         for _ in num_classes:
#             fc = nn.Sequential(
#                 SEBlock(self.out_shape),
#                 nn.Dropout(0.25),
#                 nn.Linear(self.out_shape, n_extract),
#                 #nn.BatchNorm1d(n_extract)
#             )
#             extractor.append(fc)
#         self.extractor = nn.ModuleList(extractor)

#         modules = []
#         for n_cls in num_classes:
#             margin = MagrginLinear(n_extract, n_cls)
#             modules.append(margin)
#         self.outputs = nn.ModuleList(modules)

#     def forward(self, x, label=None, target_names=None):
#         x = self.net(x)
#         output = []
#         if label is not None:
#             for extractor, output_layer, k in zip(self.extractor, self.outputs, target_names):
#                 x_feats = extractor(x)
#                 output.append(output_layer(x_feats, label[k]))
#         else:
#             for extractor, output_layer in zip(self.extractor, self.outputs):
#                 x_feats = extractor(x)
#                 output.append(output_layer(x_feats))
#         return output
