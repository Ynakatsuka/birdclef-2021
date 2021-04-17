# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Loss(torch.nn.modules.Module):
#     def __init__(self, cls_weight=1.0):
#         self.loss_cls = nn.BCEWithLogitsLoss()
#         self.cls_weight = cls_weight

#     def __call__(self, loss_fn, outputs, labels, data, is_train):
#         '''
#             outputs: {cls_logits, logits}
#             labels: {cls_labels, labels}
#         '''
#         seg_labels = labels['labels']
#         B, C, H, W = seg_labels.size()

#         cls_tp = 1

#         assert self.loss_cls is not None

#         loss_cls = self.loss_cls(input=outputs['cls_logits'], target=labels['cls_labels'])

#         logits = outputs['logits']
#         logits = logits * cls_tp
#         logits = logits.view(-1, H, W)
#         seg_labels = seg_labels.view(-1, H, W)
#         loss_seg = loss_fn(input=logits, target=seg_labels)

#         loss = loss_seg + loss_cls * self.cls_weight

#         loss_dict = {
#             'loss': loss,
#             'loss_seg': loss_seg,
#             'loss_cls': loss_cls
#         }

#         return loss_dict
