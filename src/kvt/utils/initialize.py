import torch


def initialize_model(config, model, backbone_lr_ratio=None, encoder_lr_ratio=None):
    params = model.parameters()

    # set learning rate
    if backbone_lr_ratio is None:
        backbone_lr_ratio = config.trainer.train.backbone_lr_ratio
    if encoder_lr_ratio is None:
        encoder_lr_ratio = config.trainer.train.encoder_lr_ratio

    if backbone_lr_ratio is not None:
        print("---------------------------------------------------------------")
        if backbone_lr_ratio == 0:
            print("Freezed Layers: ")
            for child in list(model.children())[:-1]:
                print(child)
                for param in child.parameters():
                    param.requires_grad = False
        else:
            print("Layer-wise Learning Rate:")
            base_lr = config.trainer.optimizer.params.lr
            params = []  # replace params
            for child in list(model.children())[:-1]:
                params.append(
                    {
                        "params": list(child.parameters()),
                        "lr": base_lr * backbone_lr_ratio,
                    }
                )
                print(child, " lr: ", base_lr * backbone_lr_ratio)

    elif encoder_lr_ratio is not None:
        print("---------------------------------------------------------------")
        raise NotImplementedError

    return model, params


def reinitialize_model(config, model):
    params = model.parameters()

    for child in list(model.children())[:-1]:
        for param in child.parameters():
            param.requires_grad = True

    return model, params
