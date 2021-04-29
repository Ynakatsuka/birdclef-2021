try:
    import torchxrayvision as xrv
except ImportError:
    xrv = None


def densenet_all_xrv(num_classes=1000, pretrained=False, **kwargs):
    weights = "all" if pretrained else None
    model = xrv.models.DenseNet(num_classes=num_classes, weights=weights, **kwargs)
    return model


def densenet_rsna_xrv(num_classes=1000, pretrained=False, **kwargs):
    weights = "rsna" if pretrained else None
    model = xrv.models.DenseNet(num_classes=num_classes, weights=weights, **kwargs)
    return model


def densenet_nih_xrv(num_classes=1000, pretrained=False, **kwargs):
    weights = "nih" if pretrained else None
    model = xrv.models.DenseNet(num_classes=num_classes, weights=weights, **kwargs)
    return model


def densenet_pc_xrv(num_classes=1000, pretrained=False, **kwargs):
    weights = "pc" if pretrained else None
    model = xrv.models.DenseNet(num_classes=num_classes, weights=weights, **kwargs)
    return model


def densenet_chex_xrv(num_classes=1000, pretrained=False, **kwargs):
    weights = "chex" if pretrained else None
    model = xrv.models.DenseNet(num_classes=num_classes, weights=weights, **kwargs)
    return model


def densenet_mimic_nb_xrv(num_classes=1000, pretrained=False, **kwargs):
    weights = "mimic_nb" if pretrained else None
    model = xrv.models.DenseNet(num_classes=num_classes, weights=weights, **kwargs)
    return model


def densenet_mimic_ch_xrv(num_classes=1000, pretrained=False, **kwargs):
    weights = "mimic_ch" if pretrained else None
    model = xrv.models.DenseNet(num_classes=num_classes, weights=weights, **kwargs)
    return model
