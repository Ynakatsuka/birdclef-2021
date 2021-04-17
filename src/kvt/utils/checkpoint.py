from collections import OrderedDict


def fix_dp_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key
        if key.startswith("model."):
            name = key[6:]
        new_state_dict[name] = value

    return new_state_dict


def fix_mocov2_state_dict(state_dict):
    """
    Ref: https://github.com/facebookresearch/CovidPrognosis/blob/master/cp_examples/sip_finetune/sip_finetune.py
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.encoder_q."):
            k = k.replace("model.encoder_q.", "")
            new_state_dict[k] = v

    return new_state_dict


def load_state_dict_on_same_size(model, state_dict):
    new_state_dict = OrderedDict()
    for name, param in model.named_parameters():
        if (name in state_dict.keys()) and (param.shape == state_dict[name].shape):
            new_state_dict[name] = state_dict[name]
        else:
            print(f"Skip {name}")

    for name, param in model.named_buffers():
        if (name in state_dict.keys()) and (param.shape == state_dict[name].shape):
            new_state_dict[name] = state_dict[name]
        else:
            print(f"Skip {name}")
            print(name in state_dict.keys())
            print(param.shape)
            print(state_dict[name].shape)

    model.load_state_dict(new_state_dict, strict=False)

    return model
