import kvt
import kvt.hooks


@kvt.HOOKS.register
class KeySelectPostForwardHook(kvt.hooks.PostForwardHookBase):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs):
        return outputs["clipwise_output"]


@kvt.HOOKS.register
class FramewiseMaxPostForwardHook(kvt.hooks.PostForwardHookBase):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs):
        res, _ = outputs["framewise_logit"].max(dim=1)
        return res
