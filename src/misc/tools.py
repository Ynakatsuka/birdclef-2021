import math
import os

import numpy as np
import torch
from tqdm import tqdm


def evaluate(
    lightning_module,
    hooks,
    config,
    mode=["validation", "test"],
    return_predictions=False,
    tta=0,
):
    print("---------------------------------------------------------------")
    print("Evaluate")

    def concatenate(v):
        # not a list or empty
        if not isinstance(v, list) or not v:
            return v

        # ndarray
        if isinstance(v[0], np.ndarray):
            return np.concatenate(v, axis=0)

        return v

    metric_dict = {}
    final_outputs = []

    lightning_module.eval()
    lightning_module.cuda()

    with torch.no_grad():
        for dl_dict in lightning_module.dataloaders:
            if not dl_dict["mode"] in (mode):
                continue

            aggregated_outputs = []
            aggregated_labels = []

            dataloader, split = dl_dict["dataloader"], dl_dict["split"]
            batch_size = dataloader.batch_size
            total_size = len(dataloader.dataset)
            total_step = math.ceil(total_size / batch_size)

            tbar = tqdm(enumerate(dataloader), total=total_step)
            for i, data in tbar:
                x = data["x"].cuda()

                outputs = lightning_module(x)
                outputs = hooks.post_forward_fn(outputs)
                outputs = outputs.detach().cpu().numpy()

                if tta > 0:
                    tta_outputs = [outputs]
                    for _ in range(tta - 1):
                        lightning_module.model.apply_tta = True
                        outputs = lightning_module(x)
                        outputs = hooks.post_forward_fn(outputs)
                        outputs = outputs.detach().cpu().numpy()
                        tta_outputs.append(outputs)
                    lightning_module.model.apply_tta = False
                    outputs = np.mean(tta_outputs, axis=0)

                aggregated_outputs.append(outputs)

                if "y" in data.keys():
                    aggregated_labels.append(data["y"].numpy())

            aggregated_outputs = concatenate(aggregated_outputs)
            aggregated_labels = concatenate(aggregated_labels)

            final_outputs.append(aggregated_outputs)

            if config.trainer.evaluation.save_prediction:
                if not os.path.exists(config.trainer.evaluation.dirpath):
                    os.makedirs(config.trainer.evaluation.dirpath)
                path = os.path.join(
                    config.trainer.evaluation.dirpath,
                    f"{split}_{config.trainer.evaluation.name}",
                )
                np.save(path, aggregated_outputs)

            if (dl_dict["mode"] == "validation") and (hooks.metric_fn is not None):
                for name, func in hooks.metric_fn.items():
                    result = func(aggregated_outputs, aggregated_labels)
                    if isinstance(result, tuple):
                        score, result_dict = result[0], result[1]
                        metric_dict[f"{split}_{name}_on_all"] = score
                        for key, value in result_dict.items():
                            metric_dict[f"{split}_{key}_on_all"] = value
                    else:
                        metric_dict[f"{split}_{name}_on_all"] = result

    if return_predictions:
        return metric_dict, final_outputs
    else:
        return metric_dict
