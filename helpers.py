import json
import os
from collections import defaultdict

import torch
import torchvision.utils


def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


def load_checkpoint(param, use_checkpoint, device):
    """
    loads the most recent (non-corrupted) checkpoint or the final model
    :param param: args number
    :param use_checkpoint: checkpointed or final model
    :return:
    """
    if not use_checkpoint:
        return torch.load(f'./model/diff-params-ARGS={param}/params-final.pt', map_location=device)
    else:
        checkpoints = os.listdir(f'./model/diff-params-ARGS={param}/checkpoint')
        checkpoints.sort(reverse=True)
        for i in checkpoints:
            try:
                file_dir = f"./model/diff-params-ARGS={param}/checkpoint/{i}"
                loaded_model = torch.load(file_dir, map_location=device)
                break
            except RuntimeError:
                continue
        return loaded_model


def load_parameters(device):
    """
    Loads the trained parameters for the detection model
    :return:
    """
    import sys

    if len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = os.listdir("./model")
    if ".DS_Store" in params:
        params.remove(".DS_Store")

    if params[0] == "CHECKPOINT":
        use_checkpoint = True
        params = params[1:]
    else:
        use_checkpoint = False

    print(params)
    for param in params:
        if param.isnumeric():
            output = load_checkpoint(param, use_checkpoint, device)
        elif param[:4] == "args" and param[-5:] == ".json":
            output = load_checkpoint(param[4:-5], use_checkpoint, device)
        elif param[:4] == "args":
            output = load_checkpoint(param[4:], use_checkpoint, device)
        else:
            raise ValueError(f"Unsupported input {param}")

        if "args" in output:
            args = output["args"]
        else:
            try:
                with open(f'./test_args/args{param[17:]}.json', 'r') as f:
                    args = json.load(f)
                args['arg_num'] = param[17:]
                args = defaultdict_from_json(args)
            except FileNotFoundError:
                raise ValueError(f"args{param[17:]} doesn't exist for {param}")

        if "noise_fn" not in args:
            args["noise_fn"] = "gauss"

        return args, output


def main():
    pass


if __name__ == '__main__':
    main()
