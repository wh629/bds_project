import os
import numpy

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_exp_name(dataset, model, lr, bs, max_epochs, seed):
    return f"{dataset}_{model}_lr_{lr}_bs_{bs}_max-epochs_{max_epochs}_seed_{seed}"


def decode_exp_name(exp_name):
    dataset, model = exp_name.split("_")[:2]
    lr, bs, max_epochs, seed = exp_name.split("_")[3::2]
    lr, bs, max_epochs, seed = float(lr), int(bs), int(max_epochs), int(seed)
    return dataset, model, lr, bs, max_epochs, seed


def make_command(dataset, model, lr, bs, max_epochs, seed, gpu_capacity, data_dir, results_dir, accumulate):

    exp_name = f"{encode_exp_name(dataset, model, lr, bs, max_epochs, seed)}"
    
    if accumulate:
        accumulation = int(numpy.ceil(bs / gpu_capacity))
    else:
        accumulation = 1
    
    command = (
        f'{os.path.join(repo_dir, "code", "main.py")} '
        f"--exp_name {exp_name} "
        f"--data_name {dataset} "
        f"--model {model} "
        f"--batch_size {bs} "
        f"--grad_accum {accumulation} "
        f"--lr {lr} "
        f"--max-epochs {max_epochs} "
        f"--data_dir {data_dir} "
        f"--save_dir {results_dir} "
    )

    return command
