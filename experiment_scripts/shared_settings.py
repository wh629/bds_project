import os
import numpy

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_exp_name(dataset, model, max_length, lr, bs, max_epochs, seed):
    return f"{dataset}_{model}_max-length_{max_length}_lr_{lr}_bs_{bs}_max-epochs_{max_epochs}_seed_{seed}"


def decode_exp_name(exp_name):
    dataset, dataset_second, model = exp_name.split("_")[:3]
    max_length, lr, bs, max_epochs, seed = exp_name.split("_")[4::2]
    max_length, lr, bs, max_epochs, seed = int(max_length), float(lr), int(bs), int(max_epochs), int(seed)
    return dataset+"_"+dataset_second, model, lr, bs, max_epochs, seed


def make_command(dataset,
                 model,
                 max_length,
                 lr,
                 bs,
                 max_epochs,
                 seed,
                 gpu_capacity,
                 data_dir,
                 results_dir,
                 accumulate,
                 check_int,
                 log_int,
                 patience,
                 early_check,
                 additional,
                 content,
                 ):

    exp_name = f"{encode_exp_name(dataset, model, max_length, lr, bs, max_epochs, seed)}"
    
    if accumulate:
        accumulation = int(numpy.ceil(bs / gpu_capacity))
        bs_fill = gpu_capacity
    else:
        accumulation = 1
        bs_fill = bs

    command = (
        f'{os.path.join(repo_dir, "code", "main.py")} '
        f"--exp_name {exp_name} "
        f"--data_name {dataset} "
        f"--input_length {max_length} "
        f"--model {model} "
        f"--batch_size {bs_fill} "
        f"--grad_accum {accumulation} "
        f"--lr {lr} "
        f"--max_epochs {max_epochs} "
        f"--data_dir {data_dir} "
        f"--save_dir {results_dir} "
        f"--check_int {check_int} "
        f"--log_int {log_int} "
        f"--patience {patience} "
        f"--early_check {early_check} "
    )

    if additional:
        n_other = int(len(content.split(','))-1)
        command += (f"--content {content} "
                    f"--n_others {n_other} ")

    return command
