"""
Hyper parameter tuning based on RoBERTa on WSC
"""

import random
import os
import argparse
from shared_settings import make_command


def select_candidates(dataset):
    lr_candidates = [1e-5, 3e-5, 5e-5]
    bs_candidates = [8, 16, 32]
    max_epochs_candidates = [1, 2]
    seed_range = 1e6
    lr = lr_candidates[random.randrange(0, len(lr_candidates), 1)]
    bs = bs_candidates[random.randrange(0, len(bs_candidates), 1)]
    max_epochs = max_epochs_candidates[random.randrange(0, len(max_epochs_candidates), 1)]
    seed = random.randint(0, seed_range)

    return lr, bs, max_epochs, seed


def submit_trials(args):
    jobs = []

    for trial in range(args.n_trials):
        # select candidates for trial
        lr, bs, max_epochs, seed = select_candidates(args.dataset)
        command = make_command(
            args.dataset,
            args.model,
            args.max_length,
            lr,
            bs,
            max_epochs,
            seed,
            args.gpu_capacity,
            args.data_dir,
            args.results_dir,
            args.accumulate,
            args.check_int,
            args.log_int,
        )
        sbatch_file = os.path.join(args.repo_dir, "experiment_scripts", f"{args.user}.sbatch")
        jobs.append(f'COMMAND="{command}" sbatch {sbatch_file}\n')

    with open(os.path.join(args.repo_dir, "experiment_scripts", "submit_sbatch.sh"), "a") as f:
        for one_job in jobs:
            f.write(one_job)

    return


if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Run BDS Experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.getenv("BDS_RESULTS_DIR", os.path.join(repo_dir, "results")),
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.getenv("BDS_DATA_DIR", os.path.join(repo_dir, "data"))
    )
    parser.add_argument("--user", type=str)

    parser.add_argument("--n-trials", type=int, help="number of trials")
    parser.add_argument(
        "--dataset",
        type=str,
        default="reviews_UIC",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='bert-base-uncased',
        help='name of RLN network. default is BERT',
        choices={'bert-base-uncased',
                 'bert-large-uncased',
                 'roberta-base',
                 'roberta-large',
                 'albert-base-v2',
                 'albert-xxlarge-v2'
                 'albert-base-v1',
                 'albert-xxlarge-v1'
                 },
    )
    
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gpu-capacity", type=int, default=2)
    parser.add_argument('--check_int', type=int, default=1000)
    parser.add_argument('--log_int',type=int, default=100)
    parser.add_argument("--accumulate", action='store_true')

    args = parser.parse_args()
    args.repo_dir = repo_dir
    submit_trials(args)
