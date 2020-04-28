import json
import pandas as pd
import os
import argparse

from shared_settings import decode_exp_name


def collect_results(args):
    results = {
        "dataset": [],
        "model": [],
        "learning_rate": [],
        "batch_size": [],
        "max_epochs": [],
        "seed": [],
        "best_val_accuracy": [],
        "best_val_f1": [],
        "best_val_loss": [],
        "best_test_accuracy": [],
        "best_test_f1": [],
        "best_test_loss": [],
        "exp_name": [],
        "best_iter": [],
        "current_iter": [],
        "total_iter": [],
        "total_epochs": [],
        "early_stop": []
    }

    def record_exp(one_exp_result):
        dataset, framing, lr, bs, max_epochs, seed = decode_exp_name(one_exp_result["exp_name"])
        results["dataset"].append(dataset)
        results["framing"].append(framing)
        results["learning_rate"].append(lr)
        results["batch_size"].append(bs)
        results["max_epochs"].append(max_epochs)
        results["seed"].append(seed)
        results["best_val_accuracy"].append(one_exp_result["val_acc"])
        results["best_val_f1"].append(one_exp_result["val_f1"])
        results["best_val_loss"].append(one_exp_result["val_loss"])
        results["best_test_accuracy"].append(one_exp_result["test_acc"])
        results["best_test_f1"].append(one_exp_result["test_f1"])
        results["best_test_loss"].append(one_exp_result["test_loss"])
        results["exp_name"].append(one_exp_result["exp_name"])
        results["best_iter"].append(one_exp_result["best_step"])
        results["current_iter"].append(one_exp_result["current_step"])
        results["total_iter"].append(one_exp_result["total_steps"])
        results["total_epochs"].append(one_exp_result["total_epochs"])
        results["early_stop"].append(one_exp_result["current_step"]<one_exp_result["total_steps"])
	

    with open(os.path.join(args.results_dir, "results.jsonl"), "r") as reader:
        for row in reader:
            one_exp_result = json.loads(row)
            record_exp(one_exp_result)

    df_raw = pd.DataFrame.from_dict(results)
    df_raw.sort_values(by=["model", "best_val_accuracy"], ascending=False, inplace=True)
    df_raw.to_csv(os.path.join(args.results_dir, "raw_results.csv"), index=False)
    
    return


if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Collect results and make tsv")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.getenv("BDS_RESULTS_DIR", os.path.join(repo_dir, "results")),
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.getenv("BDS_DATA_DIR", os.path.join(repo_dir, "data"))
    )

    args = parser.parse_args()
    args.repo_dir = repo_dir
    collect_results(args)
