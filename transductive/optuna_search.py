import argparse
import json
import os
import random
import shlex
from types import SimpleNamespace

import numpy as np
import torch

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "optuna is required for this search script. Install it in the active "
        "environment, then rerun optuna_search.py."
    ) from exc

from base_model import BaseModel
from load_data import DataLoader
from utils import select_gpu


def dataset_name_from_path(data_path):
    parts = data_path.replace("\\", "/").rstrip("/").split("/")
    return parts[-1]


def suggest_family(trial):
    params = {
        "fact_ratio": trial.suggest_float("fact_ratio", 0.85, 0.95),
        "lr": trial.suggest_float("lr", 1e-3, 8e-3, log=True),
        "decay_rate": trial.suggest_float("decay_rate", 0.990, 1.000),
        "lamb": trial.suggest_float("lamb", 1e-6, 2e-4, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
        "attn_dim": trial.suggest_categorical("attn_dim", [3, 5, 8]),
        "n_layer": trial.suggest_int("n_layer", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.40),
        "act": trial.suggest_categorical("act", ["relu", "tanh", "idd"]),
        "n_batch": trial.suggest_categorical("n_batch", [10, 20, 50]),
        "n_tbatch": trial.suggest_categorical("n_tbatch", [20, 50, 100]),
    }
    return params


def suggest_umls(trial):
    params = {
        "fact_ratio": trial.suggest_float("fact_ratio", 0.85, 0.95),
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "decay_rate": trial.suggest_float("decay_rate", 0.985, 0.999),
        "lamb": trial.suggest_float("lamb", 1e-5, 5e-4, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64, 96]),
        "attn_dim": trial.suggest_categorical("attn_dim", [3, 5, 8]),
        "n_layer": trial.suggest_int("n_layer", 4, 8),
        "dropout": trial.suggest_float("dropout", 0.0, 0.20),
        "act": trial.suggest_categorical("act", ["relu", "tanh", "idd"]),
        "n_batch": trial.suggest_categorical("n_batch", [5, 10, 20]),
        "n_tbatch": trial.suggest_categorical("n_tbatch", [20, 30, 50]),
    }
    return params


def suggest_wn18rr(trial):
    params = {
        "fact_ratio": trial.suggest_float("fact_ratio", 0.93, 0.98),
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "decay_rate": trial.suggest_float("decay_rate", 0.990, 0.9995),
        "lamb": trial.suggest_float("lamb", 1e-6, 2e-4, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
        "attn_dim": trial.suggest_categorical("attn_dim", [3, 5, 8]),
        "n_layer": trial.suggest_int("n_layer", 4, 6),
        "dropout": trial.suggest_float("dropout", 0.0, 0.25),
        "act": trial.suggest_categorical("act", ["relu", "tanh", "idd"]),
        "n_batch": trial.suggest_categorical("n_batch", [50, 80, 100]),
        "n_tbatch": trial.suggest_categorical("n_tbatch", [30, 50, 80]),
    }
    return params


def suggest_fb15k_237(trial):
    params = {
        "fact_ratio": trial.suggest_float("fact_ratio", 0.97, 0.995),
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "decay_rate": trial.suggest_float("decay_rate", 0.988, 0.999),
        "lamb": trial.suggest_float("lamb", 1e-5, 5e-4, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
        "attn_dim": trial.suggest_categorical("attn_dim", [3, 5, 8]),
        "n_layer": trial.suggest_int("n_layer", 3, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.30),
        "act": trial.suggest_categorical("act", ["relu", "tanh", "idd"]),
        "n_batch": trial.suggest_categorical("n_batch", [3, 5, 10]),
        "n_tbatch": trial.suggest_categorical("n_tbatch", [1, 2, 5]),
    }
    return params


def suggest_nell(trial):
    params = {
        "fact_ratio": trial.suggest_float("fact_ratio", 0.90, 0.98),
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "decay_rate": trial.suggest_float("decay_rate", 0.988, 0.999),
        "lamb": trial.suggest_float("lamb", 1e-5, 5e-4, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
        "attn_dim": trial.suggest_categorical("attn_dim", [3, 5, 8]),
        "n_layer": trial.suggest_int("n_layer", 4, 6),
        "dropout": trial.suggest_float("dropout", 0.05, 0.40),
        "act": trial.suggest_categorical("act", ["relu", "tanh", "idd"]),
        "n_batch": trial.suggest_categorical("n_batch", [3, 5, 10]),
        "n_tbatch": trial.suggest_categorical("n_tbatch", [1, 2, 5]),
    }
    return params


def suggest_yago(trial):
    params = {
        "fact_ratio": trial.suggest_float("fact_ratio", 0.98, 0.998),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "decay_rate": trial.suggest_float("decay_rate", 0.990, 0.9995),
        "lamb": trial.suggest_float("lamb", 1e-5, 5e-4, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
        "attn_dim": trial.suggest_categorical("attn_dim", [3, 5, 8]),
        "n_layer": trial.suggest_int("n_layer", 3, 5),
        "dropout": trial.suggest_float("dropout", 0.05, 0.40),
        "act": trial.suggest_categorical("act", ["relu", "tanh", "idd"]),
        "n_batch": trial.suggest_categorical("n_batch", [1, 2, 3, 5]),
        "n_tbatch": trial.suggest_categorical("n_tbatch", [1, 2]),
    }
    return params


def add_sheaf_family(trial, params):
    params.update({
        "topk_nodes": trial.suggest_categorical("topk_nodes", [20, 50, 100, 200]),
        "gamma": trial.suggest_float("gamma", 0.3, 0.9),
        "beta": trial.suggest_float("beta", 0.25, 1.25),
        "lambda_sheaf": trial.suggest_float("lambda_sheaf", 1e-5, 1e-2, log=True),
        "lambda_dyn": trial.suggest_float("lambda_dyn", 1e-6, 1e-3, log=True),
    })
    return params


def add_sheaf_umls(trial, params):
    params.update({
        "topk_nodes": trial.suggest_categorical("topk_nodes", [30, 50, 100]),
        "gamma": trial.suggest_float("gamma", 0.3, 0.9),
        "beta": trial.suggest_float("beta", 0.25, 1.25),
        "lambda_sheaf": trial.suggest_float("lambda_sheaf", 1e-5, 1e-2, log=True),
        "lambda_dyn": trial.suggest_float("lambda_dyn", 1e-6, 1e-3, log=True),
    })
    return params


def add_sheaf_wn18rr(trial, params):
    params.update({
        "topk_nodes": trial.suggest_categorical("topk_nodes", [500, 800, 1000, 1200]),
        "gamma": trial.suggest_float("gamma", 0.4, 0.95),
        "beta": trial.suggest_float("beta", 0.25, 1.25),
        "lambda_sheaf": trial.suggest_float("lambda_sheaf", 1e-6, 1e-2, log=True),
        "lambda_dyn": trial.suggest_float("lambda_dyn", 1e-7, 1e-3, log=True),
    })
    return params


def add_sheaf_fb15k_237(trial, params):
    params.update({
        "topk_nodes": trial.suggest_categorical("topk_nodes", [200, 500, 1000, 2000]),
        "gamma": trial.suggest_float("gamma", 0.4, 0.95),
        "beta": trial.suggest_float("beta", 0.25, 1.25),
        "lambda_sheaf": trial.suggest_float("lambda_sheaf", 1e-6, 1e-2, log=True),
        "lambda_dyn": trial.suggest_float("lambda_dyn", 1e-7, 1e-3, log=True),
    })
    return params


def add_sheaf_nell(trial, params):
    params.update({
        "topk_nodes": trial.suggest_categorical("topk_nodes", [500, 1000, 2000, 4000]),
        "gamma": trial.suggest_float("gamma", 0.4, 0.95),
        "beta": trial.suggest_float("beta", 0.25, 1.25),
        "lambda_sheaf": trial.suggest_float("lambda_sheaf", 1e-6, 1e-2, log=True),
        "lambda_dyn": trial.suggest_float("lambda_dyn", 1e-7, 1e-3, log=True),
    })
    return params


def add_sheaf_yago(trial, params):
    params.update({
        "topk_nodes": trial.suggest_categorical("topk_nodes", [500, 1000, 2000, 4000]),
        "gamma": trial.suggest_float("gamma", 0.4, 0.95),
        "beta": trial.suggest_float("beta", 0.25, 1.25),
        "lambda_sheaf": trial.suggest_float("lambda_sheaf", 1e-6, 1e-2, log=True),
        "lambda_dyn": trial.suggest_float("lambda_dyn", 1e-7, 1e-3, log=True),
    })
    return params


SEARCH_SPACES = {
    "family": suggest_family,
    "umls": suggest_umls,
    "WN18RR": suggest_wn18rr,
    "fb15k-237": suggest_fb15k_237,
    "nell": suggest_nell,
    "YAGO": suggest_yago,
}

SHEAF_SEARCH_SPACES = {
    "family": add_sheaf_family,
    "umls": add_sheaf_umls,
    "WN18RR": add_sheaf_wn18rr,
    "fb15k-237": add_sheaf_fb15k_237,
    "nell": add_sheaf_nell,
    "YAGO": add_sheaf_yago,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_options(params, loader, args, dataset, trial_number):
    result_dir = os.path.join(args.results_dir, dataset, f"trial_{trial_number:04d}")
    os.makedirs(result_dir, exist_ok=True)
    return SimpleNamespace(
        n_ent=loader.n_ent,
        n_rel=loader.n_rel,
        perf_file=os.path.join(result_dir, "perf.txt"),
        mem_file=os.path.join(result_dir, "mem.txt") if args.write_mem else "",
        model_name=args.model_name,
        **params,
    )


def build_trial_config(params, args, dataset, trial_number, trial_seed, gpu):
    return {
        "trial_number": trial_number,
        "study_name": args.study_name,
        "dataset": dataset,
        "data_path": args.data_path,
        "model_name": args.model_name,
        "seed": args.seed,
        "trial_seed": trial_seed,
        "gpu": gpu,
        "max_epochs_per_trial": args.max_epochs_per_trial,
        "early_stop_patience": args.early_stop_patience,
        "params": params,
    }


def write_config_header(path, config):
    with open(path, "a+", encoding="utf-8") as f:
        f.write("[CONFIG] " + json.dumps(config, sort_keys=True) + "\n")


def load_initial_params(source):
    if not source:
        return []
    text = source.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    if text.startswith("{") or text.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = parse_cmd_flat_object(text)
    else:
        with open(source, "r", encoding="utf-8") as f:
            payload = json.load(f)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        "--init_params must be a JSON object, a list of JSON objects, or a JSON file path."
    )


def parse_cmd_flat_object(value):
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError(f"Invalid inline JSON: {value}")
    body = text[1:-1].strip()
    if not body:
        return {}

    result = {}
    for item in body.split(","):
        if ":" not in item:
            raise ValueError(f"Invalid initial parameter item: {item}")
        key, raw_value = item.split(":", 1)
        key = key.strip().strip("'\"")
        result[key] = parse_cmd_value(raw_value.strip())
    return result


def parse_cmd_value(value):
    value = value.strip().strip("'\"")
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def command_to_string(args, params):
    cmd = [
        "python",
        "train.py",
        "--data_path",
        args.data_path,
        "--model_name",
        args.model_name,
        "--gpu",
        str(args.gpu if args.gpu is not None else 0),
    ]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    if os.name == "nt":
        return " ".join(shlex.quote(str(item)) for item in cmd)
    return shlex.join(str(item) for item in cmd)


def make_objective(args, dataset):
    suggest_fn = SEARCH_SPACES[dataset]
    sheaf_suggest_fn = SHEAF_SEARCH_SPACES[dataset]

    def objective(trial):
        trial_seed = args.seed + trial.number
        set_seed(trial_seed)

        params = suggest_fn(trial)
        if args.model_name == "sheaf_momentum":
            params = sheaf_suggest_fn(trial, params)
        trial.set_user_attr("params", params)
        print(f"\n==> Trial {trial.number} command:", flush=True)
        print(command_to_string(args, params), flush=True)
        print(f"==> Trial {trial.number} params: {json.dumps(params, sort_keys=True)}", flush=True)

        loader = DataLoader(args.data_path, fact_ratio=params["fact_ratio"])
        opts = build_options(params, loader, args, dataset, trial.number)
        write_config_header(
            opts.perf_file,
            build_trial_config(
                params,
                args,
                dataset,
                trial.number,
                trial_seed,
                getattr(args, "selected_gpu", args.gpu),
            ),
        )

        model = BaseModel(opts, loader)
        best_mrr = 0.0
        best_epoch = -1
        best_log = ""
        stale_epochs = 0

        for epoch in range(args.max_epochs_per_trial):
            valid_mrr, out_str = model.train_batch(epoch=epoch)
            print(out_str, end="", flush=True)
            with open(opts.perf_file, "a+", encoding="utf-8") as f:
                f.write(out_str)
            if valid_mrr > best_mrr:
                best_mrr = float(valid_mrr)
                best_epoch = epoch
                best_log = out_str
                stale_epochs = 0
            else:
                stale_epochs += 1
            trial.set_user_attr("last_epoch", epoch)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_log", best_log)
            trial.set_user_attr("stale_epochs", stale_epochs)
            print(
                f"==> Trial {trial.number} epoch {epoch} "
                f"best_mrr={best_mrr:.4f} best_epoch={best_epoch} "
                f"stale={stale_epochs}/{args.early_stop_patience}",
                flush=True,
            )
            if stale_epochs >= args.early_stop_patience:
                trial.set_user_attr("early_stopped", True)
                print(f"==> Trial {trial.number} early stopped by patience.", flush=True)
                break

        return best_mrr

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna search for transductive RED-GNN.")
    parser.add_argument("--data_path", type=str, default="data/family/")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--max_epochs_per_trial", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--init_params",
        type=str,
        default=None,
        help="Initial parameter group as inline JSON or a JSON file path.",
    )
    parser.add_argument("--results_dir", type=str, default="results/optuna")
    parser.add_argument("--write_mem", action="store_true")
    parser.add_argument("--model_name", type=str, default="redgnn", choices=["redgnn", "sheaf_momentum"])
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = dataset_name_from_path(args.data_path)
    if dataset not in SEARCH_SPACES:
        known = ", ".join(sorted(SEARCH_SPACES))
        raise ValueError(f"No search space for dataset '{dataset}'. Known datasets: {known}")

    os.makedirs(args.results_dir, exist_ok=True)
    study_name = args.study_name or f"{args.model_name}_trans_{dataset}"
    storage = args.storage or f"sqlite:///{os.path.join(args.results_dir, dataset, study_name + '.db')}"
    os.makedirs(os.path.join(args.results_dir, dataset), exist_ok=True)

    set_seed(args.seed)
    gpu = args.gpu if args.gpu is not None else select_gpu()
    if gpu is None:
        gpu = 0
    args.selected_gpu = gpu
    torch.cuda.set_device(gpu)
    print("gpu:", gpu)
    print("study_name:", study_name)
    print("storage:", storage)
    print("dataset:", dataset)

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=3)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    for params in load_initial_params(args.init_params):
        study.enqueue_trial(params)

    study.optimize(make_objective(args, dataset), n_trials=args.n_trials)

    print("best_value:", study.best_value)
    print("best_params:", json.dumps(study.best_params, indent=2, sort_keys=True))
    print("best_trial:", study.best_trial.number)


if __name__ == "__main__":
    main()
