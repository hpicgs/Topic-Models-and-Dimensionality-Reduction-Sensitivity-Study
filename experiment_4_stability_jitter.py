import os
from multiprocessing import Manager, Process, cpu_count

import numpy as np
from argparse import ArgumentParser

from tqdm import tqdm

from commons import get_y_paths, get_rel_path, get_file_identifiers
from metrics import get_all_metrics_stability
import pandas as pd


def get_arguments():
    parser = ArgumentParser(description="This script executes experiment 1 analysing all metric correlations")
    parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                        help="The path to the results directory of the dataset for which the correlations shall "
                             "be analysed.")
    parser.add_argument("--random_seed", dest='random_seed', type=int, default=0,
                        help="The jitter amount that shall be fixed. Defaults to 0")
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=-1,
                        help="Determines how many jobs will be executed in parallel. Default 1 means that all kernels "
                             "will be used and 1 means that"
                             "only one kernel will be used (equal to use no parallelization).")
    args = parser.parse_args()
    return args.dataset_path, args.random_seed, args.n_jobs


def main():
    dataset_path, random_seed, n_jobs = get_arguments()
    random_seed_dirs = []
    random_seed_dirs_correct_jitter = []
    dir_pairs = []
    processes = []

    for random_seed_dir in os.listdir(dataset_path):
        if "random_seed" in random_seed_dir and str(random_seed) in random_seed_dir:
            random_seed_dirs.append(os.path.join(dataset_path, random_seed_dir))
            break  # Only one random_seed per jitter amount

    for random_seed_dir in random_seed_dirs:
        for jitter_dir in os.listdir(random_seed_dir):
            if "jitter_amount" in jitter_dir:
                random_seed_dirs_correct_jitter.append(os.path.join(random_seed_dir, jitter_dir))

    for i in range(len(random_seed_dirs_correct_jitter)):
        for j in range(i + 1, len(random_seed_dirs_correct_jitter)):
            dir_pairs.append((random_seed_dirs_correct_jitter[i], random_seed_dirs_correct_jitter[j]))

    for dir1, dir2 in dir_pairs:
        file_identifiers1 = get_file_identifiers([file for file in os.listdir(dir1) if file.endswith(".npy") and not file.endswith("y.npy")])
        file_identifiers2 = get_file_identifiers([file for file in os.listdir(dir2) if file.endswith(".npy") and not file.endswith("y.npy")])
        common_file_identifiers = list(set(file_identifiers1).intersection(set(file_identifiers2)))

        y_file_names = get_y_paths(dir1, constant_y=True)
        y = np.load(y_file_names[0])
        manager = Manager()
        return_dict = manager.dict()
        jitter1 = [sub_dir.split("_")[-1] for sub_dir in dir1.split(os.path.sep) if "jitter" in sub_dir][0]
        jitter2 = [sub_dir.split("_")[-1] for sub_dir in dir2.split(os.path.sep) if "jitter" in sub_dir][0]

        for common_identifier in common_file_identifiers:
            layout1_path = os.path.join(dir1, [file for file in os.listdir(dir1) if common_identifier in file and
                                               file.endswith(".npy")][0])
            layout2_path = os.path.join(dir2, [file for file in os.listdir(dir2) if common_identifier in file and
                                               file.endswith(".npy")][0])
            key = (layout1_path, layout2_path)
            processes.append((jitter1, jitter2, Process(target=get_all_metrics_stability,
                                                        args=(layout1_path, layout2_path, y,
                                                              return_dict, "euclidean", True)), key, return_dict))

    if n_jobs != -1:
        n_jobs = min(n_jobs, cpu_count())
    else:
        n_jobs = cpu_count()

    running_jobs = []
    dataset_name = os.path.basename(dataset_path)
    df_name = "results_experiment_4_" + dataset_name + ".csv"
    if os.path.isfile(df_name):
        df = pd.read_csv(df_name)
        overall_results = df.values.tolist()
        relativ_df_path_list = [get_rel_path(pair, dataset_name) for pair
                                in df[["file_name_1", "file_name_2"]].values.tolist()]
    else:
        df = None
        overall_results = []
        relativ_df_path_list = []

    for i, (jitter1, jitter2, process, key, return_dict) in enumerate(tqdm(processes)):
        if (df is not None and float(jitter1) in df["jitter_1"].values.tolist()
                and float(jitter2) in df["jitter_2"].values.tolist()
                and get_rel_path(list(key), dataset_name) in relativ_df_path_list):
            write_if_appropriate(df_name, i, overall_results)
            continue

        process.start()
        running_jobs.append((process, key, jitter1, jitter2, return_dict))
        if i == len(processes) - 1 or len(running_jobs) == n_jobs:
            for running_process, running_key, running_seed1, running_seed2, running_return_dict in running_jobs:
                running_process.join()
                try:
                    cur_results = running_return_dict[running_key]
                except KeyError:
                    continue
                entry = [running_seed1, running_seed2]
                entry.extend(cur_results)
                overall_results.append(entry)
                running_process.close()
            running_jobs = []

        write_if_appropriate(df_name, i, overall_results)
    write_if_appropriate(df_name, 99, overall_results, write_anyway=True)


def write_if_appropriate(df_name, i, overall_results, write_anyway=False):
    if (i + 1) % 10 == 0 or write_anyway:  # Write results every 100 iterations
        df_new = pd.DataFrame(overall_results, columns=["jitter_1", "jitter_2", "spearman_correlation",
                                                        "pearson_correlation",
                                                        "cluster_ordering", "rotation",
                                                        "distance_consistency", "silhouette_coefficient",
                                                        "trustworthiness", "continuity", "local_continuity",
                                                        "mrre_missing", "mrre_false", "label_preservation",
                                                        "file_name_1",
                                                        "file_name_2"])
        df_new.to_csv(df_name, index=False)


if __name__ == "__main__":
    main()
