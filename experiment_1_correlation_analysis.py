import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, cpu_count, Manager

from commons import get_layouts_paths, get_y_paths
from metrics import metric_label_preservation, get_all_metrics_stability

# base_path = "/media/tim/b95060e6-b7d9-4de5-9c2c-fd9613641903/results_random_seed_0"
datasets = ["20_newsgroups", "lyrics", "seven_categories"]


def get_arguments():
    parser = ArgumentParser(description="This script executes experiment 1 analysing all metric correlations")
    parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                        help="The path to the results directory of the dataset for which the correlations shall "
                             "be analysed.")
    parser.add_argument('--n_files', dest='n_files', type=int, default=-1,
                        help="Determines how many files are considered as a basis for correlation. Number of total "
                             "correlations will be n_files*n_correlations. Default -1 means that all files "
                             "are considered")
    parser.add_argument('--n_correlations', dest='n_correlations', type=int,
                        help="Determines number of new correlations calculated for each in file. Number of total "
                             "correlations will be n_files*n_correlations. Default -1 means that all correlations "
                             "are considered")
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=-1,
                        help="Determines how many jobs will be executed in parallel. Default 1 means that all "
                             "only one kernel will be used (equal to use no parallelization).")

    args = parser.parse_args()
    return args.dataset_path, args.n_files, args.n_correlations, args.n_jobs


def main():
    dataset_path, n_files, n_correlations, n_jobs = get_arguments()
    dataset = os.path.basename(dataset_path)
    res_file_name = dataset + "_correlations.csv"
    if not os.path.isdir(dataset_path):
        raise IsADirectoryError("Base path must be a valid directory with results")

    layout_file_paths = get_layouts_paths(dataset_path)
    y_file_paths = get_y_paths(dataset_path, constant_y=True)

    random.seed(0)
    if len(y_file_paths) > 1:
        tmp = list(zip(layout_file_paths, y_file_paths))
        random.shuffle(tmp)
        layout_file_paths, y_file_paths = zip(*tmp)
    else:
        random.shuffle(layout_file_paths)
        y_file_paths = y_file_paths * len(layout_file_paths)

    combined_lists = list(zip(layout_file_paths, y_file_paths))
    # Upper triangular matrix except the diagonal
    if n_files != -1:
        len_outer_loop = min(len(combined_lists), n_files)
    else:
        len_outer_loop = len(combined_lists)
    total_correlations = len_outer_loop * n_correlations
    manager = Manager()
    return_dict = manager.dict()
    processes = []

    print("Preparing processes")
    with tqdm(total=total_correlations) as progress_bar:
        for i in range(len_outer_loop):
            y = np.load(y_file_paths[i])
            layout1_path = layout_file_paths[i]

            if n_correlations != -1:
                upper_bound_inner_loop = min(n_correlations + i + 1, len(combined_lists))
            else:
                upper_bound_inner_loop = len(combined_lists)

            for j in range(i + 1, upper_bound_inner_loop):
                layout2_path = layout_file_paths[j]
                processes.append(Process(target=get_all_metrics_stability, args=(layout1_path, layout2_path, y,
                                                                                 return_dict)))

                progress_bar.update(1)

    df = None
    if os.path.isfile(res_file_name):
        df = pd.read_csv(res_file_name)
        processes = processes[len(df):]

    print("Executing processes")
    with tqdm(total=len(processes)) as progress_bar:
        if n_jobs != -1:
            n_jobs = min(n_jobs, cpu_count())
        else:
            n_jobs = cpu_count()

        if len(processes) % n_jobs == 0:
            n_iterations = len(processes) // n_jobs
        else:
            n_iterations = (len(processes) // n_jobs) + 1

        for i in range(n_iterations):
            start = i * n_jobs
            end = min((i + 1) * n_jobs, len(processes))

            for j in range(start, end):
                processes[j].start()

            for j in range(start, end):
                processes[j].join()

            all_values = return_dict.values()

            df_new = pd.DataFrame(all_values, columns=["spearman_correlation", "pearson_correlation",
                                                       "cluster_ordering", "rotation",
                                                       "distance_consistency", "silhouette_coefficient",
                                                       "trustworthiness", "continuity", "local_continuity",
                                                       "mrre_missing", "mrre_false", "label_preservation",
                                                       "file_name_1",
                                                       "file_name_2"])
            if df is not None:
                df_merged = df.append(df_new, ignore_index=True)
                df_merged.to_csv(res_file_name, index=False)
            else:
                df_new.to_csv(res_file_name, index=False)

            progress_bar.update(n_jobs)


def add_label_preservation(required_len=1000):
    correlation_file_path = "/home/tim/PycharmProjects/slurm_test"
    target_column_name = "label_preservation"
    for file in os.listdir(correlation_file_path):
        if not file.endswith("correlations.csv"):
            continue

        df = pd.read_csv(os.path.join(correlation_file_path, file))
        if (target_column_name in df.columns and -1 not in df[target_column_name].tolist()) or len(df) != required_len:
            continue

        print(file)
        files = df[["file_name_1", "file_name_2"]].to_numpy()

        if target_column_name in df.columns:
            old_values = df[target_column_name].tolist()
        else:
            old_values = [-1] * len(files)
        label_preservations = []
        y_file_paths = get_y_paths(os.path.dirname(files[0][0]), constant_y=True)
        y_path = y_file_paths[0]
        y = np.load(y_path)
        for i, (file_name1, file_name2) in enumerate(tqdm(files)):
            if old_values[i] != -1:
                label_preservations.append(old_values[i])
                continue
            if os.path.isfile(file_name1) and os.path.isfile(file_name2):
                scatter1 = np.load(file_name1)
                scatter2 = np.load(file_name2)
                label_preservations.append(metric_label_preservation(scatter1, scatter2, y, k=7))
            else:
                print("Couldn't find at least one file. I will append -1")
                label_preservations.append(-1)
        df[target_column_name] = label_preservations
        df.to_csv(os.path.join(correlation_file_path, file), index=False)


if __name__ == "__main__":
    main()
