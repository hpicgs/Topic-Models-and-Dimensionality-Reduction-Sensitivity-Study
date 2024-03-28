import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from commons import get_file_identifiers, get_rel_path

whitelisted_tms = ["bert_all_mpnet_base_v2", "bert_all_distilroberta_v1",
                   "lda_linear_combined", "lda", "lsi_linear_combined_tfidf", "lsi_linear_combined", "lsi_tfidf", "lsi",
                   "nmf_linear_combined_tfidf", "nmf_linear_combined", "nmf_tfidf", "nmf",
                   "doc2vec", "bow", "tfidf"]  # Always sort from most complicated to least
whitelisted_drs = ["som", "tsne", "umap", "mds"]
whitelisted_datasets = ["20_newsgroups", "lyrics", "seven_categories"]


def get_arguments():
    parser = ArgumentParser(description="This script executes experiment 1 analysing all metric correlations")
    parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                        help="The path to the results directory of the dataset for which the correlations shall "
                             "be analysed.")
    args = parser.parse_args()
    return args.dataset_path


def filter_wrong_distances_metric_and_alpha(values, file_index):
    correct_values = []
    for value_row in values:
        file_name = value_row[file_index]

        if "lda" in file_name and ("symmetric" in file_name or "asymmetric" in file_name):
            continue

        if "lda" in file_name and ("umap_" in file_name or "tsne_" in file_name or "mds_" in file_name):
            if "jensenshannon" in file_name:
                correct_values.append(value_row)
        elif "nmf" in file_name and ("umap_" in file_name or "tsne_" in file_name or "mds_" in file_name):
            if "cosine" in file_name:
                correct_values.append(value_row)
        else:
            if "jensenshannon" not in file_name:
                correct_values.append(value_row)

    return correct_values


def test_interchange_file_names(values, columns, file_name_column="file_name_1", file_name_column_2="file_name_2"):
    index1 = columns.index(file_name_column)
    index2 = columns.index(file_name_column_2)
    new_values = []

    for i, value_row in enumerate(tqdm(values)):
        file_name_1 = value_row[index1]
        file_name_2 = value_row[index2]
        append_result = True
        for j in range(i, len(values)):
            file_name_1_comp = values[j][index1]
            file_name_2_comp = values[j][index2]

            if file_name_1 == file_name_2_comp and file_name_2 == file_name_1_comp:
                append_result = False
        if append_result:
            new_values.append(value_row)

    return new_values


def postprocess_experiment_3(file_path, file_name_column="file_name_1"):
    df = pd.read_csv(file_path)
    columns = df.columns.tolist()
    if "config" in columns or "TM" in columns:
        return

    file_column_index = columns.index(file_name_column)
    new_values = []
    values = df.values.tolist()

    for value_row in tqdm(values):
        if file_name_column in value_row:
            continue

        cur_dr, cur_tm = get_cur_dr_and_tm(file_column_index, value_row)
        sec_cur_dr, sec_cur_tm = get_cur_dr_and_tm(file_column_index + 1, value_row)

        if cur_dr != sec_cur_dr or cur_tm != sec_cur_tm:
            continue

        if cur_dr != "" and cur_tm != "":
            value_row.extend([cur_dr, cur_tm])
            new_values.append(value_row)

    new_values = filter_wrong_distances_metric_and_alpha(values=new_values, file_index=file_column_index)
    columns.extend(["config", "TM"])
    df = pd.DataFrame(new_values, columns=columns)
    df["file_name_1"] = get_rel_path(df["file_name_1"], get_dataset_name(file_path))
    df["file_name_2"] = get_rel_path(df["file_name_2"], get_dataset_name(file_path))
    df = df.drop_duplicates(subset=["file_name_1", "file_name_2"])
    df = pd.DataFrame(test_interchange_file_names(df.values.tolist(), df.columns.tolist()), columns=columns)
    df = df.sort_values(by=["config", "TM"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.to_csv(file_path, index=False)


def get_cur_dr_and_tm(file_column_index, value_row):
    cur_file_name = value_row[file_column_index]
    cur_dr = ""
    cur_tm = ""
    for valid_tm in whitelisted_tms:
        if valid_tm in cur_file_name:
            cur_tm = valid_tm
            break
    for valid_dr in whitelisted_drs:
        if valid_dr in cur_file_name:
            cur_dr = valid_dr
            break

    if cur_dr == "" or cur_tm == "":
        print(value_row)  # Something fishy happened here

    return cur_dr, cur_tm


def postprocess_experiment_4(file_path, undesired_pairs, file_name_column="file_name_1", undesired_column_1="jitter_1",
                             undesired_column_2="jitter_2"):
    df = pd.read_csv(file_path)
    columns = df.columns.tolist()
    if "config" in columns or "TM" in columns:
        return

    file_column_index = columns.index(file_name_column)
    undesired_column_index_1 = columns.index(undesired_column_1)
    undesired_column_index_2 = columns.index(undesired_column_2)
    new_values = []
    values = df.values.tolist()

    for value_row in tqdm(values):
        if file_name_column in value_row:
            continue

        cur_pair = [value_row[undesired_column_index_1], value_row[undesired_column_index_2]]
        if cur_pair in undesired_pairs:
            continue

        cur_dr, cur_tm = get_cur_dr_and_tm(file_column_index, value_row)
        sec_cur_dr, sec_cur_tm = get_cur_dr_and_tm(file_column_index + 1, value_row)
        if not (cur_dr == sec_cur_dr and cur_tm == sec_cur_tm):
            continue
        if cur_dr != "" and cur_tm != "":
            value_row.extend([cur_dr, cur_tm])
            new_values.append(value_row)

    new_values = filter_wrong_distances_metric_and_alpha(values=new_values, file_index=file_column_index)
    columns.extend(["config", "TM"])
    df = pd.DataFrame(new_values, columns=columns)
    df["file_name_1"] = get_rel_path(df["file_name_1"], get_dataset_name(file_path))
    df["file_name_2"] = get_rel_path(df["file_name_2"], get_dataset_name(file_path))
    df = df.drop_duplicates(subset=["file_name_1", "file_name_2"])
    df = pd.DataFrame(test_interchange_file_names(df.values.tolist(), df.columns.tolist()), columns=columns)
    critical_columns = df[["jitter_1", "jitter_2"]].values.tolist()
    truth_values = [(el != [0.25, 0.1]) for el in critical_columns]
    print(all(truth_values))
    df = df.iloc[truth_values]
    df = df.sort_values(by=["jitter_1", "jitter_2"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.to_csv(file_path, index=False)


def postprocess_experiment_2(file_path, file_name_column="file_name_1"):
    df = pd.read_csv(file_path)
    columns = df.columns.tolist()
    file_column_index = columns.index(file_name_column)
    df = pd.DataFrame(filter_wrong_distances_metric_and_alpha(values=df.values.tolist(), file_index=file_column_index),
                      columns=columns)

    df["file_name_1"] = get_rel_path(df["file_name_1"], get_dataset_name(file_path))
    df["file_name_2"] = get_rel_path(df["file_name_2"], get_dataset_name(file_path))
    df = df.drop_duplicates(subset=["config", "short_tm_name", "file_name_1", "file_name_2"])
    df = pd.DataFrame(test_interchange_file_names(df.values.tolist(), df.columns.tolist()), columns=columns)
    df = df.sort_values(by=["config", "short_tm_name"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.to_csv(file_path, index=False)


def get_dataset_name(file_path):
    dataset = ""
    for dataset_name in whitelisted_datasets:
        if dataset_name in file_path:
            dataset = dataset_name
    return dataset


def unify_results(base_path):
    res_file_names = set()
    dir_paths = []

    for cur_path, directories, files in os.walk(base_path):
        for directory in directories:
            dir_path = os.path.join(cur_path, directory)
            dir_paths.append(dir_path)
            for results_file in os.listdir(dir_path):
                res_file_names.add(results_file)

    for res_file_name in list(res_file_names):
        if not res_file_name.endswith(".csv"):
            continue

        cur_df = None
        for results_dir in dir_paths:
            cur_res_file_path = os.path.join(results_dir, res_file_name)
            if os.path.isfile(cur_res_file_path):
                df = pd.read_csv(cur_res_file_path)
                if ("config" in df.columns and "2" not in res_file_name) or "TM" in df.columns:
                    continue

                if cur_df is None:
                    cur_df = df.copy()
                else:
                    cur_df = cur_df.append(df)

        if cur_df is not None:
            if "results_experiment_2" in res_file_name or "results_2" in res_file_name:
                cur_df = cur_df.drop_duplicates(subset=["config", "short_tm_name", "file_name_1", "file_name_2"])
            else:
                cur_df = cur_df.drop_duplicates(subset=["file_name_1", "file_name_2"])
            cur_df.to_csv(os.path.join(base_path, res_file_name), index=False)


def main():
    dataset_path = get_arguments()

    if not os.path.isdir(dataset_path):
        raise OSError("Couldn't find given directory")
    
    unify_results(base_path=dataset_path)
    for file in os.listdir(dataset_path):
        cur_path = os.path.join(dataset_path, file)

        if "results_experiment_2" in file or "results_2" in file:
            postprocess_experiment_2(file_path=cur_path)

        if "results_experiment_3" in file or "results_3" in file:
            postprocess_experiment_3(file_path=cur_path)

        if "results_experiment_4" in file or "results_4" in file:
            postprocess_experiment_4(file_path=cur_path, undesired_pairs=[[0.1, 0.25], [0.25, 0.1]])


if __name__ == "__main__":
    main()
