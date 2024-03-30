import os
import time
from argparse import ArgumentParser
from multiprocessing import Manager, Process, cpu_count

from tqdm import tqdm
import numpy as np
import pandas as pd

from commons import get_y_paths
from metrics import get_all_metrics_stability


def get_arguments():
    parser = ArgumentParser(description="This script executes experiment 1 analysing all metric correlations")
    parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                        help="The path to the results directory of the dataset for which the correlations shall "
                             "be analysed.")
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=-1,
                        help="Determines how many jobs will be executed in parallel. Default 1 means that all "
                             "only one kernel will be used (equal to use no parallelization).")
    args = parser.parse_args()
    return args.dataset_path, args.n_jobs


def find_successor_tuples(parameter_list):
    successor_tuples = []
    for i in range(len(parameter_list) - 1):
        successor_tuples.append((parameter_list[i], parameter_list[i + 1]))
    return successor_tuples


def get_two_stage_parameter_pairs(dr_name, dict_of_parameters_of_dr, configs):
    for i, (parameter_name, parameter_list) in enumerate(dict_of_parameters_of_dr.items()):
        config_name = dr_name + str(i)
        configs[config_name] = ([], parameter_name)
        succesor_tuples = find_successor_tuples(dict_of_parameters_of_dr[parameter_name])
        for (parameter_name_pair, parameter_list_pair) in dict_of_parameters_of_dr.items():
            if parameter_name == parameter_name_pair:
                continue
            for parameter_value in parameter_list_pair:
                for pair_tuple in succesor_tuples:
                    new_entry = {parameter_name_pair: parameter_value, (parameter_name + "_pair"): pair_tuple}
                    configs[config_name][0].append(new_entry)

    return configs


def get_tm_list(dataset_name):
    special_topics = {'lyrics': 12, 'seven_categories': 14, 'emails': 8, 'ag_news': 8, 'bbc_news': 10,
                      '20_newsgroups': 20, 'reuters': 10}
    distilroberta_category_num = {"20_newsgroups": 20, "seven_categories": 7, "lyrics": 4}
    mpnet_category_num = {"20_newsgroups": 20, "lyrics": 4, "seven_categories": 7}

    topic_num = special_topics[dataset_name]
    distilroberta_topic_num = distilroberta_category_num[dataset_name]
    mpnet_topic_num = mpnet_category_num[dataset_name]
    # list of topic models
    tm_list = ['bow',  # bow
               'tfidf',  # tfidf
               f'lda_n_topics_{topic_num}_alpha_auto_iterations_1000_topic_threshold_0.0',  # lda
               f'lda_linear_combined_n_topics_{topic_num}_alpha_auto_iterations_1000_topic_threshold_0.0',
               # lda_linear_combined
               f'lsi_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100',  # lsi
               f'lsi_tfidf_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100',  # lsi_tfidf
               f'lsi_linear_combined_tfidf_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100',
               # lsi_linear_combined_tfidf
               f'lsi_linear_combined_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100',
               # lsi_linear_combined
               f'nmf_n_topics_{topic_num}',  # nmf
               f'nmf_tfidf_n_topics_{topic_num}',  # nmf_tfidf
               f'nmf_linear_combined_n_topics_{topic_num}',  # nmf_linear_combined
               f'nmf_linear_combined_tfidf_n_topics_{topic_num}',  # nmf_linear_combined_tfidf
               f'bert_all_distilroberta_v1_n_categories_{distilroberta_topic_num}',
               f'bert_all_mpnet_base_v2_n_categories_{mpnet_topic_num}',
               f'doc2vec_n_topics_{topic_num}'
               ]
    return tm_list


def TM_short(TM, dataset_name):
    special_topics = {'lyrics': 12, 'seven_categories': 14, 'emails': 8, 'ag_news': 8, 'bbc_news': 10,
                      '20_newsgroups': 20, 'reuters': 10}
    topic_num = special_topics[dataset_name]
    if TM in ['bow', 'tfidf']:
        return 'BOW'
    elif TM in [f'lda_n_topics_{topic_num}_alpha_auto_iterations_1000_topic_threshold_0.0',
                f'lda_linear_combined_n_topics_{topic_num}_alpha_auto_iterations_1000_topic_threshold_0.0']:
        return 'LDA'
    elif TM in [f'lsi_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100',
                f'lsi_tfidf_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100',
                f'lsi_linear_combined_tfidf_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100',
                f'lsi_linear_combined_n_topics_{topic_num}_decay_1.0_onepass_True_power_iters_2_extra_samples_100']:
        return 'LSI'
    elif TM in [f'nmf_n_topics_{topic_num}', f'nmf_tfidf_n_topics_{topic_num}',
                f'nmf_linear_combined_n_topics_{topic_num}', f'nmf_linear_combined_tfidf_n_topics_{topic_num}']:
        return 'NMF'
    else:
        return TM


def get_configs(umap_parameter_lists, tsne_parameter_lists, som_parameter_lists, mds_parameter_lists):
    configs = dict()
    configs["mds"] = find_successor_tuples(mds_parameter_lists["n_iter"])
    configs = get_two_stage_parameter_pairs("som", som_parameter_lists, configs)
    configs = get_two_stage_parameter_pairs("umap", umap_parameter_lists, configs)
    for i, (parameter_name, parameter_list) in enumerate(tsne_parameter_lists.items()):
        config_name = "tsne" + str(i)
        configs[config_name] = ([], parameter_name)
        succesor_tuples = find_successor_tuples(tsne_parameter_lists[parameter_name])
        other_keys = [key for key in tsne_parameter_lists.keys() if key != parameter_name]
        other_values1 = tsne_parameter_lists[other_keys[0]]
        other_values2 = tsne_parameter_lists[other_keys[1]]

        for other_value1 in other_values1:
            for other_value2 in other_values2:
                for tuple in succesor_tuples:
                    new_entry = {other_keys[0]: other_value1, other_keys[1]: other_value2,
                                 parameter_name + "_pair": tuple}
                    configs[config_name][0].append(new_entry)
    return configs


def get_mds_list_names(tm, config, dataset_name, metric="cosine"):
    list_names = []
    for n_iter1, n_iter2 in config:
        layout1 = "mds_" + dataset_name + "_" + tm + "_True_" + metric + "_" + str(n_iter1)
        layout2 = "mds_" + dataset_name + "_" + tm + "_True_" + metric + "_" + str(n_iter2)
        list_names.append((layout1, layout2))
    return list_names


def get_tsne_list_names(tm, config, dataset_name, metric="cosine", paired_attribute=""):
    list_names = []

    if paired_attribute == "perplexity":
        for entry in config:
            perplexity_1, perplexity_2 = entry[paired_attribute + "_pair"]
            n_iter = entry["n_iter"]
            learning_rate = entry["learning_rate"]
            layout1 = "tsne_" + dataset_name + "_" + tm + "_" + str(perplexity_1) + "_12.0_" + str(
                learning_rate) + "_" + str(n_iter) + "_random_barnes_hut_0.5_" + metric
            layout2 = "tsne_" + dataset_name + "_" + tm + "_" + str(perplexity_2) + "_12.0_" + str(
                learning_rate) + "_" + str(n_iter) + "_random_barnes_hut_0.5_" + metric
            list_names.append((layout1, layout2))
    elif paired_attribute == "n_iter":
        for entry in config:
            perplexity = entry["perplexity"]
            n_iter_1, n_iter_2 = entry[paired_attribute + "_pair"]
            learning_rate = entry["learning_rate"]
            layout1 = "tsne_" + dataset_name + "_" + tm + "_" + str(perplexity) + "_12.0_" + str(
                learning_rate) + "_" + str(n_iter_1) + "_random_barnes_hut_0.5_" + metric
            layout2 = "tsne_" + dataset_name + "_" + tm + "_" + str(perplexity) + "_12.0_" + str(
                learning_rate) + "_" + str(n_iter_2) + "_random_barnes_hut_0.5_" + metric
            list_names.append((layout1, layout2))
    elif paired_attribute == "learning_rate":
        for entry in config:
            perplexity = entry["perplexity"]
            n_iter = entry["n_iter"]
            learning_rate_1, learning_rate_2 = entry[paired_attribute + "_pair"]
            layout1 = "tsne_" + dataset_name + "_" + tm + "_" + str(perplexity) + "_12.0_" + str(
                learning_rate_1) + "_" + str(n_iter) + "_random_barnes_hut_0.5_" + metric
            layout2 = "tsne_" + dataset_name + "_" + tm + "_" + str(perplexity) + "_12.0_" + str(
                learning_rate_2) + "_" + str(n_iter) + "_random_barnes_hut_0.5_" + metric
            list_names.append((layout1, layout2))
    return list_names


def get_umap_list_names(tm, config, dataset_name, metric="cosine", paired_attribute=""):
    list_names = []
    if paired_attribute == "n_neighbor":
        for entry in config:
            n_neighbor_1, n_neighbor_2 = entry[paired_attribute + "_pair"]
            min_dist = entry["min_dist"]
            layout1 = "umap_" + dataset_name + "_" + tm + "_" + str(n_neighbor_1) + "_" + str(min_dist) + "_" + metric
            layout2 = "umap_" + dataset_name + "_" + tm + "_" + str(n_neighbor_2) + "_" + str(min_dist) + "_" + metric
            list_names.append((layout1, layout2))
    elif paired_attribute == "min_dist":
        for entry in config:
            n_neighbor = entry["n_neighbor"]
            min_dist_1, min_dist_2 = entry[paired_attribute + "_pair"]
            layout1 = "umap_" + dataset_name + "_" + tm + "_" + str(n_neighbor) + "_" + str(min_dist_1) + "_" + metric
            layout2 = "umap_" + dataset_name + "_" + tm + "_" + str(n_neighbor) + "_" + str(min_dist_2) + "_" + metric
            list_names.append((layout1, layout2))
    return list_names


def get_som_list_names(tm, config, dataset_name, metric="cosine", paired_attribute=""):
    list_names = []
    if paired_attribute == "n":
        for entry in config:
            n_1, n_2 = entry[paired_attribute + "_pair"]
            m = entry["m"]
            layout1 = "som_" + dataset_name + "_" + tm + "_" + str(n_1) + "_" + str(m)
            layout2 = "som_" + dataset_name + "_" + tm + "_" + str(n_2) + "_" + str(m)
            list_names.append((layout1, layout2))
    elif paired_attribute == "m":
        for entry in config:
            m_1, m_2 = entry[paired_attribute + "_pair"]
            n = entry["n"]
            layout1 = "som_" + dataset_name + "_" + tm + "_" + str(n) + "_" + str(m_1)
            layout2 = "som_" + dataset_name + "_" + tm + "_" + str(n) + "_" + str(m_2)
            list_names.append((layout1, layout2))

    return list_names


def get_filenames_single_config(tm, config, key, dataset_name):
    list_names = []
    if "lda" in tm:
        metric = "jensenshannon"
    else:
        metric = "cosine"

    if "mds" in key:
        list_names = get_mds_list_names(tm, config, dataset_name, metric=metric)
    elif "tsne" in key:
        list_names = get_tsne_list_names(tm, config[0], dataset_name, metric=metric, paired_attribute=config[1])
    elif "umap" in key:
        list_names = get_umap_list_names(tm, config[0], dataset_name, paired_attribute=config[1], metric=metric)
    elif "som" in key:
        list_names = get_som_list_names(tm, config[0], dataset_name, paired_attribute=config[1])

    return list_names


def collect_file_path_pairs(dataset_name, config, key):
    tm_config_layouts_list_pair = []
    for tm in tqdm(get_tm_list(dataset_name)):
        layouts_list = get_filenames_single_config(tm, config, key, dataset_name)
        tm_config_layouts_list_pair.append((key, tm, layouts_list))

    return tm_config_layouts_list_pair


def get_processes(tm_config_layouts_list_pair, dataset_path, dataset_name):
    tm_config_process_key_dict_list = []
    manager = Manager()
    return_dict = manager.dict()
    missing_pairs = 0

    for config, tm, layouts_list in tm_config_layouts_list_pair:
        for (layout1_path_partial, layout2_path_partial) in layouts_list:
            layout1_path = ""
            for file in os.listdir(dataset_path):
                if layout1_path_partial in file and file.endswith(".npy") and not file.endswith("y.npy"):
                    layout1_path = os.path.join(dataset_path, file)
                    break
            layout2_path = ""
            for file in os.listdir(dataset_path):
                if layout2_path_partial in file and file.endswith(".npy") and not file.endswith("y.npy"):
                    layout2_path = os.path.join(dataset_path, file)
                    break

            if layout1_path == "" or layout2_path == "":
                print(
                    "Couldn't find original file names for pair: " + str((layout1_path_partial, layout2_path_partial)))
                missing_pairs += 1
                continue
            y_file_names = get_y_paths(dataset_path, constant_y=True)
            y = np.load(y_file_names[0])

            key = (layout1_path, layout2_path)
            short_tm_name = TM_short(tm, dataset_name)
            tm_config_process_key_dict_list.append((short_tm_name, config, Process(target=get_all_metrics_stability,
                                                                                   args=(layout1_path, layout2_path, y,
                                                                                         return_dict, "euclidean", True)),
                                                    key,
                                                    return_dict))

    print("Overall missing pairs: " + str(missing_pairs))
    return tm_config_process_key_dict_list


def process_results(tm_config_process_key_dict_list, n_jobs, df_name, overall_results):
    if os.path.isfile(df_name):
        df = pd.read_csv(df_name)
    else:
        df = None
    running_jobs = []
    for i, (short_tm_name, config, process, key, return_dict) in enumerate(tqdm(tm_config_process_key_dict_list)):
        if (df is not None and short_tm_name in df["short_tm_name"].values.tolist() and
                config in df["config"].values.tolist()
                and list(key) in df[["file_name_1", "file_name_2"]].values.tolist()):
            continue

        running_jobs.append((process, key, return_dict))
        process.start()

        # First condition for last batch that couldn't get full
        if i == len(tm_config_process_key_dict_list) - 1 or len(running_jobs) == n_jobs:
            for running_process, running_key, running_return_dict in running_jobs:
                running_process.join()
                cur_results = running_return_dict[running_key]
                new_entry = [config, short_tm_name]
                new_entry.extend(cur_results)
                overall_results.append(new_entry)
                running_process.close()
            write_partial_results(df_name, overall_results)
            running_jobs = []

    write_partial_results(df_name, overall_results)


def main():
    dataset_path, n_jobs = get_arguments()

    available_datasets = ["20_newsgroups", "seven_categories", "lyrics", "reuters", "bbc_news", "emails"]
    dataset_name = ""
    for dataset in available_datasets:
        if dataset in dataset_path:
            dataset_name = dataset

    # parameters of MDS
    mds_n_iter = [100, 150, 200, 250, 300]

    # parameters of SOM
    som_n = [5, 10, 15, 20, 25, 30]
    som_m = [5, 10, 15, 20, 25, 30]

    # parameters of UMAP
    umap_min_dist = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
    umap_n_neighbors = [2, 5, 10, 20, 50, 100, 200]

    # parameters of tSNE
    tsne_perplexity = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0]
    tsne_learning_rate = [10.0, 28.0, 129.0, 359.0, 1000.0]
    tsne_n_iter = [1000, 2500, 5000, 10000]

    configs = get_configs(umap_parameter_lists={"min_dist": umap_min_dist, "n_neighbor": umap_n_neighbors},
                          tsne_parameter_lists={"perplexity": tsne_perplexity, "learning_rate": tsne_learning_rate,
                                                "n_iter": tsne_n_iter},
                          som_parameter_lists={"n": som_n, "m": som_m},
                          mds_parameter_lists={"n_iter": mds_n_iter})

    config_layouts_pair = []
    tm_config_process_key_dict_lists = []
    for key, config in configs.items():
        config_layouts_pair.append(collect_file_path_pairs(dataset_name=dataset_name, config=config, key=key))

    for entry in config_layouts_pair:
        tm_config_process_key_dict_lists.append(get_processes(entry, dataset_path, dataset_name))

    if n_jobs != -1:
        n_jobs = min(n_jobs, cpu_count())
    else:
        n_jobs = cpu_count()

    time.sleep(3)
    df_name = "results_experiment_2_" + dataset_name + ".csv"
    if os.path.isfile(df_name):
        print("Found existing result file. I will load and continue where appropriate")
        df = pd.read_csv(df_name)
        overall_results = df.values.tolist()
    else:
        overall_results = []

    print("Got all processes. I will now begin processing results with " + str(n_jobs) + " parallel jobs")
    for tm_config_process_key_list in tm_config_process_key_dict_lists:
        process_results(tm_config_process_key_list, n_jobs, df_name, overall_results)


def write_partial_results(df_name, overall_results):
    df_new = pd.DataFrame(overall_results, columns=["config", "short_tm_name", "spearman_correlation",
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
