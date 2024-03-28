from argparse import ArgumentParser, ArgumentTypeError
import itertools
import os
import shutil
import random

from config import get_global_random_seed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = ArgumentParser(description="This script generates a csv with all call strings and res_files")
    parser.add_argument('--res_dir_path', dest='res_path', type=str, default="./",
                        help="If the cwd of the script differs from the location where the call to the main.py"
                             " script is invoked, "
                             "we have to search for the file in another directory. This parameter shall point "
                             "to this location (cwd in context of the main.py script. Defaults to './'")
    parser.add_argument('--only_som', dest='only_som', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the SOM shall be processed or all other projection techniques.")
    parser.add_argument('--only_tsne', dest='only_tsne', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the t-SNE shall be processed or all other projection techniques.")
    parser.add_argument('--get_all', dest='get_all', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the SOM shall be processed or all other projection techniques.")

    parser.add_argument('--disable_model_training', dest='dmt', type=str2bool, default=False, const=True, nargs="?",
                        help="If set the program will return after preprocessing the requested dataset and "
                             "creating the topic models. No evaluation is undertaken in this case.")
    parser.add_argument('--request_default', dest='rd', type=str2bool, default=False, const=True, nargs="?",
                        help="If set only the default parameters of a DR will be evaluated")
    parser.add_argument('--force_reeval', dest='fr', type=str2bool, default=False, const=True, nargs="?",
                        help="If set only the default parameters of a DR will be evaluated")

    parser.add_argument('--big_datasets', dest='bd', type=str2bool, default=False, const=True, nargs="?",
                        help="If set, get parameters for big datasets")
    parser.add_argument('--partial_topic_reeval', dest='ptr', type=str2bool, default=False, const=True, nargs="?",
                        help="If set, a reevaluation of the topic model will be forced even if the full result is "
                             "present."
                             "If set, the program will still return early after evaluating the topic model and"
                             "therefore will not reevaluate the dimensionality reduction techniques")
    parser.add_argument('--permutation_mode', dest='permutation_mode', type=str, default="full",
                        help="Specifies which permutation is requested. Valid values are 'full', 'use_part', "
                             "'shuffle_rows', 'scale_rows', 'scale_columns', 'drop_tm', 'jitter', 'second_run'"
                             " and 'no'."
                             " Defaults to 'full' meaning that all "
                             "combinations of permutations are requested. This will lead to a long run time.")
    args = parser.parse_args()
    return (args.res_path, args.only_som, args.get_all, args.dmt, args.bd, args.only_tsne, args.rd, args.fr, args.ptr,
            args.permutation_mode)


def get_parameter_list_use_part(parameter_list, mode_key, amounts):
    new_list = []
    for amount in amounts:
        for parameter_string in parameter_list:
            parameter_string += " --permute_mode " + str(mode_key) + " --permute_ratio " + str(amount)
            new_list.append(parameter_string)
    return new_list


def get_parameter_list_shuffle_rows(parameter_list, mode_key, times):
    new_list = []
    for time in times:
        for parameter_string in parameter_list:
            parameter_string += " --permute_mode " + str(mode_key) + " --permute_shuffle_times " + str(time)
            new_list.append(parameter_string)
    return new_list


def get_parameter_list_scale_rows(parameter_list, shuffle_times):
    new_list = []
    for time in shuffle_times:
        for parameter_string in parameter_list:
            parameter_string += " --scale_rows_shuffle_times_bow " + str(time)
            new_list.append(parameter_string)
    return new_list


def get_parameter_list_scale_columns(parameter_list, shuffle_times):
    new_list = []
    for time in shuffle_times:
        for parameter_string in parameter_list:
            parameter_string += " --scale_columns_shuffle_times_bow " + str(time)
            new_list.append(parameter_string)
    return new_list


def get_parameter_list_drop_columns(parameter_list, drop_ratios, drop_ratio_tm_shuffle_times):
    new_list = []
    for ratio in drop_ratios:
        for time in drop_ratio_tm_shuffle_times:
            for parameter_string in parameter_list:
                parameter_string += " --drop_ratio_tm " + str(ratio) + " --drop_ratio_tm_shuffle_times " + str(time)
                new_list.append(parameter_string)
    return new_list


def get_parameter_list_jitter(parameter_list, jitter_amounts):
    parameter_list_jitter = []
    for jitter_amount in jitter_amounts:
        parameter_list_jitter.extend([parameter_string + " --jitter_amount " + str(jitter_amount) for parameter_string
                                      in parameter_list])

    return parameter_list_jitter


def main():
    res_dir_path, only_som, get_all_parameters, disable_model_training, request_big_datasets, only_tsne, \
        request_default, force_reeval, partial_topic_reeval, permutation_mode = get_arguments()

    if request_big_datasets:
        available_datasets = [("github_projects", "main.py")]
    else:
        available_datasets = [
            ("20_newsgroups", "main.py"),
            ("reuters", "main.py"),
            ("emails", "main.py"),
            ("seven_categories", "main.py"),
            ("bbc_news", "main.py"),
            ("lyrics", "main.py"),
        ]

    available_tms = [
        "bow",
        "tfidf",
        "lda",
        "lda_linear_combined",
        "lsi",
        "lsi_linear_combined",
        "lsi_tfidf",
        "lsi_tfidf_linear_combined",
        "nmf",
        "nmf_linear_combined",
        "nmf_tfidf",
        "nmf_tfidf_linear_combined",
        "bert",
        "doc2vec"
    ]

    # Special topics will always take priority over default topic num (equals number categories) or n_topics list below
    # special_topics = {}
    special_topics = {'lyrics': 12, 'seven_categories': 14, 'emails': 8, 'ag_news': 8, 'bbc_news': 10,
                      '20_newsgroups': 20, 'reuters': 10}

    n_topics_datasets = {}
    # n_topics_datasets = {"20_newsgroups": [20, 25, 30, 35, 40], "reuters": [65, 82, 98, 114, 130],
    #                     "emails": [8, 10, 12, 14, 16], "seven_categories": [14, 17, 21, 24, 28],
    #                     "github_projects": [16, 20, 24, 28, 32]}

    # perplexity_tsne = [5, 15, 25, 35, 45, 55]
    perplexity_tsne = []
    # n_iter_tsne = [1000, 2500, 5000, 10000]
    n_iter_tsne = []
    # learning_rate_tsne = ["auto", 10, 1000, 28, 129, 359]
    learning_rate_tsne = []

    # n_neighbors_umap = [2, 5, 10, 20, 50, 100, 200]
    n_neighbors_umap = []
    # min_dist_umap = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
    min_dist_umap = []
    # runs_mds = [100, 150, 200, 250, 300]
    runs_mds = []
    n_som = [5, 10, 15, 20, 25, 30]
    # n_som = []
    m_som = [5, 10, 15, 20, 25, 30]
    # m_som = []

    results_base_path = "results"
    if permutation_mode == "full":
        permutation_mode_list = ["use_part", "shuffle_rows"]
        # permutation_mode_list = ["shuffle_rows"]
        permute_ratio = [5, 6, 7, 8, 9, 10]
        # permute_ratio = []
        permute_shuffle_times = [1, 2, 3, 4, 5]
        scale_rows_shuffle_times_bow = [1, 2, 3, 4, 5]
        scale_columns_shuffle_times_bow = [1, 2, 3, 4, 5]
        drop_ratio_tm = [5, 10, 15]
        drop_ratio_tm_shuffle_times = [1, 2, 3, 4, 5]
        jitter_amounts = [0.1, 0.2]
    elif permutation_mode == "use_part":
        permutation_mode_list = ["use_part"]
        permute_ratio = [5, 6, 7, 8, 9, 10]
        permute_shuffle_times = []
        scale_rows_shuffle_times_bow = []
        scale_columns_shuffle_times_bow = []
        drop_ratio_tm = []
        drop_ratio_tm_shuffle_times = []
        jitter_amounts = []
    elif permutation_mode == "shuffle_rows":
        permutation_mode_list = ["shuffle_rows"]
        permute_ratio = []
        permute_shuffle_times = [1, 2, 3, 4, 5]
        scale_rows_shuffle_times_bow = []
        scale_columns_shuffle_times_bow = []
        drop_ratio_tm = []
        drop_ratio_tm_shuffle_times = []
        jitter_amounts = []
    elif permutation_mode == "scale_rows":
        permutation_mode_list = []
        permute_ratio = []
        permute_shuffle_times = []
        scale_rows_shuffle_times_bow = [1, 2, 3, 4, 5]
        scale_columns_shuffle_times_bow = []
        drop_ratio_tm = []
        drop_ratio_tm_shuffle_times = []
        jitter_amounts = []
    elif permutation_mode == "scale_columns":
        permutation_mode_list = []
        permute_ratio = []
        permute_shuffle_times = []
        scale_rows_shuffle_times_bow = []
        scale_columns_shuffle_times_bow = [1, 2, 3, 4, 5]
        drop_ratio_tm = []
        drop_ratio_tm_shuffle_times = []
        jitter_amounts = []
    elif permutation_mode == "drop_tm":
        permutation_mode_list = []
        permute_ratio = []
        permute_shuffle_times = []
        scale_rows_shuffle_times_bow = []
        scale_columns_shuffle_times_bow = []
        drop_ratio_tm = [5, 10, 15]
        drop_ratio_tm_shuffle_times = [1, 2, 3, 4, 5]
        jitter_amounts = []
    elif permutation_mode == "jitter":
        permutation_mode_list = []
        permute_ratio = []
        permute_shuffle_times = []
        scale_rows_shuffle_times_bow = []
        scale_columns_shuffle_times_bow = []
        drop_ratio_tm = []
        drop_ratio_tm_shuffle_times = []
        jitter_amounts = [0.0, 0.1]
    elif permutation_mode == "second_run":
        permutation_mode_list = []
        permute_ratio = []
        permute_shuffle_times = []
        scale_rows_shuffle_times_bow = []
        scale_columns_shuffle_times_bow = []
        drop_ratio_tm = []
        drop_ratio_tm_shuffle_times = []
        jitter_amounts = []
        results_base_path = "results_second_run"
    elif permutation_mode == "no":
        permutation_mode_list = []
        permute_ratio = []
        permute_shuffle_times = []
        scale_rows_shuffle_times_bow = []
        scale_columns_shuffle_times_bow = []
        drop_ratio_tm = []
        drop_ratio_tm_shuffle_times = []
        jitter_amounts = []
    else:
        raise ValueError("Unknown permutation mode: " + permutation_mode)

    full_list_tsne = list(itertools.product(perplexity_tsne, n_iter_tsne, learning_rate_tsne))
    full_list_umap = list(itertools.product(n_neighbors_umap, min_dist_umap))
    full_list_som = list(itertools.product(n_som, m_som))
    original_main = "main.py"
    main_scripts = list(set([el[1] for el in available_datasets]))

    # lda_alphas = ["symmetric", "asymmetric", "auto"]
    lda_alphas = ["auto"]

    if request_default:
        full_list_tsne = [(30, 1000, 'auto')]
        full_list_umap = [(15, 0.1)]
        runs_mds = [300]
        full_list_som = [(10, 10)]

    if only_som:
        parameter_list = get_som_parameter_list(full_list_som)
    elif only_tsne:
        parameter_list = get_tsne_parameter_list(full_list_tsne)
    else:
        max_length = max(len(full_list_tsne), len(full_list_umap), len(runs_mds))
        parameter_list = ["main.py"] * max_length

        for i in range(len(full_list_tsne)):
            parameter_list[i] += " --perplexity_tsne " + str(full_list_tsne[i][0]) + " --n_iter_tsne " + \
                                 str(full_list_tsne[i][1]) + " --learning_rate " + str(full_list_tsne[i][2])

        for i in range(len(full_list_umap)):
            parameter_list[i] += " --n_neighbors_umap " + str(full_list_umap[i][0]) + " --min_dist_umap " + \
                                 str(full_list_umap[i][1])

        for i in range(len(runs_mds)):
            parameter_list[i] += " --max_iter_mds " + str(runs_mds[i])

        if get_all_parameters:
            parameter_list.extend(get_som_parameter_list(full_list_som))

    # random.seed(get_global_random_seed())
    # random.shuffle(parameter_list)

    parameter_lists_tmp = []
    for dataset, main_script in available_datasets:
        if dataset in special_topics.keys():
            parameter_lists_tmp.extend([parameters + " --dataset_name " + dataset + " --n_topics_lda "
                                        + str(special_topics[dataset]) + " --n_topics_lsi " +
                                        str(special_topics[dataset]) + " --n_topics_nmf " +
                                        str(special_topics[dataset]) for parameters in parameter_list])
        elif dataset in n_topics_datasets.keys():
            for el in n_topics_datasets[dataset]:
                parameter_lists_tmp.extend([parameters + " --dataset_name " + dataset + " --n_topics_lda "
                                            + str(el) + " --n_topics_lsi " +
                                            str(el) + " --n_topics_nmf " +
                                            str(el) for parameters in parameter_list])
        else:
            parameter_lists_tmp.extend([parameters + " --dataset_name " + dataset for parameters in parameter_list])

        parameter_lists_tmp = [parameter_string.replace(original_main, main_script) for parameter_string
                               in parameter_lists_tmp]

    if disable_model_training:
        disable_model_parameter_string_part = " --disable_model_training"
    else:
        disable_model_parameter_string_part = ""
    parameter_lists_tmp = [el + disable_model_parameter_string_part for el in parameter_lists_tmp]

    if force_reeval:
        force_reeval_string_part = " --force_reeval"
    else:
        force_reeval_string_part = ""
    parameter_lists_tmp = [el + force_reeval_string_part for el in parameter_lists_tmp]

    if partial_topic_reeval:
        force_topic_reeval_string_part = " --partial_topic_reeval"
    else:
        force_topic_reeval_string_part = ""
    parameter_lists_tmp = [el + force_topic_reeval_string_part for el in parameter_lists_tmp]

    parameter_list = parameter_lists_tmp

    if len(available_tms) > 0:
        parameter_lists_tmp = []
        for available_tm in available_tms:
            if available_tm == "lda":
                for lda_alpha in lda_alphas:
                    parameter_lists_tmp.extend([parameter_string + " --topic_model " + available_tm + " --alpha_lda " +
                                                lda_alpha for parameter_string
                                                in parameter_list])
            else:
                parameter_lists_tmp.extend([parameter_string + " --topic_model " + available_tm for parameter_string
                                            in parameter_list])
        parameter_list = parameter_lists_tmp

    parameter_new = []

    if "use_part" in permutation_mode_list:
        parameter_new.extend(get_parameter_list_use_part(parameter_list=parameter_list,
                                                         mode_key='use_part', amounts=permute_ratio))
    if "shuffle_rows" in permutation_mode_list:
        parameter_new.extend(get_parameter_list_shuffle_rows(parameter_list=parameter_list,
                                                             mode_key="shuffle_rows", times=permute_shuffle_times))
    parameter_new.extend(get_parameter_list_scale_rows(parameter_list=parameter_list,
                                                       shuffle_times=scale_rows_shuffle_times_bow))
    parameter_new.extend(get_parameter_list_scale_columns(parameter_list=parameter_list,
                                                          shuffle_times=scale_columns_shuffle_times_bow))
    parameter_new.extend(get_parameter_list_drop_columns(parameter_list=parameter_list, drop_ratios=drop_ratio_tm,
                                                         drop_ratio_tm_shuffle_times=drop_ratio_tm_shuffle_times))
    parameter_new.extend(get_parameter_list_jitter(parameter_list=parameter_list, jitter_amounts=jitter_amounts))

    if permutation_mode == "no" or permutation_mode == "full":
        parameter_list.extend(parameter_new)
    else:
        parameter_list = parameter_new

    for i, parameter_string in enumerate(parameter_list):
        dataset = parameter_string.split("dataset_name")[1].strip().split(" ")[0]

        for main_script in main_scripts:
            file_name = parameter_string.replace(main_script, "").replace("-", "")
        if dataset in special_topics.keys():
            file_name = file_name.replace("n_topics_lda " + str(special_topics[dataset]), "") \
                .replace("n_topics_lsi " + str(special_topics[dataset]), "").replace("_nmf", "")
        if disable_model_training:
            file_name = file_name.replace("disable_model_training", "")
        file_name = ' '.join(file_name.split())
        file_name = file_name.replace(" ", "_")
        file_name = '_'.join(file_name.split("_"))

        file_name = str(os.path.join(results_base_path, dataset, "results_"
                                     + file_name + ".csv"))
        if not os.path.isfile("." + os.path.sep + file_name) or get_all_parameters:
            print(
                res_dir_path + file_name + "," + parameter_string + " --res_file_name " + "." + os.path.sep + file_name)
        else:
            for dataset_name in available_datasets:
                os.makedirs(os.path.join("res_files_only", results_base_path, dataset_name[0]), exist_ok=True)
            shutil.copyfile(("." + os.path.sep + file_name), os.path.join("res_files_only", file_name))

    print(len(parameter_list))


def get_som_parameter_list(full_list_som):
    parameter_list = ["main.py"] * len(full_list_som)
    for i in range(len(full_list_som)):
        parameter_list[i] += " --n_som " + str(full_list_som[i][0]) + " --m_som " + str(full_list_som[i][1]) \
                             + " --only_som=True"
    return parameter_list


def get_tsne_parameter_list(full_list_tsne):
    parameter_list = ["main.py"] * len(full_list_tsne)
    for i in range(len(full_list_tsne)):
        parameter_list[i] += " --perplexity_tsne " + str(full_list_tsne[i][0]) + " --n_iter_tsne " + \
                             str(full_list_tsne[i][1]) + " --learning_rate " + str(full_list_tsne[i][2]) + \
                             " --only_tsne "
    return parameter_list


if __name__ == "__main__":
    main()
