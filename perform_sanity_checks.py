import os
from argparse import ArgumentParser
import pandas as pd


def get_arguments():
    parser = ArgumentParser(description="This script executes experiment 1 analysing all metric correlations")
    parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                        help="The path to the results directory of the dataset for which the correlations shall "
                             "be analysed.")
    args = parser.parse_args()
    return args.dataset_path


def sanity_checks_experiment_2(cur_path, bonus_factor=1):
    df = pd.read_csv(cur_path)
    max_layouts = {"mds": 4, "som0": 30, "som1": 30, "umap0": 35, "umap1": 36, "tsne0": 100, "tsne1": 96, "tsne2": 90}
    overall_present = 0
    overall_max = 0

    required_configurations = ["mds", "som0", "som1", "umap0", "umap1", "tsne0", "tsne1", "tsne2"]
    if all([el in df["config"].values.tolist() for el in required_configurations]):
        print("all configs present!")
    else:
        print("a config is missing")

    required_short_tm_names = ["LDA", "LSI", "NMF", "bert_all_distilroberta_v1", "bert_all_mpnet_base_v2", "BOW",
                               "doc2vec"]
    factor_tm = {"LDA": 2, "LSI": 4, "NMF": 4, "bert_all_distilroberta_v1": 1, "bert_all_mpnet_base_v2": 1,
                 "BOW": 2, "doc2vec": 1}
    for config in required_configurations:
        config_df = df.loc[df["config"] == config]
        print(f'Length of config df: {str(len(config_df))}')
        for tm in required_short_tm_names:
            config_tm_df = config_df.iloc[[(tm in el) for el in config_df["short_tm_name"]]]
            overall_present += len(config_tm_df)
            overall_max += max_layouts[config] * factor_tm[tm] * bonus_factor
            print(f'Length of values for config df combination {config} and {tm}: {str(len(config_tm_df))} of possible'
                  f' maximal {str(max_layouts[config] * factor_tm[tm] * bonus_factor)}. This equals a share of'
                  f' {str((len(config_tm_df) / (max_layouts[config] * factor_tm[tm] * bonus_factor)) * 100)}%')

    print(f"Overall coverage: {(overall_present / overall_max) * 100}%")
    with open("report_experiment_2.txt", 'a+') as report_file:
        report_file.write(cur_path + "\n")
        report_file.write(f"Overall coverage: {(overall_present / overall_max) * 100}%")
        report_file.write("\n\n")


def sanity_checks_experiment_3(cur_path, bonus_factor=1):
    df = pd.read_csv(cur_path)
    required_configs = ["mds", "som", "umap", "tsne"]
    required_tms = ["lda", "lda_linear_combined", "lsi", "lsi_linear_combined", "lsi_tfidf",
                    "lsi_linear_combined_tfidf",
                    "nmf", "nmf_linear_combined", "nmf_linear_combined_tfidf", "nmf_tfidf",
                    "bert_all_distilroberta_v1", "bert_all_mpnet_base_v2", "bow", "tfidf",
                    "doc2vec"]

    max_layouts = {"mds": 5, "som": 36, "umap": 42, "tsne": 144}
    overall_present = 0
    overall_max = 0

    for config in required_configs:
        config_df = df.loc[df["config"] == config]
        print(f'Length of config df: {str(len(config_df))}')
        for tm in required_tms:
            config_tm_df = config_df.loc[df["TM"] == tm]
            overall_present += len(config_tm_df)
            overall_max += max_layouts[config] * bonus_factor
            print(f'Length of values for config df combination {config} and {tm}: {str(len(config_tm_df))} of possible'
                  f' maximal {str(max_layouts[config] * bonus_factor)}. This equals a share of'
                  f' {str((len(config_tm_df) / (max_layouts[config] * bonus_factor)) * 100)}%')

    print(f"Overall coverage: {(overall_present / overall_max) * 100}%")
    with open("report_experiment_3.txt", 'a+') as report_file:
        report_file.write(cur_path + "\n")
        report_file.write(f"Overall coverage: {(overall_present / overall_max) * 100}%")
        report_file.write("\n\n")


def sanity_checks_experiment_4(cur_path, bonus_factor=1):
    df = pd.read_csv(cur_path)
    required_configs = ["mds", "som", "umap", "tsne"]
    required_tms = ["lda", "lda_linear_combined", "lsi", "lsi_linear_combined", "lsi_tfidf",
                    "lsi_linear_combined_tfidf", "nmf",
                    "nmf_linear_combined", "nmf_linear_combined_tfidf", "nmf_tfidf",
                    "bert_all_distilroberta_v1", "bert_all_mpnet_base_v2", "bow", "tfidf",
                    "doc2vec"]
    overall_present = 0
    overall_max = 0
    max_layouts = {"mds": 10, "som": 72, "umap": 84, "tsne": 288}

    for config in required_configs:
        config_df = df.loc[df["config"] == config]
        print(f'Length of config df: {str(len(config_df))}')
        for tm in required_tms:
            config_tm_df = config_df.loc[df["TM"] == tm]
            overall_present += len(config_tm_df)
            overall_max += max_layouts[config] * bonus_factor
            print(f'Length of values for config df combination {config} and {tm}: {str(len(config_tm_df))} of possible'
                  f' maximal {str(max_layouts[config] * bonus_factor)}. This equals a share of'
                  f' {str((len(config_tm_df) / (max_layouts[config] * bonus_factor)) * 100)}%')

    print(f"Overall coverage: {(overall_present / overall_max) * 100}%")
    with open("report_experiment_4.txt", 'a+') as report_file:
        report_file.write(cur_path + "\n")
        report_file.write(f"Overall coverage: {(overall_present / overall_max) * 100}%")
        report_file.write("\n\n")


def main():
    results_file_path = get_arguments()
    df2 = None
    df3 = None
    df4 = None

    for res_file in os.listdir(results_file_path):
        cur_path = os.path.join(results_file_path, res_file)

        if "overall" in res_file:
            continue

        elif "results_experiment_2" in res_file:
            sanity_checks_experiment_2(cur_path)
            if df2 is None:
                df2 = pd.read_csv(cur_path)
            else:
                df2 = df2.append(pd.read_csv(cur_path))

        elif "results_experiment_3" in res_file:
            sanity_checks_experiment_3(cur_path)
            if df3 is None:
                df3 = pd.read_csv(cur_path)
            else:
                df3 = df3.append(pd.read_csv(cur_path))

        elif "results_experiment_4" in res_file:
            sanity_checks_experiment_4(cur_path)
            if df4 is None:
                df4 = pd.read_csv(cur_path)
            else:
                df4 = df4.append(pd.read_csv(cur_path))

    df2.to_csv(os.path.join(results_file_path, "overall_results_2.csv"), index=False)
    sanity_checks_experiment_2(os.path.join(results_file_path, "overall_results_2.csv"), bonus_factor=3)

    df3.to_csv(os.path.join(results_file_path, "overall_results_3.csv"), index=False)
    sanity_checks_experiment_3(os.path.join(results_file_path, "overall_results_3.csv"), bonus_factor=3)

    df4.to_csv(os.path.join(results_file_path, "overall_results_4.csv"), index=False)
    sanity_checks_experiment_4(os.path.join(results_file_path, "overall_results_4.csv"), bonus_factor=3)


if __name__ == "__main__":
    main()
