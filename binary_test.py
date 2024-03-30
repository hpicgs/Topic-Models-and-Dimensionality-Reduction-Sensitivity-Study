from itertools import combinations

import pandas as pd
import os
from scipy.stats import binomtest

from tqdm import tqdm

datasets = ["lyrics", "20_newsgroups", "seven_categories"]


def binary_tests_stability_2(required_dr, required_tm, base_path, count_improvement_function):
    n_alpha = 0
    n_beta = 0
    n_gamma = 0
    n_samples = 0

    for dataset in datasets:
        real_path = base_path.replace("20_newsgroups", dataset)
        df = pd.read_csv(real_path)
        df = calculate_alpha_beta_gamma(df)

        n_alpha, n_beta, n_gamma, n_samples = count_improvement_function(df, n_alpha, n_beta, n_gamma, n_samples,
                                                                         required_dr, required_tm)

    print(f"Results for experiment {required_dr} and {required_tm}:")
    result = binomtest(n_alpha, n=n_samples, p=0.5, alternative='greater')
    print(f"p-value for alpha: {result.pvalue}")

    result = binomtest(n_beta, n=n_samples, p=0.5, alternative='greater')
    print(f"p-value for beta: {result.pvalue}")

    result = binomtest(n_gamma, n=n_samples, p=0.5, alternative='greater')
    print(f"p-value for gamma: {result.pvalue}")


def count_improvements_tfidf(df, n_alpha, n_beta, n_gamma, n_samples, required_dr, required_tm):
    columns = df.columns.tolist()
    values = df.values.tolist()
    alpha_index = columns.index('alpha')
    beta_index = columns.index('beta')
    gamma_index = columns.index('gamma')
    file_names_1 = df["file_name_1"].tolist()
    file_names_2 = df["file_name_2"].tolist()
    visited_indices = []
    combined_file_names = [(file_names_1[i], file_names_2[i]) for i in
                           range(len(file_names_1))]
    for i, file_pair in enumerate(tqdm(combined_file_names)):
        if i in visited_indices:
            continue

        if "tfidf" in file_pair[0] or required_dr not in file_pair[0]:
            continue
        elif "bow" in file_pair[0] and required_tm == "bow":
            searched_pair = (file_pair[0].replace("bow", "tfidf"), file_pair[1].replace("bow", "tfidf"))

        elif "lsi_linear_combined" in file_pair[0] and required_tm == "lsi":
            searched_pair = (file_pair[0].replace("lsi_linear_combined", "lsi_linear_combined_tfidf"),
                             file_pair[1].replace("lsi_linear_combined", "lsi_linear_combined_tfidf"))


        elif "lsi" in file_pair[0] and required_tm == "lsi":
            searched_pair = (file_pair[0].replace("lsi", "lsi_tfidf"), file_pair[1].replace("lsi", "lsi_tfidf"))

        elif "nmf_linear_combined" in file_pair[0] and required_tm == "nmf":
            searched_pair = (file_pair[0].replace("nmf_linear_combined", "nmf_linear_combined_tfidf"),
                             file_pair[1].replace("nmf_linear_combined", "nmf_linear_combined_tfidf"))

        elif "nmf" in file_pair[0] and required_tm == "nmf":
            searched_pair = (file_pair[0].replace("nmf", "nmf_tfidf"), file_pair[1].replace("nmf", "nmf_tfidf"))
        else:
            continue

        try:
            index = combined_file_names.index(searched_pair)
            if combined_file_names.count(searched_pair) > 1:
                print(f"Attention!!! Multiple entries: {combined_file_names.count(searched_pair)}")
                test = [(i, el) for i, el in enumerate(combined_file_names) if el == searched_pair]
                continue
        except ValueError:
            continue

        if values[index][alpha_index] > values[i][alpha_index]:
            n_alpha += 1
        if values[index][beta_index] > values[i][beta_index]:
            n_beta += 1
        if values[index][gamma_index] > values[i][gamma_index]:
            n_gamma += 1

        n_samples += 1

        visited_indices.append(index)
    return n_alpha, n_beta, n_gamma, n_samples


def count_improvements_linear_combined(df, n_alpha, n_beta, n_gamma, n_samples, required_dr, required_tm):
    columns = df.columns.tolist()
    values = df.values.tolist()
    alpha_index = columns.index('alpha')
    beta_index = columns.index('beta')
    gamma_index = columns.index('gamma')
    file_names_1 = df["file_name_1"].tolist()
    file_names_2 = df["file_name_2"].tolist()
    visited_indices = []
    combined_file_names = [(file_names_1[i], file_names_2[i]) for i in
                           range(len(file_names_1))]
    for i, file_pair in enumerate(tqdm(combined_file_names)):
        if i in visited_indices:
            continue

        if "linear_combined" in file_pair[0] or required_dr not in file_pair[0]:
            continue
        elif "lda" in file_pair[0] and required_tm == "lda":
            searched_pair = (
                file_pair[0].replace("lda", "lda_linear_combined"), file_pair[1].replace("lda", "lda_linear_combined"))

        elif "lsi" in file_pair[0] and "tfidf" in file_pair[0] and required_tm == "lsi":
            searched_pair = (file_pair[0].replace("lsi_tfidf", "lsi_linear_combined_tfidf"),
                             file_pair[1].replace("lsi_tfidf", "lsi_linear_combined_tfidf"))


        elif "lsi" in file_pair[0] and required_tm == "lsi":
            searched_pair = (file_pair[0].replace("lsi_n", "lsi_linear_combined_n"),
                             file_pair[1].replace("lsi", "lsi_linear_combined_n"))

        elif "nmf" in file_pair[0] and "tfidf" in file_pair[0] and required_tm == "nmf":
            searched_pair = (file_pair[0].replace("nmf_tfidf", "nmf_linear_combined_tfidf"),
                             file_pair[1].replace("nmf_tfidf", "nmf_linear_combined_tfidf"))

        elif "nmf" in file_pair[0] and required_tm == "nmf":
            searched_pair = (file_pair[0].replace("nmf_n", "nmf_tfidf_n"), file_pair[1].replace("nmf_n", "nmf_tfidf_n"))
        else:
            continue

        try:
            index = combined_file_names.index(searched_pair)
            if combined_file_names.count(searched_pair) > 1:
                print(f"Attention!!! Multiple entries: {combined_file_names.count(searched_pair)}")
                continue
        except ValueError:
            continue

        if values[index][alpha_index] > values[i][alpha_index]:
            n_alpha += 1
        if values[index][beta_index] > values[i][beta_index]:
            n_beta += 1
        if values[index][gamma_index] > values[i][gamma_index]:
            n_gamma += 1

        n_samples += 1

        visited_indices.append(index)
    return n_alpha, n_beta, n_gamma, n_samples


def calculate_corrected_alpha_beta_gamma(df, data_set):
    df = calculate_alpha_beta_gamma(df)
    df_high = pd.read_csv(
        f"results_experiment_5_corpus_similarity/results_experiment_5_corpus_similarity_{data_set}.csv")
    df_high = calculate_alpha_beta_gamma(df_high)

    alpha_jitter010 = df_high['alpha'].tolist()[0]
    alpha_jitter025 = df_high['alpha'].tolist()[1]
    beta_jitter010 = df_high['beta'].tolist()[0]
    beta_jitter025 = df_high['beta'].tolist()[1]
    gamma_jitter010 = df_high['gamma'].tolist()[0]
    gamma_jitter025 = df_high['gamma'].tolist()[1]
    corrected_alphas = []
    corrected_betas = []
    corrected_gammas = []

    for i in range(len(df)):
        if df["jitter_1"][i] == 0.1 or df["jitter_2"][i] == 0.1:
            corrected_alphas.append(1 - abs(df["alpha"][i] - alpha_jitter010))
            corrected_betas.append(1 - abs(df["beta"][i] - beta_jitter010))
            corrected_gammas.append(1 - abs(df["gamma"][i] - gamma_jitter010))
        if df["jitter_1"][i] == 0.25 or df["jitter_2"][i] == 0.25:
            corrected_alphas.append(1 - abs(df["alpha"][i] - alpha_jitter025))
            corrected_betas.append(1 - abs(df["beta"][i] - beta_jitter025))
            corrected_gammas.append(1 - abs(df["gamma"][i] - gamma_jitter025))

    df["alpha"] = corrected_alphas
    df["beta"] = corrected_betas
    df["gamma"] = corrected_gammas
    return df


def binary_tests_stability_1(required_tm, required_dr, base_path, count_improvement_function):
    n_alpha = 0
    n_beta = 0
    n_gamma = 0
    n_samples = 0

    for dataset in datasets:
        real_path = base_path.replace("20_newsgroups", dataset)
        df = pd.read_csv(real_path)
        df = calculate_corrected_alpha_beta_gamma(df, dataset)
        n_alpha, n_beta, n_gamma, n_samples = count_improvement_function(df, n_alpha, n_beta, n_gamma, n_samples,
                                                                         required_dr, required_tm)

    print(f"Results for experiment {required_dr} and {required_tm}:")
    result = binomtest(n_alpha, n=n_samples, p=0.5, alternative='greater')
    print(f"p-value for alpha: {result.pvalue}")

    result = binomtest(n_beta, n=n_samples, p=0.5, alternative='greater')
    print(f"p-value for beta: {result.pvalue}")

    result = binomtest(n_gamma, n=n_samples, p=0.5, alternative='greater')
    print(f"p-value for gamma: {result.pvalue}")


def binary_tests_stability_3(required_tm, required_dr, base_path, count_improvement_function):
    binary_tests_stability_2(required_tm=required_tm, required_dr=required_dr,
                             base_path=base_path, count_improvement_function=count_improvement_function)


def main():
    base_path_1 = "results_experiment_jitter/results_experiment_4_20_newsgroups.csv"
    base_path_2 = "results_experiment_hyperparameters/results_experiment_2_20_newsgroups.csv"
    base_path_3 = "results_experiment_randomness/results_experiment_3_20_newsgroups.csv"
    tms = ["bow", "lsi", "nmf"]
    drs = ["mds", "som", "tsne", "umap"]

    for tm in tms:
        for dr in drs:
            binary_tests_stability_1(required_tm=tm, required_dr=dr, base_path=base_path_1,
                                     count_improvement_function=count_improvements_tfidf)
            binary_tests_stability_2(required_tm=tm, required_dr=dr, base_path=base_path_2,
                                     count_improvement_function=count_improvements_tfidf)
            binary_tests_stability_3(required_tm=tm, required_dr=dr, base_path=base_path_3,
                                     count_improvement_function=count_improvements_tfidf)

    tms = ["lda", "lsi", "nmf"]

    for tm in tms:
        for dr in drs:
            binary_tests_stability_1(required_tm=tm, required_dr=dr, base_path=base_path_1,
                                     count_improvement_function=count_improvements_linear_combined)
            binary_tests_stability_2(required_tm=tm, required_dr=dr, base_path=base_path_2,
                                     count_improvement_function=count_improvements_linear_combined)
            binary_tests_stability_3(required_tm=tm, required_dr=dr, base_path=base_path_3,
                                     count_improvement_function=count_improvements_linear_combined)


def calculate_alpha_beta_gamma(df):
    df["alpha"] = (0.25 * (df["trustworthiness"] + df["continuity"] + df["mrre_missing"] + df["mrre_false"]) + df[
        'label_preservation'] + df["local_continuity"]) / 3.0
    df["beta"] = 0.5 * (0.5 * (0.5 * (df["pearson_correlation"] + 1) + 0.5 * (df["spearman_correlation"] + 1)) + 0.5 * (
            df["cluster_ordering"] + 1))
    df["gamma"] = 1 - (df["distance_consistency"])

    return df


if __name__ == "__main__":
    main()
