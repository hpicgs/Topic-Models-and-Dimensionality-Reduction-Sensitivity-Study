import hashlib

import pandas as pd
import numpy as np
from tqdm import tqdm

from commons import permute_columns
from create_topic_layout import convert_text_to_corpus, get_jittered_corpus, create_bow_model
from main import get_raw_dataset
import os

from metrics import get_all_metrics_stability
from nlp_standard_preprocessing import load_dataset_if_able


def main():
    jitter_amounts = [0.1, 0.25]
    datasets = ["20_newsgroups", "seven_categories", "lyrics", "reuters", "bbc_news", "emails"]
    results_dict = dict()
    results = []

    for dataset in datasets:
        results_file = f"results_experiment_5_corpus_similarity_{dataset}.csv"
        if os.path.isfile(results_file):
            continue

        model_path = os.path.join("models", dataset)
        filter_threshold_bow, needs_lemmatization, perform_standard_preprocessing, x, y = get_raw_dataset(dataset)
        dataset_dir = os.path.join("data", dataset)
        file_path = os.path.join(dataset_dir, dataset + "_words_list_" + str(len(x)) + ".pkl")
        if os.path.isfile(file_path):
            x = load_dataset_if_able(file_path)
        to_discard = [i for i, text in enumerate(x) if len(text) <= 1]
        x = [x[i] for i in range(len(x)) if i not in to_discard]
        y = np.array([y[i] for i in range(len(y)) if i not in to_discard])
        suffix_permutation = str(hashlib.sha256(str(x).encode()).hexdigest())[:15]

        dictionary, corpus = convert_text_to_corpus(x)
        dictionary.save(os.path.join(model_path, "dictionary" + "_" + str(len(dictionary))))
        n_columns = max(list(dictionary.keys()))

        corpus, suffix_change_columns = permute_columns(corpus, 0, 0,
                                                        n_columns,
                                                        0,
                                                        0,
                                                        0, 0,
                                                        0)

        for jitter_amount in tqdm(jitter_amounts):

            bow_path_jitter = os.path.join(model_path,
                                           "bow_model_" + str(0.001) + "_" + str(len(dictionary)) +
                                           "_" + suffix_permutation + "_" + suffix_change_columns + "_j" +
                                           str(jitter_amount) + ".npy")

            bow_path = os.path.join(model_path,
                                    "bow_model_" + str(0.001) + "_" + str(len(dictionary)) +
                                    "_" + suffix_permutation + "_" + suffix_change_columns + ".npy")

            for file in os.listdir(model_path):
                if "bow_model" in file and "_j" + str(jitter_amount) in file:
                    bow_path_jitter = os.path.join(model_path, file)
            data1 = np.load(bow_path_jitter)

            for file in os.listdir(model_path):
                if "bow_model" in file and "_j" not in file:
                    bow_path = os.path.join(model_path, file)
                    data2 = np.load(bow_path)
                    if data1.shape == data2.shape:
                        break


            get_all_metrics_stability(bow_path, bow_path_jitter, y, results_dict, metric="cosine",
                                      calculate_rotation=False)
            key = (bow_path, bow_path_jitter)
            cur_result = [0.0, jitter_amount]
            cur_result.extend(results_dict[key])
            results.append(cur_result)

        res_df = pd.DataFrame(results, columns=["jitter_1", "jitter_2", "spearman_correlation",
                                                "pearson_correlation",
                                                "cluster_ordering", "rotation",
                                                "distance_consistency", "silhouette_coefficient",
                                                "trustworthiness", "continuity", "local_continuity",
                                                "mrre_missing", "mrre_false", "label_preservation",
                                                "file_name_1",
                                                "file_name_2"])
        res_df.to_csv(results_file)


if __name__ == "__main__":
    main()
