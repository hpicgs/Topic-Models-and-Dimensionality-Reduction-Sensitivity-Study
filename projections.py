import gc
import pickle
import time

from pynndescent import NNDescent
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import matplotlib

from commons import shorten_res_file_name, print_current_memory_usage
from config import get_global_random_seed
from metrics import evaluate_layouts

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sparse_som import *
from sklearn_som.som import SOM
from scipy.spatial.distance import pdist, squareform, jensenshannon, cosine
import os
import warnings


def create_tsne_layout(x, y, result_path, dataset_name, test_name, res_dict, perplexity=30.0, early_exaggeration=12.0,
                       learning_rate='auto', n_iter=250, init="random", method="barnes_hut", angle=0.5,
                       metric='cosine', suffix_permutation="", suffix_change_columns=""):
    dest_arr_path = os.path.join(result_path,
                                 "tsne_" + dataset_name + "_" + test_name + "_" + str(perplexity) + "_" + str(
                                     early_exaggeration) + "_" + str(learning_rate) + "_" + str(n_iter) + "_" + str(
                                     init) + "_" + str(
                                     method) + "_" + str(angle) + "_" + str(metric)
                                 + "_" + suffix_permutation + "_" + suffix_change_columns)

    dest_arr_path = shorten_res_file_name(dest_arr_path, offset=10)
    dest_file_path = dest_arr_path + ".png"

    if os.path.isfile(dest_arr_path + ".npy"):
        embedded_vectors = np.load(dest_arr_path + ".npy")
        res_dict[dataset_name + "_" + test_name + '_tsne'] = embedded_vectors
        warnings.warn("T-SNE layout already processed! I will not reprocess it and return early.")
        return res_dict

    try:
        embedded_vectors = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
                                learning_rate=learning_rate, n_iter=n_iter,
                                init=init, method=method, angle=angle, random_state=get_global_random_seed(),
                                metric=metric).fit_transform(x)
    except (ValueError, AssertionError) as e:
        method = "exact"
        print("Encountered error while processing TSNE: " + str(e), flush=True)
        print("I will retry with exact computation of neighbours", flush=True)

        try:
            embedded_vectors = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
                                    learning_rate=learning_rate, n_iter=n_iter,
                                    init=init, method=method, angle=angle, random_state=get_global_random_seed(),
                                    metric=metric).fit_transform(
                x)
        except:
            print("Failed to recover. I will return without having processed t-SNE", flush=True)
            return res_dict

    plt.scatter([x[0] for x in embedded_vectors], [x[1] for x in embedded_vectors], c=y, alpha=0.5)
    plt.savefig(dest_file_path)
    plt.close()

    np.save(file=dest_arr_path, arr=embedded_vectors)
    res_dict[dataset_name + "_" + test_name + '_tsne'] = embedded_vectors
    return res_dict


def create_umap_layout(x, y, result_path, dataset_name, test_name, res_dict, n_neighbors=15, min_dist=0.1,
                       metric='euclidean',
                       spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1.0,
                       negative_sample_rate=5, suffix_permutation="", suffix_change_columns=""):
    dest_arr_path = os.path.join(result_path,
                                 "umap_" + dataset_name + "_" + test_name + "_" + str(n_neighbors) + "_" + str(
                                     min_dist) + "_" + str(metric) + "_" + str(spread) + "_" + str(
                                     set_op_mix_ratio) + "_" + str(
                                     local_connectivity) + "_" + str(repulsion_strength) + "_" + str(
                                     negative_sample_rate) + "_" + suffix_permutation + "_" + suffix_change_columns)

    dest_arr_path = shorten_res_file_name(dest_arr_path, offset=10)
    dest_file_path = dest_arr_path + ".png"

    if os.path.isfile(dest_arr_path + ".npy"):
        embedded_vectors = np.load(dest_arr_path + ".npy")
        res_dict[dataset_name + "_" + test_name + '_umap'] = embedded_vectors
        warnings.warn("UMAP layout already processed! I will not reprocess it and return early.")
        return res_dict

    knn_search_index = NNDescent(
        x,
        n_neighbors=n_neighbors,
        n_jobs=None,
        low_memory=False,
        verbose=True,
        compressed=False
    )
    knn_indices, knn_dists = knn_search_index.neighbor_graph

    embedded_vectors = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, spread=spread,
                            set_op_mix_ratio=set_op_mix_ratio, local_connectivity=local_connectivity,
                            repulsion_strength=repulsion_strength, random_state=get_global_random_seed(),
                            verbose=True, n_jobs=1,
                            negative_sample_rate=negative_sample_rate,
                            precomputed_knn=(knn_indices, knn_dists, knn_search_index)).fit_transform(x)

    plt.scatter([x[0] for x in embedded_vectors], [x[1] for x in embedded_vectors], c=y, alpha=0.5)
    plt.savefig(dest_file_path)
    plt.close()

    np.save(file=dest_arr_path, arr=embedded_vectors)
    res_dict[dataset_name + "_" + test_name + '_umap'] = embedded_vectors
    return res_dict


def create_pca_layout(x, y, result_path, dataset_name, test_name, res_dict):
    dest_file_path = os.path.join(result_path, "pca_" + dataset_name + "_" + test_name + ".png")
    dest_arr_path = os.path.join(result_path, "pca_" + dataset_name + "_" + test_name)

    if os.path.isfile(dest_arr_path + ".npy"):
        embedded_vectors = np.load(dest_arr_path + ".npy")
        res_dict[dataset_name + "_" + test_name + '_pca'] = embedded_vectors
        return res_dict
    embedded_vectors = PCA(n_components=2, random_state=get_global_random_seed()).fit_transform(x)

    plt.scatter([x[0] for x in embedded_vectors], [x[1] for x in embedded_vectors], c=y, alpha=0.5)
    plt.savefig(dest_file_path)
    plt.close()

    np.save(file=dest_arr_path, arr=embedded_vectors)
    res_dict[dataset_name + "_" + test_name + '_pca'] = embedded_vectors
    return res_dict


def create_mds_layout(x, y, result_path, dataset_name, res_dict, test_name, metric=True, dissimilarity_metric='cosine',
                      max_iter=300, suffix_permutation="", suffix_change_columns=""):
    dest_arr_path = os.path.join(result_path, "mds_" + dataset_name + "_" + test_name + "_" + str(metric) + "_" + str(
        dissimilarity_metric) + "_" + str(max_iter) + "_" + suffix_permutation + "_" + suffix_change_columns)
    dest_arr_path = shorten_res_file_name(dest_arr_path, offset=10)
    dest_file_path = dest_arr_path + ".png"

    if os.path.isfile(dest_arr_path + ".npy"):
        embedded_vectors = np.load(dest_arr_path + ".npy")
        res_dict[dataset_name + "_" + test_name + '_mds'] = embedded_vectors
        warnings.warn("MDS layout already processed! I will not reprocess it and return early.")
        return res_dict

    dists = pdist(X=x, metric=dissimilarity_metric)
    dists = np.nan_to_num(dists)
    dist_matrix = squareform(dists)
    dist_matrix = 1 - ((dist_matrix - np.min(dist_matrix)) / (np.max(dist_matrix) - np.min(dist_matrix)))
    embedded_vectors = MDS(n_components=2, random_state=get_global_random_seed(),
                           metric=metric, dissimilarity='precomputed',
                           max_iter=max_iter, normalized_stress=False).fit_transform(dist_matrix)

    plt.scatter([x[0] for x in embedded_vectors], [x[1] for x in embedded_vectors], c=y, alpha=0.5)
    plt.savefig(dest_file_path)
    plt.close()

    np.save(file=dest_arr_path, arr=embedded_vectors)
    res_dict[dataset_name + "_" + test_name + '_mds'] = embedded_vectors
    return res_dict


def create_som_layout(x, y, result_path, dataset_name, res_dict, test_name, n=10, m=10, use_five_percent_share=True,
                      use_sparse_som=True, suffix_permutation="", suffix_change_columns=""):
    dest_arr_path = os.path.join(result_path, "som_" + dataset_name + "_" + test_name + "_" + str(n) + "_" + str(
        m) + "_" + suffix_permutation + "_" + suffix_change_columns)

    dest_arr_path = shorten_res_file_name(dest_arr_path, offset=10)
    dest_file_path = dest_arr_path + ".png"
    mode_path = dest_arr_path + ".mode"

    if os.path.isfile(dest_arr_path + ".npy") and os.path.isfile(mode_path):
        embedded_vectors = np.load(dest_arr_path + ".npy")
        use_five_percent_share = pickle.load(open(mode_path, "rb"))
        if use_five_percent_share:
            res_dict[dataset_name + "_" + test_name + '_som'] = embedded_vectors
        else:
            res_dict[dataset_name + "_" + test_name + '_som_1000_fix'] = embedded_vectors
        warnings.warn("SOM layout already processed! I will not reprocess it and return early.")
        return res_dict
    elif os.path.isfile(dest_arr_path + ".npy") and use_five_percent_share:
        embedded_vectors = np.load(dest_arr_path + ".npy")
        res_dict[dataset_name + "_" + test_name + '_som'] = embedded_vectors
        warnings.warn("SOM layout already processed! I will not reprocess it and return early."
                      " I couldn't validate the PCA mode though."
                      " Therefore I will assume, that the five percent share was used")
        return res_dict

    if x.shape[1] > 1000:
        if use_five_percent_share:
            x = PCA(n_components=0.95, svd_solver="full", random_state=get_global_random_seed()).fit_transform(x)
        else:
            x = PCA(n_components=1000, random_state=get_global_random_seed()).fit_transform(x)

    print("Deactivated PCA preprocessing", flush=True)
    if use_sparse_som:
        print("Using sparse SOM", flush=True)
        X = csr_matrix(x)
        print("Got csr matrix!", flush=True)
        _, dimensions = X.shape
        print(X, flush=True)
        print(type(X.indices.dtype), flush=True)
        print(type(X.indptr.dtype), flush=True)
        print(type(X.shape[0]), flush=True)
        print(type(X.shape[1]), flush=True)
        print(type(X.nnz), flush=True)
        som = Som(m, n, dimensions, topology.HEXA, verbose=2)
        print("Created SOM object!", flush=True)
        som.train(X)
        print("Trained SOM!", flush=True)
        embedded_vectors = som.bmus(X)
        print("Created SOM layout!", flush=True)
    else:
        som = SOM(m=m, n=n, dim=x.shape[1])
        som.fit(x)
        embedded_labels = som.predict(x)
        distances = som.transform(x)
        embedded_vectors = np.array(
            [[int(embedded_labels[i] / n) + distances[i][embedded_labels[i]], int(embedded_labels[i] % n)]
             for i in range(len(embedded_labels))])

    plt.scatter([x[0] for x in embedded_vectors], [x[1] for x in embedded_vectors], c=y, alpha=0.5)
    plt.savefig(dest_file_path)
    plt.close()

    np.save(file=dest_arr_path, arr=embedded_vectors)
    with open(mode_path, "wb+") as mode_file:
        pickle.dump(use_five_percent_share, mode_file)
    if use_five_percent_share:
        res_dict[dataset_name + "_" + test_name + '_som'] = embedded_vectors
    else:
        res_dict[dataset_name + "_" + test_name + '_som_1000_fix'] = embedded_vectors
    return res_dict


def create_layouts(x, y, result_path, dataset_name, test_name, umap_hyperparameters=None, tsne_hyperparameters=None,
                   pca_hyperparameters=None, mds_hyperparameters=None, som_hyperparameters=None, only_som=False,
                   only_tsne=False, suffix_permutation="", suffix_change_columns=""):
    os.makedirs(result_path, exist_ok=True)
    if mds_hyperparameters is None:
        mds_hyperparameters = {}
    if pca_hyperparameters is None:
        pca_hyperparameters = {}
    if tsne_hyperparameters is None:
        tsne_hyperparameters = {}
    if umap_hyperparameters is None:
        umap_hyperparameters = {}
    if som_hyperparameters is None:
        som_hyperparameters = {}

    if "lda" in test_name:
        metric = "jensenshannon"
    else:
        metric = "cosine"
    umap_hyperparameters["metric"] = metric
    mds_hyperparameters["dissimilarity_metric"] = metric

    res_dict = dict()
    x = np.nan_to_num(x)

    print("Begin processing layouts", flush=True)
    if only_som:
        start = time.time()
        print("Begin SOM layout!", flush=True)
        res_dict = create_som_layout(x=x, y=y, result_path=result_path, dataset_name=dataset_name,
                                     test_name=test_name, res_dict=res_dict,
                                     suffix_permutation=suffix_permutation, suffix_change_columns=suffix_change_columns,
                                     **som_hyperparameters)
        print("Finished SOM layout!", flush=True)
        print("Elapsed time for getting SOM layout: " + str(time.time() - start), flush=True)
    elif only_tsne:
        start = time.time()
        res_dict = create_tsne_layout(x=x, y=y, result_path=result_path, dataset_name=dataset_name,
                                      test_name=test_name, res_dict=res_dict, metric=metric,
                                      suffix_permutation=suffix_permutation,
                                      suffix_change_columns=suffix_change_columns,
                                      **tsne_hyperparameters)
        print("Finished TSNE layout!", flush=True)
        print("Elapsed time for getting TSNE layout: " + str(time.time() - start), flush=True)
    else:
        start = time.time()
        res_dict = create_tsne_layout(x=x, y=y, result_path=result_path, dataset_name=dataset_name,
                                      test_name=test_name, res_dict=res_dict, metric=metric,
                                      suffix_permutation=suffix_permutation,
                                      suffix_change_columns=suffix_change_columns,
                                      **tsne_hyperparameters)
        print("Finished TSNE layout!", flush=True)
        print("Elapsed time for getting TSNE layout: " + str(time.time() - start), flush=True)

        start = time.time()
        res_dict = create_umap_layout(x=x, y=y, result_path=result_path, dataset_name=dataset_name,
                                      test_name=test_name, res_dict=res_dict,
                                      suffix_permutation=suffix_permutation,
                                      suffix_change_columns=suffix_change_columns,
                                      **umap_hyperparameters)
        print("Finished UMAP layout!", flush=True)
        print("Elapsed time for getting UMAP layout: " + str(time.time() - start), flush=True)

        start = time.time()
        res_dict = create_mds_layout(x=x, y=y, result_path=result_path, dataset_name=dataset_name,
                                     test_name=test_name, res_dict=res_dict,
                                     suffix_permutation=suffix_permutation, suffix_change_columns=suffix_change_columns,
                                     **mds_hyperparameters)
        print("Finished MDS layout!", flush=True)
        print("Elapsed time for getting MDS layout: " + str(time.time() - start), flush=True)
    print("Created all layouts!", flush=True)

    return res_dict


def evaluate_experiment(dataset_name, dict_of_hyperparameters, experiment, high_embedding, only_som, only_tsne, res_df,
                        res_file_name, res_path, suffix_change_columns, suffix_permutation, y):
    print("Processing experiment: " + str(experiment), flush=True)
    embeddings = create_layouts(high_embedding, y, res_path, dataset_name, experiment,
                                umap_hyperparameters=dict_of_hyperparameters['umap'],
                                tsne_hyperparameters=dict_of_hyperparameters['tsne'],
                                som_hyperparameters=dict_of_hyperparameters['som'],
                                mds_hyperparameters=dict_of_hyperparameters['mds'], only_som=only_som,
                                only_tsne=only_tsne, suffix_permutation=suffix_permutation,
                                suffix_change_columns=suffix_change_columns)
    start = time.time()
    res_df = evaluate_layouts(res_path, high_embedding, y, embeddings, dict_of_hyperparameters, old_df=res_df,
                              res_file_name=res_file_name, topic_level_experiment_name=experiment,
                              use_bow_for_comparison=True, suffix_permutation=suffix_permutation,
                              suffix_change_columns=suffix_change_columns)
    print("Elapsed time for evaluation: " + str(time.time() - start), flush=True)
    # if "bow" not in experiment:
    #    res_df = evaluate_layouts(res_path, high_embedding, y, embeddings, dict_of_hyperparameters, old_df=res_df,
    #                              res_file_name=res_file_name, topic_level_experiment_name=experiment,
    #                              use_bow_for_comparison=False)
    print("Finished evaluation of experiment: " + str(experiment), flush=True)
    print_current_memory_usage()
    del embeddings
    gc.collect()
    print_current_memory_usage()

    return res_df
