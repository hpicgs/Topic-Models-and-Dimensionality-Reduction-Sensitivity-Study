#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, NearestNeighbors
from s_dbw import S_Dbw
from scipy import spatial, stats
from collections import Counter

import numpy as np


SAVE_EVALUATION_INTERMEDIATE_ARTEFACTS = False


# Combination of:
# (1) Jeon, Hyeon, et al. "ZADU: A Python Library for Evaluating the Reliability of
#       Dimensionality Reduction Embeddings." 2023 IEEE Visualization and Visual Analytics (VIS). IEEE, 2023.
# (2) Espadoto, Mateus, et al. "Toward a quantitative survey of dimension reduction techniques."
#       IEEE transactions on visualization and computer graphics 27.3 (2019): 2153-2173.
# (3) Atzberger, Daniel and Cech, Tim, et al. "Large-Scale Evaluation of Topic Models and Dimensionality Reduction
#       Methods for 2D Text Spatialization." IEEE Transactions on Visualization and Computer Graphics (2023).


def compute_distance_list(X, eval_distance_metric='euclidean'):
    return spatial.distance.pdist(X, eval_distance_metric)


def metric_neighborhood_hit(X, y, k=7):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))


def calculate_cluster_score_diff(X_high, X_low, y, score_func, additional_params=None):
    if additional_params is None:
        additional_params = {}
    original_score = score_func(X_high, y, **additional_params)
    projection_score = score_func(X_low, y, **additional_params)
    return calculate_projection_metric_diff(original_score, projection_score)


def calculate_projection_metric_diff(original_score, projection_score):
    diff = original_score - projection_score
    return diff


def metric_trustworthiness(X_high, X_low, D_high_m, D_low_m, k=7):
    D_high, D_low = get_squared_distances_if_necessary(D_high_m, D_low_m)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    try:
        trustworthiness = float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())
    except AttributeError:  # Everything stayed constant
        trustworthiness = 1.0

    return trustworthiness


def metric_continuity(X_high, X_low, D_high_l, D_low_l, k=7):
    D_high, D_low = get_squared_distances_if_necessary(D_high_l, D_low_l)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    try:
        continuity = float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())
    except AttributeError:  # Everything stayed the same
        continuity = 1.0

    return continuity


def get_squared_distances_if_necessary(D_high_l, D_low_l):
    if isinstance(D_high_l, list) or len(D_high_l.shape) == 1:
        D_high = spatial.distance.squareform(D_high_l)
        D_low = spatial.distance.squareform(D_low_l)
    else:
        D_high = D_high_l
        D_low = D_low_l
    return D_high, D_low


def metric_shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high, D_low)[0]


def metric_normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)


def metric_mse(X, X_hat):
    return np.mean(np.square(X - X_hat))


def metric_distance_consistency(X_2d, y):
    clf = NearestCentroid()
    clf.fit(X=X_2d, y=y)
    nearest_centroids = clf.predict(X=X_2d)
    num_same_label = sum([1 if y[i] == nearest_centroids[i] else 0 for i in range(len(y))])
    return num_same_label / len(y)


def compute_all_metrics(X, X_2d, D_high, D_low, y, eval_distance_metric='cosine'):
    start_time = time.time()
    T = metric_trustworthiness(X, X_2d, D_high, D_low)
    print("Elapsed time for getting Trustworthiness: " + str(time.time() - start_time) + " seconds", flush=True)
    start_time = time.time()
    C = metric_continuity(X, X_2d, D_high, D_low)
    print("Elapsed time for getting Continuity: " + str(time.time() - start_time) + " seconds", flush=True)
    start_time = time.time()
    R = metric_shepard_diagram_correlation(D_high, D_low)
    print("Elapsed time for getting Shepard Diagram Correlation: " + str(time.time() - start_time) + " seconds",
          flush=True)
    start_time = time.time()
    S = metric_normalized_stress(D_high, D_low)
    print("Elapsed time for getting Normalized Stress: " + str(time.time() - start_time) + " seconds", flush=True)
    start_time = time.time()
    N = metric_neighborhood_hit(X_2d, y)
    print("Elapsed time for getting 7-Neighborhood Hit: " + str(time.time() - start_time) + " seconds", flush=True)

    start_time = time.time()
    calinski_harabaz_low = calinski_harabasz_score(X_2d, y)
    print("Elapsed time for getting Calinski Harabasz Index: " + str(time.time() - start_time) + " seconds", flush=True)
    start_time = time.time()
    silhouette_low = silhouette_score(X_2d, y, metric='euclidean')
    print("Elapsed time for getting Silhouette Coefficient: " + str(time.time() - start_time) + " seconds", flush=True)
    start_time = time.time()
    davies_bouldin_low = davies_bouldin_score(X_2d, y)
    print("Elapsed time for getting Davies Bouldin Index: " + str(time.time() - start_time) + " seconds", flush=True)
    start_time = time.time()
    # For documentation see: https://github.com/alashkov83/S_Dbw
    sdbw_low = S_Dbw(X_2d, y, centers_id=None, method='Kim', alg_noise='sep', centr='mean', nearest_centr=True,
                       metric='euclidean')
    print("Elapsed time for getting SDBW index: " + str(time.time() - start_time) + " seconds")
    start_time = time.time()
    dsc_low = metric_distance_consistency(X_2d, y)
    print("Elapsed time for getting Distance Consistency: " + str(time.time() - start_time) + " seconds", flush=True)

    return T, C, R, S, N, calinski_harabaz_low, silhouette_low, davies_bouldin_low, sdbw_low, dsc_low


def evaluate_layouts(results_path, x, y, layouts_dict, dict_of_hyperparameter_dicts, old_df=None,
                     res_file_name="results", topic_level_experiment_name="", use_bow_for_comparison=True,
                     eval_distance_metric='cosine', suffix_permutation="", suffix_change_columns=""):
    results = []
    if use_bow_for_comparison:
        high_distance_path = os.path.join(results_path, "high_distances_bow_" + suffix_permutation + "_" +
                                          suffix_change_columns)
    else:
        high_distance_path = os.path.join(results_path, "high_distances_" + topic_level_experiment_name + "_" +
                                          suffix_permutation + "_" + suffix_change_columns)

    if os.path.isfile(high_distance_path + ".npy"):
        D_high = np.load(high_distance_path + ".npy")
    else:
        D_high = compute_distance_list(x, eval_distance_metric=eval_distance_metric)
        if SAVE_EVALUATION_INTERMEDIATE_ARTEFACTS:
            np.save(file=high_distance_path, arr=D_high)

    for experiment, embedding in layouts_dict.items():
        low_distance_path = os.path.join(results_path, "low_distances", experiment + "_" + suffix_permutation + "_" +
                                         suffix_change_columns)
        os.makedirs(low_distance_path, exist_ok=True)

        if os.path.isfile(low_distance_path + ".npy"):
            D_low = np.load(low_distance_path + ".npy")
        else:
            D_low = compute_distance_list(embedding, eval_distance_metric='euclidean')
            if SAVE_EVALUATION_INTERMEDIATE_ARTEFACTS:
                np.save(file=low_distance_path, arr=D_low)
        T, C, R, S, N, CA, SI, DB, SDBW, DSC = compute_all_metrics(x, embedding, D_high, D_low, y,
                                                                    eval_distance_metric=eval_distance_metric)
        print(str(dict_of_hyperparameter_dicts))
        experiment += ("_" + str(dict_of_hyperparameter_dicts["permute"]))
        experiment += ("_" + str(dict_of_hyperparameter_dicts["permute_ratio"]))
        experiment += ("_" + str(dict_of_hyperparameter_dicts["permute_shuffle_times"]))
        experiment += ("_" + str(dict_of_hyperparameter_dicts["scale_rows_shuffle_times_bow"]))
        experiment += ("_" + str(dict_of_hyperparameter_dicts["scale_columns_shuffle_times_bow"]))
        experiment += ("_" + str(dict_of_hyperparameter_dicts["drop_ratio_tm"]))
        experiment += ("_" + str(dict_of_hyperparameter_dicts["drop_ratio_tm_shuffle_times"]))
        results.append([experiment, T, C, R, S, N, CA, SI, DB, SDBW, DSC, str(dict_of_hyperparameter_dicts)])

    new_df = pd.DataFrame(results, columns=["Experiment", "Trustworthiness", "Continuity", "Shephard Diagram "
                                                                                           "Correlation", "Normalized "
                                                                                                          "Stress",
                                            "7-Neighborhood Hit", "Calinski-Harabasz-Index", "Silhouette "
                                                                                             "coefficient",
                                            "Davies-Bouldin-Index", "SDBW validity index", "Distance consistency",
                                            "Complete List of Hyperparameters"])

    if old_df is not None:
        new_df = pd.concat([new_df, old_df])
    new_df.to_csv(res_file_name.replace(".csv", "_partial.csv"), index=False)

    return new_df


def get_distance_matrix(x):
    return spatial.distance.squareform(x)


def count_common_elements(arr1, arr2):
    if isinstance(arr1, np.ndarray):
        arr1 = arr1.tolist()
    if isinstance(arr2, np.ndarray):
        arr2 = arr2.tolist()

    n_same_elements = 0
    for sub_arr1, sub_arr2 in list(zip(arr1, arr2)):
        counter1 = dict(Counter(sub_arr1))
        counter2 = dict(Counter(sub_arr2))
        same_elements = set(counter1.keys()).intersection(counter2.keys())

        for element in same_elements:
            n_same_elements += min(counter1[element], counter2[element])

    return n_same_elements


def metric_spearman_correlation(D_scatter1, D_scatter2):
    return stats.spearmanr(D_scatter1, D_scatter2)[0]


def metric_pearson_correlation(D_scatter1, D_scatter2):
    return stats.pearsonr(D_scatter1, D_scatter2)[0]


def metric_cluster_ordering(x_low1, x_low2, y):
    clf_scatter1 = NearestCentroid().fit(X=x_low1, y=y)
    clf_scatter2 = NearestCentroid().fit(X=x_low2, y=y)

    distance_list1 = compute_distance_list(clf_scatter1.centroids_)
    distance_list2 = compute_distance_list(clf_scatter2.centroids_)

    return metric_spearman_correlation(distance_list1, distance_list2)


def center_scatterplot(x):
    n, m = x.shape
    n_mean = sum(x[:, 0].tolist()) / n
    m_mean = sum(x[:, 1].tolist()) / n

    return x - [n_mean, m_mean]


def procrustes_manual(x, y, scaling=True, reflection='best'):
    n, m = x.shape
    mu_x = x.mean(0)
    mu_y = y.mean(0)

    x_0 = x - mu_x
    y_0 = y - mu_y
    ss_x = (x_0 ** 2).sum()
    ss_y = (y_0 ** 2).sum()

    # centred Frobenius norm
    norm_x = np.sqrt(ss_x)
    norm_y = np.sqrt(ss_y)

    # scale to equal (unit) norm
    x_0 /= norm_x
    y_0 /= norm_y

    # optimum rotation matrix of y
    combined_matrix = np.dot(x_0.T, y_0)
    unitary_arrays_u, singular_values, unitary_arrays_v_transposed = np.linalg.svd(combined_matrix, full_matrices=False)
    unitary_arrays_v = unitary_arrays_v_transposed.T
    combined_unitary_arrays = np.dot(unitary_arrays_v, unitary_arrays_u.T)

    if reflection != "best":
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(combined_unitary_arrays) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            unitary_arrays_v[:, -1] *= -1
            singular_values[-1] *= -1
            combined_unitary_arrays = np.dot(unitary_arrays_v, unitary_arrays_u.T)

    trace_singular_values = singular_values.sum()

    if scaling:
        # optimum scaling of y
        opt_scaling = trace_singular_values * norm_x / norm_y

        # standardized distance between X and opt_scaling*y*combined_unitary_arrays + c
        standardized_distance = 1 - trace_singular_values ** 2

        transformed_coords = norm_x * trace_singular_values * np.dot(y_0, combined_unitary_arrays) + mu_x
    else:
        opt_scaling = 1
        standardized_distance = 1 + ss_y / ss_x - 2 * trace_singular_values * norm_y / norm_x
        transformed_coords = norm_y * np.dot(y_0, combined_unitary_arrays) + mu_x

    if y.shape[1] < m:
        combined_unitary_arrays = combined_unitary_arrays[:y.shape[1], :]
    transformation_matrix = mu_x - opt_scaling * np.dot(mu_y, combined_unitary_arrays)

    transformation_values = {'rotation': combined_unitary_arrays, 'scale': opt_scaling,
                             'translation': transformation_matrix}

    return standardized_distance, transformed_coords, transformation_values


def get_rotation_angle(rotation_matrix):
    angle_radiant = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.degrees(angle_radiant)


def metric_label_preservation(x_low1, x_low2, y, k=7):
    label_preservation = 0

    neighbors_scatter1 = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(x_low1)
    neighbors_scatter2 = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(x_low2)

    for i in range(x_low1.shape[0]):
        distances_scatter1, indices_scatter1 = neighbors_scatter1.kneighbors(x_low1[i].reshape(1, -1))
        distances_scatter2, indices_scatter2 = neighbors_scatter2.kneighbors(x_low2[i].reshape(1, -1))

        indices_scatter1 = indices_scatter1.tolist()
        indices_scatter2 = indices_scatter2.tolist()
        labels_scatter1 = np.array([y[index] for index in indices_scatter1])
        labels_scatter2 = np.array([y[index] for index in indices_scatter2])

        label_preservation_pointwise = (count_common_elements(labels_scatter1, labels_scatter2) - 1) / k
        label_preservation += label_preservation_pointwise

    return label_preservation / x_low1.shape[0]


def metric_local_continuity(n, distances_squared1, distances_squared2, k=7):
    k_nearest_neighbors_low1 = distances_squared1.argsort()[:, :k + 1][:, 1:]
    k_nearest_neighbors_low2 = distances_squared2.argsort()[:, :k + 1][:, 1:]

    local_distortion_list = np.array([np.intersect1d(k_nearest_neighbors_low1[i],
                                                     k_nearest_neighbors_low2[i]).shape[0] - ((k * k) / (n - 1)) for i
                                      in range(n)])
    local_distortion_list /= k
    return np.mean(local_distortion_list)


def knn_with_ranking(points, k, distance_matrix):
    knn_indices = np.empty((points.shape[0], k), dtype=np.int32)
    ranking = np.empty((points.shape[0], points.shape[0]), dtype=np.int32)

    for i in range(points.shape[0]):
        distance_to_i = distance_matrix[i]
        sorted_indices = np.argsort(distance_to_i)
        knn_indices[i] = sorted_indices[1:k + 1]
        ranking[i] = np.argsort(sorted_indices)

    return knn_indices, ranking


def metric_mrre_help(base_ranking, target_ranking, target_knn_indices, k=7):
    local_distortion_list = []
    points_num = target_knn_indices.shape[0]
    for i in range(points_num):
        base_rank_arr = base_ranking[i][target_knn_indices[i]]
        target_rank_arr = target_ranking[i][target_knn_indices[i]]
        local_distortion_list.append(np.sum(np.abs(base_rank_arr - target_rank_arr) / target_rank_arr))

    c = sum([abs(points_num - 2 * i + 1) / i for i in range(1, k + 1)])
    local_distortion_list = np.array(local_distortion_list)
    local_distortion_list = 1 - local_distortion_list / c

    average_distortion = np.mean(local_distortion_list)

    return average_distortion


def metric_mrre(x_low1, x_low2, distances_squared1, distances_squared2, k=7):
    knn_indices1, ranking1 = knn_with_ranking(x_low1, k, distances_squared1)
    knn_indices2, ranking2 = knn_with_ranking(x_low2, k, distances_squared2)

    mrre_false = metric_mrre_help(ranking1, ranking2, knn_indices2, k)
    mrre_missing = metric_mrre_help(ranking2, ranking1, knn_indices1, k)

    return mrre_false, mrre_missing


def absolute_difference_of_metric(metric_func, layout1, layout2, y):
    return abs(metric_func(layout1, y) - metric_func(layout2, y))


def get_all_metrics_stability(layout1_path, layout2_path, y, return_dict, metric="euclidean", calculate_rotation=True):
    key = (layout1_path, layout2_path)
    # start_time = time.time()
    # print("Started experiment for correlation pair: " + str(key) + " at " + str(time.ctime()))

    # Doing it here to not use so much memory upfront during creation of Processes
    layout1 = np.load(layout1_path)
    distances1 = compute_distance_list(layout1, eval_distance_metric=metric)
    distances_square_form1 = get_distance_matrix(distances1)
    layout2 = np.load(layout2_path)
    distances2 = compute_distance_list(layout2, eval_distance_metric=metric)
    distances_square_form2 = get_distance_matrix(distances2)

    spearman_correlation = metric_spearman_correlation(distances1, distances2)
    pearson_correlation = metric_pearson_correlation(distances1, distances2)
    cluster_ordering = metric_cluster_ordering(layout1, layout2, y)

    if calculate_rotation:
        standardized_distance, transformed_coords, transformation_values = procrustes_manual(center_scatterplot(layout1),
                                                                                             center_scatterplot(layout2),
                                                                                             reflection="False")
        rotation = get_rotation_angle(transformation_values["rotation"])
    else:
        rotation = np.nan
    distance_consistency = absolute_difference_of_metric(metric_distance_consistency, layout1, layout2, y)
    silhouette_coefficient = absolute_difference_of_metric(silhouette_score, layout1, layout2, y)
    trustworthiness = metric_trustworthiness(layout1, layout2, distances_square_form1, distances_square_form2, k=7)
    continuity = metric_continuity(layout1, layout2, distances_square_form1, distances_square_form2, k=7)
    local_continuity = metric_local_continuity(layout1.shape[0], distances_square_form1, distances_square_form2, k=7)
    mrre_missing, mrre_false = metric_mrre(layout1, layout2, distances_square_form1, distances_square_form2, k=7)
    label_preservation = metric_label_preservation(layout1, layout2, y, k=7)

    metrics = [spearman_correlation, pearson_correlation, cluster_ordering, rotation, distance_consistency,
               silhouette_coefficient, trustworthiness, continuity, local_continuity, mrre_missing, mrre_false,
               label_preservation]

    metrics.extend([layout1_path, layout2_path])
    return_dict[key] = metrics
    # print("Finished experiment for correlation pair: " + str(key) + " in " + str(time.time() - start_time) +
    #      " seconds.")
