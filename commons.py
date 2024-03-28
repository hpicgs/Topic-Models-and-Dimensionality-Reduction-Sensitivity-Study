import os
import pickle
import random
from argparse import ArgumentTypeError

import humanize
import numpy as np
import psutil
import hashlib
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import get_global_random_seed


def get_rel_path(path_list, dataset_name):
    return [get_single_rel_path(dataset_name, path) for path in path_list]


def get_single_rel_path(dataset_name, path):
    paths = path.split(os.path.sep)
    relevant_paths = []
    found_begin = False
    for path in paths:
        if dataset_name in path:
            found_begin = True

        if found_begin:
            relevant_paths.append(path)

    try:
        final_path = os.path.join(*relevant_paths)
    except TypeError:
        print("Found malformed path")
        final_path = ""
    return final_path


def transform_sparse_model_to_dense_matrix(bow_corpus, vector_length):
    dense_matrix = []
    for tokenized_line in bow_corpus:
        cur_pos = 0
        dense_vector = []
        for i in range(0, vector_length):
            if len(tokenized_line) > 0 and tokenized_line[cur_pos][0] == i:
                dense_vector.append(tokenized_line[cur_pos][1])
                cur_pos = min(cur_pos + 1, len(tokenized_line) - 1)
            else:
                dense_vector.append(0)
        dense_matrix.append(np.array(dense_vector))
    dense_matrix = np.array(dense_matrix)
    return dense_matrix


def write_list(a_list, file_path):
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb+') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')


def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def split_file(file, chunksize):
    chunks = [file[x:x + chunksize] for x in range(0, len(file), chunksize)]
    return chunks


def print_current_memory_usage():
    print("Current memory usage: " + str(humanize.naturalsize(psutil.Process(os.getpid()).memory_info().rss)),
          flush=True)


def get_stratified_parts(x, y, ratio):
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    x_new = []
    y_new = []

    for i, (train_indices, test_indices) in enumerate(skf.split(x, y)):
        x_new.extend([x[i] for i in range(len(x)) if i in test_indices])
        y_new.extend(y[test_indices])

        if i >= ratio:
            return np.array(x_new), np.array(y_new)

    return x_new, np.array(y_new)


def shuffle_data(x, y, times):
    indices = list(range(len(x)))
    shuffler = random.Random(get_global_random_seed())

    for _ in range(times):
        shuffler.shuffle(indices)

    x = np.array(x)
    y = np.array(y)
    return x[indices], y[indices]


def permute_rows(x, y, mode="change_rows", ratio=5, times=1):
    if mode == "use_part":
        x, y = get_stratified_parts(x, y, ratio)
    if mode == "shuffle_rows":
        x, y = shuffle_data(x, y, times)

    return list(x), np.array(y)


def drop_columns(x, n_columns, drop_ratio_tm, drop_ratio_tm_shuffle_times, drop_columns_func):
    real_ratio = drop_ratio_tm * 5  # as documented in argument parser
    n_columns_to_drop = int(n_columns * (real_ratio / 100))
    indices = list(range(n_columns))
    shuffler = random.Random(get_global_random_seed())

    for _ in range(drop_ratio_tm_shuffle_times):
        shuffler.shuffle(indices)

    columns_to_drop = indices[:n_columns_to_drop]
    x = drop_columns_func(x, columns_to_drop)

    return x


def scale_columns(x, n_columns, scale_columns_shuffle_times_bow, scale_columns_func):
    indices = list(range(n_columns))
    shuffler = random.Random(get_global_random_seed())

    for _ in range(scale_columns_shuffle_times_bow):
        shuffler.shuffle(indices)

    indices = indices[:int(len(indices) * 0.2)]  # Only scale 20% of columns by five
    x = scale_columns_func(x, indices)

    return x


def scale_rows(x, scale_rows_shuffle_times_bow, scale_row_func):
    n_rows = len(x)
    indices = list(range(n_rows))
    shuffler = random.Random(get_global_random_seed())

    for _ in range(scale_rows_shuffle_times_bow):
        shuffler.shuffle(indices)

    indices = indices[:int(len(indices) * 0.2)]  # Only scale 20% of columns by two
    x = scale_row_func(x, indices)
    return x


def permute_columns(x, drop_ratio_tm, drop_ratio_tm_shuffle_times, n_columns, scale_columns_shuffle_times_bow,
                    scale_rows_shuffle_times_bow, scale_columns_func, scale_rows_func, drop_columns_func):
    if scale_columns_shuffle_times_bow > 0:
        x = scale_columns(x, n_columns, scale_columns_shuffle_times_bow, scale_columns_func)
    if scale_rows_shuffle_times_bow > 0:
        x = scale_rows(x, scale_rows_shuffle_times_bow, scale_rows_func)
    if drop_ratio_tm > 0 and drop_ratio_tm_shuffle_times > 0:
        x = drop_columns(x, n_columns, drop_ratio_tm, drop_ratio_tm_shuffle_times, drop_columns_func)
    suffix_change_columns = str(hashlib.sha256(str(x).encode()).hexdigest())[:15]
    return x, suffix_change_columns


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str_or_float_value(v, accepted_strings=None, argument_name=""):
    if accepted_strings is None:
        accepted_strings = []
    try:
        v = float(v)
        return v
    except ValueError:
        if v in accepted_strings:
            return v
        else:
            raise ArgumentTypeError('Argument ' + argument_name + " has either to be float or in " +
                                    str(accepted_strings))


def amend_permute_parameters_to_hyperparameter_dict(dict_of_hyperparameters, drop_ratio_tm, drop_ratio_tm_shuffle_times,
                                                    permute, permute_ratio, permute_shuffle_times,
                                                    scale_columns_shuffle_times_bow, scale_rows_shuffle_times_bow,
                                                    jitter_amount):
    dict_of_hyperparameters["permute"] = permute
    dict_of_hyperparameters["permute_ratio"] = permute_ratio
    dict_of_hyperparameters["permute_shuffle_times"] = permute_shuffle_times
    dict_of_hyperparameters["scale_rows_shuffle_times_bow"] = scale_rows_shuffle_times_bow
    dict_of_hyperparameters["scale_columns_shuffle_times_bow"] = scale_columns_shuffle_times_bow
    dict_of_hyperparameters["drop_ratio_tm"] = drop_ratio_tm
    dict_of_hyperparameters["drop_ratio_tm_shuffle_times"] = drop_ratio_tm_shuffle_times
    dict_of_hyperparameters["jitter_amount"] = jitter_amount

    return dict_of_hyperparameters


def shorten_res_file_name(res_file_name, offset=40):
    max_length_path = min(os.pathconf('/', 'PC_PATH_MAX'), os.pathconf('/', 'PC_NAME_MAX')) - offset
    if len(res_file_name) >= max_length_path:
        print("Encountered path with length: " + str(len(res_file_name)) +
              "! I will have to shorten it to avoid subsequent osErrors. For this, I will use a hash of the original"
              " name!", flush=True)
        name_hash = str(hashlib.sha256(str(res_file_name).encode()).hexdigest())[:15]
        name_max_len = max_length_path - len(name_hash) - offset  # Account for .csv and overhead
        res_file_name = res_file_name[:name_max_len] + "_" + name_hash + ".csv"
    return res_file_name


def add_jitter(df, jitter_amount=0.1, save_file_path=None):
    """
    Add jittering to a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame
        The input DataFrame containing the data.
    - jitter_amount: float (default: 0.1)
        The amount of jittering to be applied to the data.
        Higher values result in greater jittering.

    Returns:
    - jittered_df: pandas DataFrame
        The DataFrame with jittered data.
    """

    if os.path.isfile(save_file_path):
        print("Found data file with desired amount of jittering. I will use this and return early!", flush=True)
        return np.load(save_file_path + ".npy")

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=None)

    # Determine the range of jittering based on the data range
    print("Computing range")
    data_range = df.max() - df.min()

    # Add random noise to each data point
    for column in df.columns:
        # Calculate the amount of jittering for the current column
        jitter_range = jitter_amount * data_range[column]

        # Generate random noise with the same shape as the column
        noise = np.random.uniform(low=-jitter_range / 2, high=jitter_range / 2, size=len(df))

        # Add noise to the column
        df[column] += noise

    data_new = df.to_numpy()
    if save_file_path is not None and not os.path.isfile(save_file_path):
        np.save(save_file_path, arr=data_new)

    return data_new


def get_layouts_paths(dataset_path):
    filenames = os.listdir(dataset_path)
    file_paths = []

    for file in filenames:
        if file.endswith("_y.npy"):
            continue
        elif file.endswith(".npy"):
            if file.replace(".npy", ".png") in filenames:
                file_paths.append(os.path.join(dataset_path, file))

    return file_paths


def get_y_paths(dataset_path, constant_y=False):
    filenames = os.listdir(dataset_path)
    file_paths = []

    for file in filenames:
        if not file.endswith("y.npy"):
            continue
        elif file.replace("y.npy", ".csv") in filenames or file == "y.npy":
            file_paths.append(os.path.join(dataset_path, file))
            if constant_y:
                return file_paths  # Return after first path added if y is constant and valid

    return file_paths


def get_file_identifiers(file_list, except_n_parts=2):
    identifiers = []
    for file in file_list:
        identifier = "_".join(file.split("_")[:-except_n_parts])
        identifiers.append(identifier)

    return identifiers
