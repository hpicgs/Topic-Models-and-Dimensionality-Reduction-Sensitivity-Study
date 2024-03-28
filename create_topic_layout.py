import hashlib
import math
import os
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd

from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel, TfidfModel, ldamodel, Nmf, CoherenceModel
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer
from operator import add

import ldamallet
from commons import transform_sparse_model_to_dense_matrix, print_current_memory_usage, permute_columns

import umap
from sentence_transformers import SentenceTransformer
import hdbscan

from config import get_global_random_seed, get_bert_model_name

ENABLE_MODEL_SAVING = True


def convert_text_to_corpus(word_lists):
    dictionary = Dictionary(word_lists)
    corpus = [dictionary.doc2bow(text) for text in word_lists]
    return dictionary, corpus


def infer_paths_from_base_paths(base_path):
    model_path = base_path + ".model"
    dense_matrix_path = base_path + "_dense"
    linear_matrix_path = base_path + "_linear_combined"

    return model_path, dense_matrix_path, linear_matrix_path


def evaluate_topic_model(topic_model, corpus, dataset_base_path, n_topics, alpha_lda,
                         model_type, dictionary, word_lists=None):
    evaluation_file_path = os.path.join(dataset_base_path, "model_evaluation.csv")

    print("Attempt to evaluate topic model of type: " + model_type, flush=True)
    coherence_u_mass = get_coherence_u_mass_score(corpus, topic_model, dictionary)
    print_current_memory_usage()

    if word_lists is not None and "github_projects" not in dataset_base_path:
        coherence_c_v = get_coherence_c_v(topic_model, word_lists, dictionary)
        print_current_memory_usage()

        coherence_c_uci = get_coherence_c_uci(topic_model, word_lists, dictionary)
        print_current_memory_usage()

        coherence_c_npmi = get_coherence_c_npmi(topic_model, word_lists, dictionary)
        print_current_memory_usage()

        print("Successfully calculated all coherence scores", flush=True)
    else:
        print("Since words lists was not present or github_projects dataset was used,"
              " I can't calculate other coherence scores and will set -inf instead",
              flush=True)
        coherence_c_v = -np.inf
        coherence_c_uci = -np.inf
        coherence_c_npmi = -np.inf

    if "lda" in model_type:
        bound = topic_model.log_perplexity(chunk=corpus)
        perplexity = pow(2, -bound)  # See https://radimrehurek.com/gensim/models/ldamodel.html
        print("Successfully calculated perplexity", flush=True)
    else:
        print("Perplexity can only be calculated for LDA. Since present model is not LDA, I will set it to -1",
              flush=True)
        perplexity = -1

    print_current_memory_usage()
    values = [model_type, n_topics, alpha_lda, coherence_u_mass, coherence_c_v, coherence_c_uci,
              coherence_c_npmi, perplexity]
    if os.path.isfile(evaluation_file_path):
        try:
            frame = pd.read_csv(evaluation_file_path)
        except pd.errors.EmptyDataError:
            try:
                print("Failed to read evaluation file. I will go sleep for three seconds and retry.", flush=True)
                time.sleep(3)
                frame = pd.read_csv(evaluation_file_path)
            except pd.errors.EmptyDataError:
                print("Failed again to read evaluation file. I will skip evaluation and continue.", flush=True)
                return
        frame2 = pd.DataFrame([values], columns=["model_type", "n_topics", "alpha_lda",
                                                 "coherence_u_mass", "coherence_c_v", "coherence_c_uci",
                                                 "coherence_c_npmi",
                                                 "perplexity"])
        frame = frame.append(frame2, ignore_index=True)
    else:
        frame = pd.DataFrame(columns=["model_type", "n_topics", "alpha_lda",
                                      "coherence_u_mass", "coherence_c_v", "coherence_c_uci", "coherence_c_npmi",
                                      "perplexity"])
        frame.loc[0] = values

    frame.to_csv(evaluation_file_path, index=False)
    print("Finished evaluating model and wrote all results successfully.", flush=True)


def get_coherence_c_npmi(topic_model, word_lists, dictionary):
    coherence_model_c_npmi = CoherenceModel(model=topic_model, texts=word_lists, coherence='c_npmi',
                                            dictionary=dictionary)
    coherence_c_npmi = coherence_model_c_npmi.get_coherence()
    print("Successfully calculated coherence_c_npmi", flush=True)
    return coherence_c_npmi


def get_coherence_c_uci(topic_model, word_lists, dictionary):
    coherence_model_c_uci = CoherenceModel(model=topic_model, texts=word_lists, coherence='c_uci',
                                           dictionary=dictionary)
    coherence_c_uci = coherence_model_c_uci.get_coherence()
    print("Successfully calculated coherence_c_uci", flush=True)
    return coherence_c_uci


def get_coherence_c_v(topic_model, word_lists, dictionary):
    coherence_model_c_v = CoherenceModel(model=topic_model, texts=word_lists, coherence='c_v', dictionary=dictionary)
    coherence_c_v = coherence_model_c_v.get_coherence()
    print("Successfully calculated coherence_c_v", flush=True)
    return coherence_c_v


def get_coherence_u_mass_score(corpus, topic_model, dictionary):
    coherence_model_u_mass = CoherenceModel(model=topic_model, corpus=corpus, coherence='u_mass', dictionary=dictionary)
    coherence_u_mass = coherence_model_u_mass.get_coherence()
    print("Successfully calculated coherence_u_mass", flush=True)
    return coherence_u_mass


def create_lda_model(dictionary, corpus, n_topics, dataset_base_path, word_lists=None,
                     alpha="asymmetric", iterations=1000,
                     topic_threshold=0.0,
                     use_mallet=False, id2word=None, filter_indices=None,
                     disable_model_training=False, topic_model="", evaluate_model=True, suffix_permutation="",
                     suffix_change_columns="", jitter_amount=0):
    if filter_indices is None:
        filter_indices = []

    if jitter_amount > 0:
        base_path = os.path.join(dataset_base_path,
                                 "lda_" + str(n_topics) + "_" + str(alpha) + "_" + str(iterations) + "_" + str(
                                     topic_threshold) + "_" + str(len(dictionary)) + "_" + suffix_permutation + "_" +
                                 suffix_change_columns + "_j" + str(jitter_amount))
    else:
        base_path = os.path.join(dataset_base_path,
                                 "lda_" + str(n_topics) + "_" + str(alpha) + "_" + str(iterations) + "_" + str(
                                     topic_threshold) + "_" + str(len(dictionary)) + "_" + suffix_permutation + "_" +
                                 suffix_change_columns)

    model_path, dense_matrix_path, linear_matrix_path = infer_paths_from_base_paths(base_path)

    print("Looking for LDA model at: " + str(model_path), flush=True)
    if os.path.isfile(dense_matrix_path + ".npy") and os.path.isfile(linear_matrix_path + ".npy"):
        print("Already found requested LDA model. I will load and return early", flush=True)

        if topic_model == "lda" or topic_model == "":
            dense_vectors = np.load(dense_matrix_path + ".npy")

            if evaluate_model:
                topic_model_tmp = topic_model + "_" + suffix_permutation + "_" + suffix_change_columns
                model = ldamodel.LdaModel.load(model_path)
                evaluate_topic_model(model, corpus, dataset_base_path, n_topics, alpha, topic_model_tmp, dictionary,
                                     word_lists=word_lists)
        else:
            dense_vectors = None

        if topic_model == "lda_linear_combined" or topic_model == "":
            linear_combined_matrix = np.load(linear_matrix_path + ".npy")
        else:
            linear_combined_matrix = None

        return dense_vectors, linear_combined_matrix
    elif disable_model_training:
        return None, None

    topic_model += "_" + suffix_permutation + "_" + suffix_change_columns
    if use_mallet:
        model = ldamallet.LdaMallet(mallet_path=os.path.join(os.environ["MALLET_HOME"], "mallet"), corpus=corpus,
                                    id2word=dictionary, num_topics=n_topics, alpha=alpha, iterations=iterations,
                                    topic_threshold=topic_threshold, random_seed=get_global_random_seed())
        vectors = model[corpus]
        dense_vectors = transform_sparse_model_to_dense_matrix(vectors, n_topics)
        topic_term_matrix = model.word_topics / np.max(model.word_topics)
        linear_combined_matrix = np.matmul(dense_vectors, topic_term_matrix)
    else:
        if os.path.isfile(model_path):
            model = ldamodel.LdaModel.load(model_path)
        else:
            model = ldamodel.LdaModel(corpus=corpus, num_topics=n_topics, id2word=id2word, alpha=alpha,
                                      iterations=iterations, eta='symmetric')
        vectors = model[corpus]
        dense_vectors = transform_sparse_model_to_dense_matrix(vectors, n_topics)
        topic_term_matrix = model.get_topics()
        linear_combined_matrix = np.matmul(dense_vectors, topic_term_matrix)

    if evaluate_model:
        evaluate_topic_model(model, corpus, dataset_base_path, n_topics, alpha, topic_model, dictionary,
                             word_lists=word_lists)
    if dataset_base_path is not None and ENABLE_MODEL_SAVING:
        model.save(model_path)

    linear_combined_matrix = np.transpose(np.array([linear_combined_matrix[:, i]
                                                    for i in range(linear_combined_matrix.shape[1])
                                                    if i not in filter_indices]))

    if ENABLE_MODEL_SAVING:
        np.save(file=dense_matrix_path, arr=dense_vectors)
        np.save(file=linear_matrix_path, arr=linear_combined_matrix)
    return dense_vectors, linear_combined_matrix


def create_lsi_model(dictionary, corpus, n_topics, dataset_base_path, decay=1.0, onepass=True, power_iters=2,
                     extra_samples=100, min_density=0.001, filter_indices=None, tfidf_sparse=None,
                     disable_model_training=False, topic_model="", evaluate_model=True, word_lists=None,
                     suffix_permutation="", suffix_change_columns="", jitter_amount=0):
    if filter_indices is None:
        filter_indices = []

    if jitter_amount > 0:
        base_path = os.path.join(dataset_base_path,
                                 "lsi_" + str(n_topics) + "_" + str(decay) + "_" + str(onepass) + "_" + str(
                                     power_iters) + "_" + str(extra_samples) + "_" + str(len(dictionary)) +
                                 "_" + suffix_permutation + "_" + suffix_change_columns + "_j" + str(jitter_amount))
    else:
        base_path = os.path.join(dataset_base_path,
                                 "lsi_" + str(n_topics) + "_" + str(decay) + "_" + str(onepass) + "_" + str(
                                     power_iters) + "_" + str(extra_samples) + "_" + str(len(dictionary)) +
                                 "_" + suffix_permutation + "_" + suffix_change_columns)

    model_path, dense_matrix_path, linear_matrix_path = infer_paths_from_base_paths(base_path)

    base_path_tfidf = base_path.replace("lsi", "lsi_tfidf")
    model_path_tfidf, dense_matrix_path_tfidf, linear_matrix_path_tfidf = infer_paths_from_base_paths(base_path_tfidf)

    if os.path.isfile(dense_matrix_path + ".npy") and os.path.isfile(linear_matrix_path + ".npy") and \
            os.path.isfile(dense_matrix_path_tfidf + ".npy") and os.path.isfile(linear_matrix_path_tfidf + ".npy"):
        print("Already found requested LSI model. I will load and return early", flush=True)

        if topic_model == "lsi" or topic_model == "":
            dense_vectors = np.load(dense_matrix_path + ".npy")
        else:
            dense_vectors = None

        if topic_model == "lsi_linear_combined" or topic_model == "":
            linear_combined_matrix = np.load(linear_matrix_path + ".npy")
        else:
            linear_combined_matrix = None

        if topic_model == "lsi_tfidf" or topic_model == "":
            dense_vectors_tfidf = np.load(dense_matrix_path_tfidf + ".npy")
        else:
            dense_vectors_tfidf = None

        if topic_model == "lsi_tfidf_linear_combined" or topic_model == "":
            linear_combined_matrix_tfidf = np.load(linear_matrix_path_tfidf + ".npy")
        else:
            linear_combined_matrix_tfidf = None

        if evaluate_model:
            topic_model_tmp = topic_model + "_" + suffix_permutation + "_" + suffix_change_columns
            model = LsiModel.load(model_path)
            evaluate_topic_model(model, corpus, dataset_base_path, n_topics, "", topic_model_tmp, dictionary,
                                 word_lists=word_lists)

            model_tfidf = LsiModel.load(model_path_tfidf)
            evaluate_topic_model(model_tfidf, corpus, dataset_base_path, n_topics, "", topic_model_tmp,
                                 dictionary, word_lists=word_lists)

        return dense_vectors, linear_combined_matrix, dense_vectors_tfidf, linear_combined_matrix_tfidf
    elif disable_model_training:
        return None, None, None, None

    topic_model += "_" + suffix_permutation + "_" + suffix_change_columns
    if os.path.isfile(model_path):
        model = LsiModel.load(model_path)
    else:
        model = LsiModel(corpus, id2word=dictionary, num_topics=n_topics, decay=decay, onepass=onepass,
                         power_iters=power_iters, extra_samples=extra_samples, random_seed=get_global_random_seed())
        if dataset_base_path is not None and ENABLE_MODEL_SAVING:
            model.save(model_path)

    if evaluate_model:
        evaluate_topic_model(model, corpus, dataset_base_path, n_topics, "", topic_model, dictionary,
                             word_lists=word_lists)
    vectors = model[corpus]

    topic_term_matrix = model.get_topics()
    dense_vectors = transform_sparse_model_to_dense_matrix(vectors, n_topics)
    linear_combined_matrix = np.matmul(dense_vectors, topic_term_matrix)
    linear_combined_matrix = np.transpose(np.array([linear_combined_matrix[:, i]
                                                    for i in range(linear_combined_matrix.shape[1])
                                                    if i not in filter_indices]))

    if os.path.isfile(model_path_tfidf):
        model = LsiModel.load(model_path_tfidf)
    else:
        model = LsiModel(tfidf_sparse, id2word=dictionary, num_topics=n_topics, decay=decay, onepass=onepass,
                         power_iters=power_iters, extra_samples=extra_samples, random_seed=get_global_random_seed())

    if model_path_tfidf is not None and ENABLE_MODEL_SAVING:
        model.save(model_path_tfidf)

    try:
        if evaluate_model:
            evaluate_topic_model(model, corpus, dataset_base_path, n_topics, "", topic_model, dictionary,
                                 word_lists=word_lists)
    except ValueError as e:
        warnings.warn(str(e))
        return dense_vectors, linear_combined_matrix, None, None
    vectors = model[tfidf_sparse]

    topic_term_matrix_tfidf = model.get_topics()
    dense_vectors_tfidf = transform_sparse_model_to_dense_matrix(vectors, n_topics)
    linear_combined_matrix_tfidf = np.matmul(dense_vectors_tfidf, topic_term_matrix_tfidf)
    linear_combined_matrix_tfidf = np.transpose(np.array([linear_combined_matrix_tfidf[:, i]
                                                          for i in range(linear_combined_matrix_tfidf.shape[1])
                                                          if i not in filter_indices]))

    if ENABLE_MODEL_SAVING:
        np.save(file=dense_matrix_path, arr=dense_vectors)
        np.save(file=linear_matrix_path, arr=linear_combined_matrix)
        np.save(file=dense_matrix_path_tfidf, arr=dense_vectors_tfidf)
        np.save(file=linear_matrix_path_tfidf, arr=linear_combined_matrix_tfidf)
    return dense_vectors, linear_combined_matrix, dense_vectors_tfidf, linear_combined_matrix_tfidf


def get_sparse_tfidf_model(dataset_base_bath, corpus, dictionary, min_density, suffix_permutation="",
                           suffix_change_columns="", jitter_amount=0):
    _, model_path_sparse = get_tfidf_model_names(dataset_base_bath, dictionary, min_density,
                                                 suffix_permutation, suffix_change_columns, jitter_amount=jitter_amount)
    if os.path.isfile(model_path_sparse + ".npy"):
        print("Already found requested sparse Tfidf model. I will load and return early", flush=True)
        try:
            sparse_matrix = np.load(model_path_sparse + ".npy", allow_pickle=True)
        except:
            print("Failed to load model. I will fall back on training it.", flush=True)
            model = TfidfModel(corpus)
            sparse_matrix = model[corpus]
            if ENABLE_MODEL_SAVING:
                np.save(file=model_path_sparse, arr=sparse_matrix, allow_pickle=True)
        return sparse_matrix
    else:
        print("I couldn't find requested sparse Tfidf model. Therefore, I will train it", flush=True)
        model = TfidfModel(corpus)
        sparse_matrix = model[corpus]
        if ENABLE_MODEL_SAVING:
            np.save(file=model_path_sparse, arr=sparse_matrix, allow_pickle=True)
            print("Saved sparse Tfidf model. Please note that I did not save the original Tfidf model. Please call"
                  " create_tfidf_model for this purpose.", flush=True)
        return sparse_matrix


def create_tfidf_model(dictionary, corpus, dataset_base_path, min_density=0.001, filter_indices=None,
                       disable_model_training=False, suffix_permutation="", suffix_change_columns="",
                       jitter_amount=0):
    if filter_indices is None:
        filter_indices = []
    model_path, model_path_sparse = get_tfidf_model_names(dataset_base_path, dictionary, min_density,
                                                          suffix_permutation=suffix_permutation,
                                                          suffix_change_columns=suffix_change_columns,
                                                          jitter_amount=jitter_amount)

    if os.path.isfile(model_path + ".npy") and os.path.isfile(model_path_sparse + ".npy"):
        print("Already found requested Tfidf model. I will load and return early", flush=True)
        dense_matrix = np.load(model_path + ".npy")
        sparse_matrix = np.load(model_path_sparse + ".npy", allow_pickle=True)
        return dense_matrix, sparse_matrix
    elif disable_model_training:
        return None, get_sparse_tfidf_model(dataset_base_bath=dataset_base_path, corpus=corpus, dictionary=dictionary,
                                            min_density=min_density)

    model = TfidfModel(corpus)
    sparse_matrix = model[corpus]
    dense_matrix = transform_sparse_model_to_dense_matrix(sparse_matrix, len(dictionary))
    dense_matrix = np.transpose(np.array([dense_matrix[:, i]
                                          for i in range(dense_matrix.shape[1])
                                          if i not in filter_indices]))

    if ENABLE_MODEL_SAVING:
        np.save(file=model_path, arr=dense_matrix)
        np.save(file=model_path_sparse, arr=sparse_matrix, allow_pickle=True)
    return dense_matrix, sparse_matrix


def get_tfidf_model_names(dataset_base_path, dictionary, min_density, suffix_permutation="", suffix_change_columns="",
                          jitter_amount=0):
    model_path = os.path.join(dataset_base_path, "tfidf_model_" + str(min_density) + "_" + str(len(dictionary)) + "_" +
                              suffix_permutation + "_" + suffix_change_columns + "_j" + str(jitter_amount))
    model_path_sparse = model_path.replace("tfidf_model", "tfidf_model_sparse")
    return model_path, model_path_sparse


def filter_sparse_columns(dense_matrix, min_density):
    print("Matrix shape before filtering with density " + str(min_density) + ": " + str(dense_matrix.shape), flush=True)

    res_matrix = []
    filtered_indices = []
    for i in range(dense_matrix.shape[1]):
        column = dense_matrix[:, i]
        positive_values = sum([not math.isclose(val, 0.0) for val in column])
        share = positive_values / len(column)
        if share >= min_density:
            res_matrix.append(column)
        else:
            filtered_indices.append(i)

    dense_matrix = np.transpose(np.array(res_matrix))
    print("Matrix shape after filtering with density " + str(min_density) + ": " + str(dense_matrix.shape), flush=True)
    return dense_matrix, np.array(filtered_indices)


def create_bow_model(dictionary, corpus, dataset_base_path, min_density=0.001, suffix_permutation="",
                     suffix_change_columns="", jitter_amount=0):
    if jitter_amount > 0:
        model_path = os.path.join(dataset_base_path,
                                  "bow_model_" + str(min_density) + "_" + str(len(dictionary)) +
                                  "_" + suffix_permutation + "_" + suffix_change_columns + "_j" +
                                  str(jitter_amount))
    else:
        model_path = os.path.join(dataset_base_path, "bow_model_" + str(min_density) + "_" + str(len(dictionary)) +
                                  "_" + suffix_permutation + "_" + suffix_change_columns)
    filtered_indices_path = os.path.join(dataset_base_path, "filtered_indices_" + str(min_density) + "_"
                                         + str(len(dictionary)))
    if os.path.isfile(model_path + ".npy") and os.path.isfile(filtered_indices_path + ".npy"):
        print("Already found requested BOW model. I will load and return early", flush=True)
        dense_matrix = np.load(model_path + ".npy")
        filtered_indices = np.load(filtered_indices_path + ".npy")
        return dense_matrix, filtered_indices

    dense_matrix = transform_sparse_model_to_dense_matrix(corpus, len(dictionary))
    dense_matrix, filtered_indices = filter_sparse_columns(dense_matrix, min_density=min_density)

    if ENABLE_MODEL_SAVING:
        np.save(file=model_path, arr=dense_matrix)
        np.save(file=filtered_indices_path, arr=filtered_indices)
    return dense_matrix, filtered_indices


def create_nmf_layout(dictionary, corpus, dataset_base_path, filter_indices, min_density, tfidf_sparse, n_topics,
                      disable_model_training=False, topic_model="", evaluate_model=True, word_lists=None,
                      suffix_permutation="", suffix_change_columns="", jitter_amount=0):
    if filter_indices is None:
        filter_indices = []

    if jitter_amount > 0:
        base_path = os.path.join(dataset_base_path,
                                 "nmf_" + str(n_topics) + "_" + str(len(dictionary)) + "_" + suffix_permutation + "_" +
                                 suffix_change_columns + "_j" + str(jitter_amount))
    else:
        base_path = os.path.join(dataset_base_path,
                                 "nmf_" + str(n_topics) + "_" + str(len(dictionary)) + "_" + suffix_permutation + "_" +
                                 suffix_change_columns)
    model_path, dense_matrix_path, linear_matrix_path = infer_paths_from_base_paths(base_path)

    base_path_tfidf = base_path.replace("nmf", "nmf_tfidf")
    model_path_tfidf, dense_matrix_path_tfidf, linear_matrix_path_tfidf = infer_paths_from_base_paths(base_path_tfidf)

    if os.path.isfile(dense_matrix_path + ".npy") and os.path.isfile(linear_matrix_path + ".npy") and \
            os.path.isfile(dense_matrix_path_tfidf + ".npy") and os.path.isfile(linear_matrix_path_tfidf + ".npy"):
        print("Already found requested NMF model. I will load and return early", flush=True)

        if topic_model == "nmf" or topic_model == "":
            dense_vectors = np.load(dense_matrix_path + ".npy")
        else:
            dense_vectors = None

        if topic_model == "nmf_linear_combined" or topic_model == "":
            linear_combined_matrix = np.load(linear_matrix_path + ".npy")
        else:
            linear_combined_matrix = None

        if topic_model == "nmf_tfidf" or topic_model == "":
            dense_vectors_tfidf = np.load(dense_matrix_path_tfidf + ".npy")
        else:
            dense_vectors_tfidf = None

        if topic_model == "nmf_tfidf_linear_combined" or topic_model == "":
            linear_combined_matrix_tfidf = np.load(linear_matrix_path_tfidf + ".npy")
        else:
            linear_combined_matrix_tfidf = None

        if evaluate_model:
            topic_model_tmp = topic_model + "_" + suffix_permutation + "_" + suffix_change_columns
            model = Nmf.load(model_path)
            evaluate_topic_model(model, corpus, dataset_base_path, n_topics, "", topic_model_tmp, dictionary,
                                 word_lists=word_lists)

            model_tfidf = Nmf.load(model_path_tfidf)
            evaluate_topic_model(model_tfidf, corpus, dataset_base_path, n_topics, "", topic_model_tmp,
                                 dictionary, word_lists=word_lists)

        return dense_vectors, linear_combined_matrix, dense_vectors_tfidf, linear_combined_matrix_tfidf
    elif disable_model_training:
        return None, None, None, None

    if os.path.isfile(model_path):
        model = Nmf.load(model_path)
    else:
        model = Nmf(corpus=corpus, num_topics=n_topics)

    topic_model += "_" + suffix_permutation + "_" + suffix_change_columns
    if evaluate_model:
        evaluate_topic_model(model, corpus, dataset_base_path, n_topics, "", topic_model, dictionary,
                             word_lists=word_lists)
    vectors = model[corpus]
    dense_vectors = transform_sparse_model_to_dense_matrix(vectors, n_topics)
    topic_term_matrix = model.get_topics()
    linear_combined_matrix = np.matmul(dense_vectors, topic_term_matrix)
    if dataset_base_path is not None and ENABLE_MODEL_SAVING:
        model.save(model_path)

    linear_combined_matrix = np.transpose(np.array([linear_combined_matrix[:, i]
                                                    for i in range(linear_combined_matrix.shape[1])
                                                    if i not in filter_indices]))

    if ENABLE_MODEL_SAVING:
        np.save(file=dense_matrix_path, arr=dense_vectors)
        np.save(file=linear_matrix_path, arr=linear_combined_matrix)

    if os.path.isfile(model_path_tfidf):
        model = Nmf.load(model_path_tfidf)
    else:
        model = Nmf(corpus=tfidf_sparse, num_topics=n_topics)

    if model_path_tfidf is not None and ENABLE_MODEL_SAVING:
        model.save(model_path_tfidf)

    try:
        if evaluate_model:
            evaluate_topic_model(model, corpus, dataset_base_path, n_topics, "", topic_model, dictionary,
                                 word_lists=word_lists)
    except ValueError as e:
        warnings.warn(str(e))
        return dense_vectors, linear_combined_matrix, None, None
    vectors = model[tfidf_sparse]
    dense_vectors_tfidf = transform_sparse_model_to_dense_matrix(vectors, n_topics)
    topic_term_matrix = model.get_topics()
    linear_combined_matrix_tfidf = np.matmul(dense_vectors_tfidf, topic_term_matrix)

    linear_combined_matrix = np.transpose(np.array([linear_combined_matrix_tfidf[:, i]
                                                    for i in range(linear_combined_matrix_tfidf.shape[1])
                                                    if i not in filter_indices]))

    if ENABLE_MODEL_SAVING:
        np.save(file=dense_matrix_path_tfidf, arr=dense_vectors_tfidf)
        np.save(file=linear_matrix_path_tfidf, arr=linear_combined_matrix_tfidf)
    return dense_vectors, linear_combined_matrix, dense_vectors_tfidf, linear_combined_matrix_tfidf


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes


def create_bert_layout(words_lists, n_categories, dataset_base_path, dictionary, disable_model_training=False,
                       suffix_permutation="", suffix_change_columns="", jitter_amount=0):
    if jitter_amount > 0:
        model_path = os.path.join(dataset_base_path,
                                  "bert_model_" + str(n_categories) + "_" + str(len(dictionary)) + "_" +
                                  suffix_permutation + "_" + suffix_change_columns + "_j" + str(jitter_amount))
    else:
        model_path = os.path.join(dataset_base_path,
                                  "bert_model_" + str(n_categories) + "_" + str(len(dictionary)) + "_" +
                                  suffix_permutation + "_" + suffix_change_columns)
    topics_path = model_path.replace("bert_model", "bert_topics") + ".pkl"
    topic_ids_path = topics_path.replace("bert_topics", "bert_topics_id")

    if os.path.isfile(model_path + ".npy") and os.path.isfile(topics_path) and os.path.isfile(topic_ids_path + ".npy"):
        print("Already found requested BERT model. I will load and return early", flush=True)
        embeddings = np.load(model_path + ".npy")
        return embeddings
    elif disable_model_training:
        return None

    corpus = [" ".join(word_list) for word_list in words_lists]
    model = SentenceTransformer(get_bert_model_name())
    embeddings = model.encode(corpus, show_progress_bar=True)

    if ENABLE_MODEL_SAVING:
        np.save(file=model_path, arr=embeddings)

    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=5,
                                metric='cosine').fit_transform(embeddings)

    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(umap_embeddings)

    topic_ids = cluster.labels_
    if ENABLE_MODEL_SAVING:
        np.save(file=topic_ids_path, arr=topic_ids)

    docs_df = pd.DataFrame(corpus, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(corpus))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    with open(topics_path, "wb+") as topics_file:
        pickle.dump(top_n_words, topics_file)

    return embeddings


def drop_columns_corpus(corpus, columns_to_drop):
    for i, doc in enumerate(corpus):
        new_doc = []
        for j, el in enumerate(doc):
            if el[0] not in columns_to_drop:
                new_doc.append((el[0], el[1]))
        corpus[i] = new_doc


def scale_columns_corpus(corpus, indices):
    for i, doc in enumerate(corpus):
        for j, el in enumerate(doc):
            if el[0] in indices:
                corpus[i][j] = (el[0], 5 * el[1])


def scale_rows_corpus(corpus, indices):
    for i in indices:
        for j, el in enumerate(corpus[i]):
            corpus[i][j] = (el[0], 2 * el[1])


def get_jittered_corpus(corpus, jitter_amount, model_path):
    jitter_corpus_file_path = os.path.join(model_path, "corpus_jittered_" + str(len(corpus))
                                           + "_" + str(jitter_amount) + ".pkl")
    if os.path.isfile(jitter_corpus_file_path):
        print("Found corpus with requested amount of jitter (" + str(jitter_amount) + ")."
                                                                                      " I will load and return early!",
              flush=True)
        with open(jitter_corpus_file_path, 'rb') as in_file:
            corpus = pickle.load(in_file)
    else:
        jittered_corpus = add_jitter_to_corpus(corpus, jitter_amount)
        with open(jitter_corpus_file_path, 'wb+') as out_file:
            pickle.dump(obj=jittered_corpus, file=out_file)
        corpus = jittered_corpus
    return corpus


def add_jitter_to_document(document, jitter_amount):
    freqs = [freq for word_id, freq in document]
    noise = [max(0, int(freq * (1 + random.uniform(-jitter_amount, jitter_amount)))) for freq in freqs]
    noisy_freqs = list(map(add, freqs, noise))
    document = [(document[i][0], noisy_freqs[i]) for i in range(len(document))]
    return document


def add_jitter_to_corpus(corpus, jitter_amount):
    print("Requested jittering for corpus with amount " + str(jitter_amount), flush=True)
    start = time.time()
    new_corpus = [add_jitter_to_document(document, jitter_amount) for document in corpus]
    print("Time elapsed for creating jittered corpus " + str(time.time() - start), flush=True)
    return new_corpus


def create_doc2vec_layout(words_lists, dataset_base_path, disable_model_training, suffix_permutation,
                          suffix_change_columns, jitter_amount, n_topics):
    if jitter_amount > 0:
        model_path = os.path.join(dataset_base_path,
                                  "doc2vec_model_" +
                                  suffix_permutation + "_" + suffix_change_columns + "_j" + str(jitter_amount))
    else:
        model_path = os.path.join(dataset_base_path,
                                  "doc2vec_model_" + suffix_permutation + "_" + suffix_change_columns)

    if os.path.isfile(model_path + ".npy"):
        print("Already found requested BERT model. I will load and return early", flush=True)
        embeddings = np.load(model_path + ".npy")
        return embeddings
    elif disable_model_training:
        return None


    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(words_lists)]
    model = Doc2Vec(documents, vector_size=n_topics, workers=3)  # vector_size is topic_num
    vectors = model.dv.vectors

    model.save(model_path)
    np.save(file=str(model_path), arr=vectors)

    return vectors


def create_all_topic_models(word_lists, model_path, hyperparameters_lda=None, hyperparameters_lsi=None,
                            hyperparameters_nmf=None, hyperparameters_bert=None, hyperparameters_doc2vec=None,
                            min_density=0.001,
                            disable_model_training=False, topic_model="", evaluate_model=True,
                            scale_columns_shuffle_times_bow=0, scale_rows_shuffle_times_bow=0, drop_ratio_tm=0,
                            drop_ratio_tm_shuffle_times=0, suffix_permutation="", jitter_amount=0):
    if hyperparameters_lsi is None:
        hyperparameters_lsi = {'n_topics': 50}
    if hyperparameters_lda is None:
        hyperparameters_lda = {'n_topics': 50}
    if hyperparameters_nmf is None:
        hyperparameters_nmf = {'n_topics': 50}
    if hyperparameters_bert is None:
        hyperparameters_bert = {'n_categories': 5}
    os.makedirs(model_path, exist_ok=True)

    dictionary, corpus = convert_text_to_corpus(word_lists)
    dictionary.save(os.path.join(model_path, "dictionary" + "_" + str(len(dictionary))))
    n_columns = max(list(dictionary.keys()))

    corpus, suffix_change_columns = permute_columns(corpus, drop_ratio_tm, drop_ratio_tm_shuffle_times, n_columns,
                                                    scale_columns_shuffle_times_bow, scale_rows_shuffle_times_bow,
                                                    scale_columns_corpus, scale_rows_corpus, drop_columns_corpus)

    if jitter_amount > 0:
        corpus = get_jittered_corpus(corpus, jitter_amount, model_path)

    if (topic_model == "bow" or topic_model == "") and not disable_model_training:
        start = time.time()
        bow_dense, filtered_indices = create_bow_model(dictionary=dictionary, corpus=corpus,
                                                       dataset_base_path=model_path,
                                                       min_density=min_density,
                                                       suffix_permutation=suffix_permutation,
                                                       suffix_change_columns=suffix_change_columns,
                                                       jitter_amount=jitter_amount)
        print("Created BOW model!", flush=True)
        print("Elapsed time for getting BOW model: " + str(time.time() - start), flush=True)
    else:
        bow_dense, filtered_indices = None, None

    if (topic_model == "tfidf" or topic_model == "") and not disable_model_training:
        start = time.time()
        tfidf_dense, tfidf_sparse = create_tfidf_model(dictionary=dictionary, corpus=corpus,
                                                       dataset_base_path=model_path,
                                                       min_density=min_density, filter_indices=filtered_indices,
                                                       disable_model_training=disable_model_training,
                                                       suffix_permutation=suffix_permutation,
                                                       suffix_change_columns=suffix_change_columns,
                                                       jitter_amount=jitter_amount)
        print("Created Tfidf model!", flush=True)
        print("Elapsed time for getting Tfidf model: " + str(time.time() - start), flush=True)
    else:
        tfidf_dense = None
        tfidf_sparse = get_sparse_tfidf_model(dictionary=dictionary, corpus=corpus, dataset_base_bath=model_path,
                                              min_density=min_density, suffix_permutation=suffix_permutation,
                                              suffix_change_columns=suffix_change_columns, jitter_amount=jitter_amount)

    if ("lda" in topic_model or topic_model == "") and not disable_model_training:
        start = time.time()
        lda_dense, lda_linear_combined = create_lda_model(dictionary=dictionary, corpus=corpus,
                                                          dataset_base_path=model_path, id2word=dictionary,
                                                          filter_indices=filtered_indices,
                                                          disable_model_training=disable_model_training,
                                                          topic_model=topic_model, word_lists=word_lists,
                                                          evaluate_model=evaluate_model,
                                                          suffix_permutation=suffix_permutation,
                                                          suffix_change_columns=suffix_change_columns,
                                                          jitter_amount=jitter_amount,
                                                          **hyperparameters_lda)
        print("Created LDA model!", flush=True)
        print("Elapsed time for getting LDA model: " + str(time.time() - start), flush=True)
    else:
        lda_dense, lda_linear_combined = None, None

    if ("lsi" in topic_model or topic_model == "") and not disable_model_training:
        start = time.time()
        lsi_dense, lsi_linear_combined, lsi_dense_tfidf, lsi_linear_combined_tfidf = create_lsi_model(
            dictionary=dictionary,
            corpus=corpus,
            dataset_base_path=model_path,
            filter_indices=filtered_indices,
            min_density=min_density,
            tfidf_sparse=tfidf_sparse,
            disable_model_training=disable_model_training,
            topic_model=topic_model, word_lists=word_lists, evaluate_model=evaluate_model,
            suffix_permutation=suffix_permutation, suffix_change_columns=suffix_change_columns,
            jitter_amount=jitter_amount,
            **hyperparameters_lsi)
        print("Created LSI model!", flush=True)
        print("Elapsed time for getting LSI model: " + str(time.time() - start), flush=True)
    else:
        lsi_dense, lsi_linear_combined, lsi_dense_tfidf, lsi_linear_combined_tfidf = None, None, None, None

    if ("nmf" in topic_model or topic_model == "") and not disable_model_training:
        start = time.time()
        nmf_dense, nmf_linear_combined, nmf_tfidf_dense, nmf_tfidf_linear_combined = create_nmf_layout(
            dictionary=dictionary, corpus=corpus,
            dataset_base_path=model_path, filter_indices=filtered_indices,
            min_density=min_density, tfidf_sparse=tfidf_sparse, disable_model_training=disable_model_training,
            topic_model=topic_model, word_lists=word_lists, evaluate_model=evaluate_model,
            suffix_permutation=suffix_permutation, suffix_change_columns=suffix_change_columns,
            jitter_amount=jitter_amount,
            **hyperparameters_nmf)
        print("Created NMF model!", flush=True)
        print("Elapsed time for getting NMF model: " + str(time.time() - start), flush=True)
    else:
        nmf_dense, nmf_linear_combined, nmf_tfidf_dense, nmf_tfidf_linear_combined = None, None, None, None

    if (topic_model == "bert" or topic_model == "") and not disable_model_training:
        start = time.time()
        bert_dense = create_bert_layout(words_lists=word_lists, dictionary=dictionary, dataset_base_path=model_path,
                                        disable_model_training=disable_model_training,
                                        n_categories=hyperparameters_bert["n_categories"],
                                        suffix_permutation=suffix_permutation,
                                        suffix_change_columns=suffix_change_columns,
                                        jitter_amount=jitter_amount)
        print("Created BERT model!", flush=True)
        print("Elapsed time for getting BERT model: " + str(time.time() - start), flush=True)
    else:
        bert_dense = None

    if (topic_model == "doc2vec" or topic_model == "") and not disable_model_training:
        start = time.time()
        doc2vec_dense = create_doc2vec_layout(words_lists=word_lists, dataset_base_path=model_path,
                                              disable_model_training=disable_model_training,
                                              suffix_permutation=suffix_permutation,
                                              suffix_change_columns=suffix_change_columns,
                                              jitter_amount=jitter_amount, n_topics=hyperparameters_doc2vec["n_topics"])

        print("Created doc2vec model!", flush=True)
        print("Elapsed time for getting doc2vec model: " + str(time.time() - start), flush=True)
    else:
        doc2vec_dense = None

    parameter_string_lda = "_".join([str(item[0]) + "_" + str(item[1]) for item in hyperparameters_lda.items()])

    parameter_string_lsi = "_".join([str(item[0]) + "_" + str(item[1]) for item in hyperparameters_lsi.items()])

    parameter_string_nmf = "_".join([str(item[0]) + "_" + str(item[1]) for item in hyperparameters_nmf.items()])

    parameter_string_bert = "_".join([str(item[0]) + "_" + str(item[1]) for item in hyperparameters_bert.items()])

    parameter_string_doc2vec = "_".join([str(item[0]) + "_" + str(item[1]) for item in hyperparameters_doc2vec.items()])

    models = {'bow': bow_dense,
              'tfidf': tfidf_dense,
              'lda_' + parameter_string_lda: lda_dense,
              'lda_linear_combined_' + parameter_string_lda: lda_linear_combined,
              'lsi_' + parameter_string_lsi: lsi_dense,
              'lsi_linear_combined_' + parameter_string_lsi: lsi_linear_combined,
              'lsi_tfidf_' + parameter_string_lsi: lsi_dense_tfidf,
              'lsi_linear_combined_tfidf_' + parameter_string_lsi: lsi_linear_combined_tfidf,
              'nmf_' + parameter_string_nmf: nmf_dense,
              'nmf_linear_combined_' + parameter_string_nmf: nmf_linear_combined,
              'nmf_tfidf_' + parameter_string_nmf: nmf_tfidf_dense,
              'nmf_linear_combined_tfidf_' + parameter_string_nmf: nmf_tfidf_linear_combined,
              'bert_' + parameter_string_bert: bert_dense,
              'doc2vec_' + parameter_string_doc2vec: doc2vec_dense}
    models = {key: value for key, value in models.items() if value is not None}

    return models, str(hashlib.sha256(str(corpus).encode()).hexdigest())[:15]
