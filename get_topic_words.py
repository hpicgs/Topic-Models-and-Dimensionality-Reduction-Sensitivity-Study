import os
import pickle
from io import StringIO

import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel, ldamodel, Nmf
from sentence_transformers import SentenceTransformer


def get_model_word_list(cur_dir, model_base_name, model_type, num_words_per_topic=10):
    word_list = []
    full_model_path = os.path.join(cur_dir, model_base_name)

    if model_type == "lda":
        lda_model = ldamodel.LdaModel.load(full_model_path)
        n = lda_model.num_topics
        word_list = [lda_model.show_topic(i, topn=num_words_per_topic) for i in range(n)]
    elif model_type == "lsi":
        lsi_model = LsiModel.load(full_model_path)
        n = lsi_model.num_topics
        word_list = [lsi_model.show_topic(i, topn=num_words_per_topic) for i in range(n)]
    elif model_type == "nmf":
        nmf_model = Nmf.load(full_model_path)
        n = nmf_model.num_topics
        num_terms = nmf_model.id2word.num_terms
        dictionary = Dictionary.load(os.path.join(cur_dir, "dictionary_" + str(num_terms)))
        word_list = [nmf_model.get_topic_terms(i, topn=num_words_per_topic) for i in range(n)]
        word_list_tmp = []
        for topic_list in word_list:
            word_list_tmp.append([(dictionary.get(i), prob) for i, prob in topic_list])
        word_list = word_list_tmp
    elif "bert" in model_type:
        dir_name = os.path.dirname(full_model_path)
        base_name = os.path.basename(full_model_path)
        new_base_name = base_name.replace("model", "topics").replace(".npy", ".pkl")
        new_dir_name = os.path.join(dir_name, new_base_name)
        with open(new_dir_name, 'rb') as in_file:
            bert_topics = pickle.load(in_file)

        word_list = [bert_topic_list[:min(num_words_per_topic, len(bert_topic_list))] for bert_topic_list
                     in bert_topics.values()]

    return word_list


def get_all_model_words(base_path="models"):
    datasets = ["20_newsgroups", "lyrics", "seven_categories"]
    model_words_results_path = "model_words"
    os.makedirs(model_words_results_path, exist_ok=True)

    for cur_dir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".model") or "bert" in file:
                dataset = os.path.basename(os.path.normpath(cur_dir))
                if dataset not in datasets:
                    continue

                model_type = file.split("_")[0]
                word_list = get_model_word_list(cur_dir, file, model_type)
                if len(word_list) == 0:
                    continue

                os.makedirs(os.path.join(model_words_results_path, dataset), exist_ok=True)
                with open(os.path.join(model_words_results_path, dataset, file + ".txt"), "w+") as out_file:
                    out_file.write("# Top 10 words per topics for model: " + file + "\n")
                    out_file.write("# Model type is: " + model_type + "\n")
                    out_file.write("# For dataset: " + dataset + "\n\n\n")

                    out_file.write("topic_num,word,prob\n")
                    for i in range(len(word_list)):
                        for word, prob in word_list[i]:
                            out_file.write(str(i) + "," + str(word) + "," + str(prob) + "\n")


def transform_model_words_to_csv_format(base_path="model_words"):
    for cur_dir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):
                dataset = os.path.basename(os.path.normpath(cur_dir))
                cd_path = os.path.dirname(cur_dir)

                with open(os.path.join(cur_dir, file), 'r') as in_file:
                    content = in_file.readlines()[5:]

                df = pd.read_table(StringIO("\n".join(content)), sep=",")
                topic_nums = set(df["topic_num"].to_numpy())
                value_dict = {key: (df.loc[df["topic_num"] == key]["word"]).to_numpy() for key in topic_nums}
                df = pd.DataFrame.from_dict(value_dict)
                df.to_csv(os.path.join(cd_path, dataset + "_" + file.replace(".txt", ".csv")), sep=",", index=False)


def main():
    get_all_model_words()
    transform_model_words_to_csv_format()


if __name__ == "__main__":
    main()
