# Topic-Models-and-Dimensionality-Reduction-Sensitivity-Study

![Bildschirmfoto vom 2024-03-31 18-00-57](https://github.com/hpicgs/Topic-Models-and-Dimensionality-Reduction-Sensitivity-Study/assets/27726055/815312e6-443e-40dd-8311-2067094780c3)


# Benchmark Specs

This is the repository for the study "A Large-Scale Sensitivity Analysis on Latent Embeddings and Dimensionality Reductions for Text Spatializations".

It contains all execution and analysis modules.

## Datasets

* `20_newsgroups`
* `lyrics`
* `seven_categories`
* `reuters`
* `BBC news`
* `emails`

## Topic Models

* `VSM` of gensim
* `VSM` of gensim + tf-idf-Weighting
* `LSI` of gensim
* `LSI` of gensim + Linear Combined
* `LSI` of gensim + tf-idf-Weighting
* `LSI` of gensim + tf-idf-Weighting + Linear Combined
* `NMF` of gensim
* `NMF` of gensim + Linear Combined
* `NMF` of gensim + tf-idf-Weighting
* `NMF` of gensim + tf-idf-Weighting + Linear Combined
* `LDA` of gensim
* `LDA` of gensim + Linear Combined
* `Doc2Vec` of gensim
* `BERT_all_mpnet_base_v2` the best sentence embedding model according to https://www.sbert.net/docs/pretrained_models.html
* `BERT_all_distilroberta_v1` the second best sentence embedding model according to https://www.sbert.net/docs/pretrained_models.html

## Dimension Reduction Techniques

* `MDS` with one investigated hyperparameter `n_iter`
* `SOM` with two investigated hyperparameters `n` and `m`
* `t-SNE` with three investigated hyperparameters `learning_rate`, `n_iter`, and `perplexity`
* `UMAP` with two investigated hyperparameters `min_dist` and `n_neighbors`

## Quality Metrics

Local Stability Metrics
* Trustworthiness
* Continuity
* Mean Relative Rank Errors
* Local Continuity Meta-Criterion
* Label Preservation

Global Stability Metrics
* Pearson's Correlation
* Soearnab's Rank Correlation
* Cluster Ordering

Class Seperation Metric
* Absolute Difference Distance Consistency

Further scores:
* Silhouette Coefficient
* Rotation from Procrustes Analysis

## Analysis

* Heatmaps
* Binary Tests
* Correlation Tests

# Dev Setup (Ubuntu)

We have written our code for a Ubuntu 22.04 system.

## Dependencies

* openjdk-19-jdk
* ant
* python3-minimal
* python3.10-full
* python3-pip
* git
* RScript from r-base-core

Please install this via

```bash
> sudo apt install openjdk-19-jdk ant python3-minimal python3.10-full python3-pip git r-base-core
```

## Setup

```bash
> pip3 install numpy==1.23.5
> pip3 install -r requirements.txt
> python3 -m spacy download en_core_web_sm
> python3 -m spacy download en_core_web_lg
```

For postprocessing we also need ggplot2. Please install it via executing:

```bash
> R
> > install.packages("ggplot2")
``
and answering yes at every prompt.

## Run

### Parameter Generator

```bash
> python3 parameter_generator.py > parameters.csv
```

### ML Processing

Repeated calls to main.py using a wide range of parameters (see parameter generator) like this call:
```bash
> python3 main.py --perplexity_tsne 30 --n_iter_tsne 1000 --learning_rate auto --n_neighbors_umap 15 --min_dist_umap 0.1 --max_iter_mds 300 --dataset_name 20_newsgroups --topic_model lsi_tfidf --res_file_name ./results/20_newsgroups/results_perplexity_tsne_30_n_iter_tsne_1000_learning_rate_auto_n_neighbors_umap_15_min_dist_umap_0.1_max_iter_mds_300_dataset_name_20_newsgroups_topic_model_lsi_tfidf.csv
```
*For replication, we recommend you to (first) test a command like above. For running the full benchmark you will most probably need a computer cluster and about two weeks. Further calls can be produced by parameter_generator.py. See above.*

### Analysis

After finishing your runs, it is recommended to run parameter_generator.py again to see which job did finish and which not. The results_files then are copied to a directory called res_files_only, where the results can be collected. Thereafter, the results can be analyzed with one of the four analysis script: experiment_1_correlation_analysis.py, experiment_2_stability_hyperparameters, experiment_3_stability_randomness, experiment_4_stability_jitter.py. The script experiment_5_high_dimensional_similarity calculates the stability metrics for the standard corpus compared to a jittered corpus.

So the standard workflow for [path_to_results_of_dataset] with, e.g., 10 parallel jobs, is:

```bash
> python3 experiment_2_stability_hyperparameters.py [path_to_results_of_dataset]/random_seed_0/jitter_amount_0.0 --n_jobs 10
> python3 experiment_3_stability_randomness.py [path_to_results_of_dataset] --n_jobs 10
> python3 experiment_4_stability_jitter.py [path_to_results_of_dataset] --n_jobs 10
> python3 experiment_4_stability_jitter.py [path_to_results_of_dataset] --n_jobs 10 --random_seed 42
> python3 experiment_5_high_dimensional_similarity.py
```

This will create result csv files in your current working directory.
Please note that you should avoid [path_to_results_of_dataset] a final path seperator at the end of your path so that os.path.basename works properly. In addition, those analysis may take a while (depending on the dataset and your machine a couple of second up to 5 minutes per comparison).

In addition, to get a small report of covered analysis results, perform some sanity checks and get a large file of all results per experiment you may want to run:

```bash
> python3 perform_sanity_checks [path_to_experiment_results_from_calls_mentioned_above]
```

### Postprocessing

To execute this step you need ggplot2. Please refer to "Setup" for instructions how to install this package.
For aesthetics we used ggplot2 for postprocessing our scatter plots in the paper. This is done via the get_r_pdf_plots.py script. If you executed the call under "ML Processing", the according postprocessing call would be:

```bash
> python3 get_r_pdf_plots.py --base_path results/20_newsgroups --dataset_name 20_newsgroups
```

Afterwards, you can find the new scatter plots in the "Analysis_Visualization" directory.

# Docker Setup

## Build

```bash
> docker build . -t python-ml_batch:latest projections_benchmark --build-arg PLATFORM=amd64
```

## Run

```bash
> docker run python-ml_batch python3 main.py --perplexity_tsne 40 --n_iter_tsne 6000 --dataset_name reuters --res_file_name ./results/reuters/results_perplexity_tsne_40_n_iter_tsne_6000_dataset_name_reuters.csv
```
Additionally, mounts and workdir need to be set accordingly.

## Batch Run

```bash
> ./batch.sh
```

## Results

All of our results obtained by the method described above can be found in result_files/results_[experiment_name].zip
After unzipping, the results for each dataset may be found in results_experiment_[experiment_number]_[dataset_name].csv. 
In addition, a summary covering report is also placed in the respective experiment directory.

You may find all layout `.npy` (numpy-files) files under: https://drive.google.com/drive/folders/10pZX_hUKPYbOEynvOJ_RCM1frztO53Fw?usp=sharing 
