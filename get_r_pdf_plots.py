import os
import shutil
import subprocess
from argparse import ArgumentParser
from get_selected_results import create_csv_from_npy_file


def main():
    base_dir, dataset_name = get_base_path()
    base_path = os.path.join(os.getcwd(), base_dir)
    r_base_path = os.path.join(os.getcwd(), "Analysis_Visualization")
    r_script = "Scatterplot.R"

    get_pdfs_from_npy_files(base_path, r_base_path, r_script, dataset_name)


def get_base_path():
    parser = ArgumentParser(description="These scripts create all layouts for all datasets for one hyperparameter"
                                        "configuration. Please note, that you will handle the hyperparameter loop"
                                        "from outside. Compare to sbatch.")
    parser.add_argument('--base_path', dest='base_path', type=str, default=os.getcwd(),
                        help="The base path where the data for ggPlot is located. Please only give the"
                             "directory path from root (location of execution of the script). Defaults to CWD.")
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, default="20_newsgroups",
                        help="Dataset name. Available options are: 20_newsgroups, reuters, emails, github_projects,"
                             " seven_categories. Defaults to 20_newsgroups.")
    args = parser.parse_args()
    return args.base_path, args.dataset_name


def get_pdfs_from_npy_files(base_path, r_base_path, r_script, dataset_name, force_recreation_of_csv_files=False):
    create_csv_from_npy_file(dataset_name=dataset_name, selected_res_path=base_path)

    for file in os.listdir(base_path):
        if not file.endswith(".csv") or "results" in file:
            continue

        r_path = os.path.join(r_base_path, file)
        shutil.copy(os.path.join(base_path, file), r_path)
        subprocess.check_call(["Rscript", r_script, r_path], cwd=r_base_path)


if __name__ == "__main__":
    main()
