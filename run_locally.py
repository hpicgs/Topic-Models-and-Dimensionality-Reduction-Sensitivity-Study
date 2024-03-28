#!/bin/env python3

import os
from argparse import ArgumentParser

import pandas as pd


def get_arguments():
    parser = ArgumentParser(description="This scripts executes a parameter configuration locally")
    parser.add_argument('--offset', dest='offset', type=int, default=0,
                        help="Specifies the offset currently 0, 1, 2 is possible")
    args = parser.parse_args()
    return args.offset


def main():
    offset = get_arguments()
    df = pd.read_csv("special_parameters_emails.csv", header=None)
    parameters = df[df.columns[1]].to_numpy()

    for i in range(int(len(parameters) / 3)):
        cmd = "python3 " + parameters[i + offset]
        os.system(cmd)


if __name__ == "__main__":
    main()
