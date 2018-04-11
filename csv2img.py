import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Python script for converting a csv of values into images formatted for
# import into Inception-v3.
#
# Originally for scRNA-seq data, however this can be implemented on
# anything with the following structure (in csv format):
#
# cols: variables
# rows: entries
#
# Ex:
#       Gene1   Gene2   ...  GeneN  ID
# Smpl1  1.3     2.3    ...    0.0  TypeA
# Smpl2  0.0     1.6    ...    1.1  TypeB
#  ...   ...     ...    ...    ...  ...
# SmplN  3.1     0.0    ...    1.4  TypeA
#
# That will be converted into the following file structure:
#
# -output_dir
#   -TypeA
#       -1.jpg
#       -N.jpg
#   -TypeB
#       -2.jpg
#
# USAGE:
#   python3 csv2img.py \
#       --csv=/path/to/input.csv \
#       --margin=1 \                    # 0: cols, 1: rows; DEFAULT=rows
#       --output_dir=path/to/output_dir/ \
#       --is_validation=True           # if True: Do not look for ID col


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv',
                        help='full path to input csv.')
    parser.add_argument('--margin',
                        default=1,
                        help='1 for row:entry format, 0 for transposed.')
    parser.add_argument('--output_dir',
                        help='full path for root dir to store jpg files.')
    parser.add_argument('--is_validation',
                        default=True,
                        help='If True, there will be no ID column '
                             'expected.')
    args = parser.parse_args()

    if args.csv:
        csv_fname = args.csv
    else:
        raise ValueError('`--csv` required')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        raise ValueError('`--output_dir` required')

    margin = args.margin
    is_validation = args.is_validation

    return csv_fname, output_dir, margin, is_validation


def input_csv_data(csv_fname, margin, is_validation):
    """Function to import csv files. Returns values and if not
validation: ID labels"""
    csv = pd.read_csv(
        filepath_or_buffer=csv_fname,
        header=None  # Expecting input to not have colnames.
    )
    if is_validation is False:
        if margin == 1:
            labels = csv.iloc[:, -1]
            csv = csv.iloc[:, :-1]
            return csv, labels

        elif margin == 0:  # Needs to be corrected
            labels = csv.iloc[-1, :]
            # csv = csv.iloc[:-1, :]
            return csv, labels

    elif is_validation is True:
        entries = csv.shape[0] # Needs to be fixed with margin
        labels = pd.Series(['VAL'] * entries)  # Create VAL labels
        # to make sure
        # all values are stored in the same dir.
        return csv, labels


def check_for_output_dir(output_dir, labels):
    """Checks for presence of files in output_dir.  Makes new files
    if necessary"""
    unique_labels = labels.unique()

    new_dir_count = 0
    for label in unique_labels:
        path = os.path.join(output_dir, label)
        if os.path.exists(path):
            print('\tOutput directory for {} exists at: {} '.format(
                label, path))
        else:
            os.mkdir(path)
            new_dir_count += 1

    print('\n\tNew Directories Generated: {}'.format(new_dir_count))

def create_jpg_imgs(csv, labels, margin, output_dir):
    """Generates jpg file outputs"""

    # need to work margin in here.
    idx = 0
    new_img_count = int(len(labels))
    for label in labels:
        path = os.path.join(output_dir, label, str(idx) + '.jpg')

        if os.path.exists(path):
            print('\tOutput path for {} at row index {} exists '
                  'at: {}'.format(label, idx, path))
            new_img_count -= 1

        img = np.array(csv.iloc[idx, :])  # turn into numpy array

        # Figure out size to make square image
        side_length = np.sqrt(len(img))
        side_length = int(np.ceil(side_length))
        new_len = np.square(side_length)
        padding = int(new_len - len(img))

        new_img = np.append(img, [[np.min(img)] * padding])

        # Reshape to square
        new_img = new_img.reshape((side_length, side_length))

        # Save to destination
        plt.imsave(fname=path, arr=new_img)

        idx += 1


    print('\n\tNew Images Generated: {}'.format(new_img_count))


def main():
    csv_fname, output_dir, margin, is_validation = parse_arguments()

    print('\nImporting Data...')

    csv, labels = input_csv_data(
        csv_fname=csv_fname,
        margin=margin,
        is_validation=is_validation
    )

    print('\nChecking output file structure...')

    check_for_output_dir(output_dir=output_dir, labels=labels)

    print('\nCreating images...')

    create_jpg_imgs(csv=csv, labels=labels, margin=margin,
                    output_dir=output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
