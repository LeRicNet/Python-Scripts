"""
Python3 script to standardize a .jpg image dataset.

Based on:
'New Variants of a Method of MRI Scale Standardization'
Nyúl LG, Udopa JK, Zhang X
IEEE Transactions on Medical Imaging. Feb 2000

Author: Eric Prince
Date: 2018-03-12
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--input_dir', type=str,
    help='Path for .jpg files to be standardized'
)
parser.add_argument(
    '--output_dir', type=str,
    help='Path to save new standardized .jpg files'
)
parser.add_argument(
    '--pc1', type=float, default=0.0,
    help='Decimal percentage for PC1'
)
parser.add_argument(
    '--pc2', type=float, default=0.998,
    help='Decimal percentage for PC2'
)
parser.add_argument(
    '--s1', type=int, default=1,
    help='Integer value for S1'
)
parser.add_argument(
    '--s2', type=int, default=4095,
    help='Integer value for S2'
)
parser.add_argument(
    '--graph', type=bool, default=False,
    help='Boolean True/False to present new images.'
)


def input_image(fname, PC1, PC2):

    IMG = mpimg.imread(fname)
    IMG = IMG[:,:,0]
    img_flattened = IMG.flatten()
    pixel_range, counts = np.unique(img_flattened, return_counts=True)

    d = {
        'pixel': pixel_range,
        'count': counts
    }
    data = pd.DataFrame(data=d)

    pc1_index = np.floor(PC1 * len(data)).astype(int)
    pc2_index = np.floor(PC2 * len(data)).astype(int)

    median_j = np.median(img_flattened)

    return IMG, data, pc1_index, pc2_index, median_j


def transform_pixels(BW_IMG, mu_i, mu_s, pc1, pc2, S1, S2):
    """Transform pixel values to standard scale"""

    print('\nTransforming Image...')
    flattened = np.array(BW_IMG).astype(int).flatten()

    transformed = []
    for pixel in flattened:
        if pixel <= mu_i:
            t_x = np.floor(
                mu_s + (pixel - mu_i) *
                (S1 - mu_s) / (pc1 - mu_i)
            )
        else:
            t_x = np.floor(
                mu_s + (pixel - mu_i) *
                (S2 - mu_s) / (pc2 - mu_i)
            )
        transformed.append(t_x)

    out = np.array(transformed).astype(int)
    size = np.sqrt(len(out)).astype(int)
    print('\tNew Image Shape is: ({}, {})'.format(size, size))
    out = out.reshape(size, size)

    return out


def plot_imgs(img_dir):
    img_list = os.listdir(img_dir)
    img_list = [i for i in img_list if i.endswith('.jpg')]
    img_list_full = [os.path.join(img_dir, i) for i in img_list]

    nrow = 7
    ncol = np.ceil(len(img_list_full) / nrow).astype(int)

    print('Count: {}'.format(len(img_list_full)))
    print('NCOL: {}'.format(ncol))
    print('NROW: {}'.format(nrow))
    print('Total Spaces: {}'.format(ncol * nrow))

    fig = plt.figure(1)
    for i in range(len(img_list_full)):
        idx = i + 1

        fname = img_list_full[i]
        img = mpimg.imread(fname)
        a = fig.add_subplot(nrow, ncol, idx)
        imgplot = plt.imshow(img)
        a.set_axis_off()

    plt.show()

def main():

    FLAGS, _ = parser.parse_known_args()

    DIR_PATH = FLAGS.input_dir
    OUTPATH = FLAGS.output_dir

    PC1 = FLAGS.pc1
    PC2 = FLAGS.pc2

    S1 = FLAGS.s1
    S2 = FLAGS.s2

    file_list = os.listdir(DIR_PATH)
    file_list = [f for f in file_list if f.endswith('.jpg')]
    file_list_full = [os.path.join(DIR_PATH, f) for f in file_list]

    file_names = []
    images = []
    median_vals = []
    pc1_vals = []
    pc2_vals = []
    for i in range(len(file_list_full)):
        fname = file_list_full[i]
        img_j, data, pc1_j, pc2_j, median_j = input_image(fname=fname, PC1=PC1, PC2=PC2)
        file_names.append(file_list[i])
        images.append(img_j)
        median_vals.append(median_j)
        pc1_vals.append(pc1_j)
        pc2_vals.append(pc2_j)

    d = {
        'filename': file_names,
        'original image': images,
        'median vals': median_vals,
        'pc1 vals': pc1_vals,
        'pc2 vals': pc2_vals
    }

    data = pd.DataFrame(data=d)

    mu_s = np.median(data['median vals'])

    for i in range(len(data['filename'])):
        fname = data['filename'][i]
        img = data['original image'][i]
        mu_i = data['median vals'][i]
        pc1_i = data['pc1 vals'][i]
        pc2_i = data['pc2 vals'][i]

        transformed = transform_pixels(
            BW_IMG=img,
            mu_i=mu_i,
            mu_s=mu_s,
            pc1=pc1_i,
            pc2=pc2_i,
            S1=S1,
            S2=S2
        )
        outfilename = os.path.join(OUTPATH, fname)
        plt.imsave(fname=outfilename, arr=transformed, cmap='gray')

    if FLAGS.graph:
        plot_imgs(img_dir=FLAGS.output_dir)


if __name__ == '__main__':
    main()
