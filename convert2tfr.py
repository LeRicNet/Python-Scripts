import tensorflow as tf
import numpy as np
import glob
import random
from PIL import Image
import argparse
import os


# Python script utility to take a directory containing 2 different
# categories of images in .jpg format and convert them to .tfrecord
# format for subsequent TensorFlow work.

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help=".jpeg files to be\ "
                                            "converted to .tfrecord "
                                            "files")
    parser.add_argument("--output_dir", help="output path for "
                                             ".tfrecord files")
    parser.add_argument("--file_diff", help="string to use for "
                                            "record separation")
    parser.add_argument("--dataset_prefix", help="string to append as"
                                                 " prefix for "
                                                 ".tfrecord files")
    args = parser.parse_args()

    if args.image_dir:
        image_dir = args.image_dir
    else:
        raise ValueError('--image_dir required')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        raise ValueError('--output_dir required')
    if args.file_diff:
        fdiff = args.file_diff
    if args.dataset_prefix:
        dtpx = args.dataset_prefix

    image_path_pattern = os.path.join(image_dir,
                                      "*.jpg")
    images = glob.glob(image_path_pattern)
    random.seed(42)
    random.shuffle(images)

    labels = [0 if fdiff in img else 1 for img in images]

    data = list(zip(images, labels))

    data_size = len(data)
    split_size = int(0.7*data_size)

    training_imgs, training_lbls = zip(*data[:split_size])
    validation_imgs, validation_lbls = zip(*data[split_size:])


    # build training tfrecords files..
    training_length = len(training_imgs)
    for i in range(training_length):
        idx = i + 1
        tfrecord_filename = '{}/{}-train-{}-of-{}.tfrecord'.format(output_dir,\
                            dtpx, idx, training_length)

        # Initiating the writer and creating the tfrecords file.

        writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    # Loading the location of all files - image dataset

    label = training_lbls[i]
    img = Image.open(training_imgs[i])
    img = np.array(img.resize((512,512)))

    feature = {'label': _int64_feature(label),
               'image': _bytes_feature(img.tostring())}


    # Create an example protocol buffer

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Writing the serialized example.

    writer.write(example.SerializeToString())

    writer.close()

    validation_length = len(validation_imgs)
    for i in range(validation_length):
        idx = i + 1
        tfrecord_filename = '{}/{}-validation-{}-of-{' \
                            '}.tfrecord'.format(output_dir, dtpx ,idx,
                                                validation_length)

        # Initiating the writer and creating the tfrecords file.

        writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        # Loading the location of all files - image dataset

        label = validation_lbls[i]
        img = Image.open(validation_imgs[i])
        img = np.array(img.resize((512,512)))

        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(img.tostring())}


        # Create an example protocol buffer

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Writing the serialized example.

        writer.write(example.SerializeToString())

        writer.close()
