import argparse
import os
import sys
import pandas as pd
import numpy as np

from src.library import capsulify_Real_


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add the command line arguments
    parser.add_argument('--input-image-path', type=str, help='Path to the input image', required=True)
    parser.add_argument('--dataset-csv-path', type=str, help='Path to the CSV file containing columns "id" and "count"', required=True)
    parser.add_argument('--images-dir-path', type=str, help='Path to the directory containing the images as [id].png', required=True)
    parser.add_argument('--count-limit', type=bool, help='True if the number of caps is limited (default is False)', required=False)
    parser.add_argument('--nb-rotations', type=int, help='Number of rotations (default is 8)', required=False)
    parser.add_argument('--nb-caps-cols', type=int, help='Number of capsules columns (default is 20)', required=False)
    parser.add_argument('--image-width', type=int, help='Width of the output image (default is 2000)', required=False)
    parser.add_argument('--output-image-path', type=str, help='Path to the output image', required=True)

    # Parse the arguments
    args = parser.parse_args()

    # ===== args.input_image_path =====
    if not os.path.isfile(args.input_image_path):
        print(f'Error: {args.input_image_path} is not a valid file')
        sys.exit(1)

    # ===== args.input_image_path =====
    if not os.path.isfile(args.dataset_csv_path):  # check if the file exists
        print(f'Error: {args.dataset_csv_path} is not a valid file')
        sys.exit(1)
    try:  # check if the dataset is a valid CSV file
        dataset = pd.read_csv(args.dataset_csv_path, sep=';', index_col='id')
    except Exception as e:
        print(f'Error: {args.dataset_csv_path} is not a valid CSV file: {e}')
        sys.exit(1)
    if 'count' not in dataset.columns:  # check if the dataset contains the column "count"
        print(f'Error: {args.dataset_csv_path} does not contain the column "count"')
        sys.exit(1)
    if not np.array_equal(np.array(dataset.index), 1 + np.arange(len(dataset))):  # check if the dataset is sorted by "id"
        print(f'Error: {args.dataset_csv_path} is not sorted by "id"')
        sys.exit(1)

    # ===== args.images_dir_path =====
    if not os.path.isdir(args.images_dir_path):  # check if the directory exists
        print(f'Error: {args.images_dir_path} is not a valid directory')
        sys.exit(1)
    images_path_list = os.listdir(args.images_dir_path)
    for identifier in dataset.index:  # check if the images are present
        if str(identifier) + '.png' not in images_path_list:
            print(f'Error: {args.images_dir_path} does not contain the image {identifier}.png')
            sys.exit(1)
    images_path_list = [os.path.join(args.images_dir_path, image) for image in images_path_list]  # add the root directory to the images paths

    # ===== args.nb_rotations =====
    if args.nb_rotations is None:
        args.nb_rotations = 8

    # ===== args.nb_caps_cols =====
    if args.nb_caps_cols is None:
        args.nb_caps_cols = 20

    # ===== args.image_width =====
    if args.image_width is None:
        args.image_width = 2000

    # Pour supprimer des capsules
    # caps_ids_a_jeter = []
    # for id in caps_ids_a_jeter:
    #     dataset_capsules['nombre_capsules'][id] = 0

    # my_capsules_numbers = list(dataset_capsules['nombre_capsules'])
    capsulify_Real_(
        input_image_path=args.input_image_path,
        images_path_list=images_path_list,
        count_list=list(dataset['count']),
        count_limit=args.count_limit,
        params_ordre_remplissage={'mode': "from_center", 'noise': 100},
        nb_rotations=args.nb_rotations,
        nb_caps_cols=args.nb_caps_cols,
        image_width=args.image_width,
        output_image_path=args.output_image_path
    )
