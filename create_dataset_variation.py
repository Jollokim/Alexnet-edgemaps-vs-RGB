import argparse
import os

import cv2 as cv

from preprocess import r_preprocess


# function for defining all the commandline parameters
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    # directory parameters:
    parser.add_argument('--root_dir', type=str, required=True, help='root dir of original dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir')

    # method parameter
    parser.add_argument('--method', type=str, required=True, help='Method to preprocess images with')

    return parser


# function that creates the new variation of the dataset
# dataset variation uses the same train, valid, test split
def create_variation(args):
    sets = ['train', 'valid', 'test']

    # gets the preprocessing function
    method = r_preprocess(args.method)

    # makes sure already existing variation don't get demolished D:
    if os.path.exists(args.output_dir):
        raise Exception("Output dir already exist")

    # creates the dataset folder
    os.mkdir(args.output_dir)

    for s in sets:
        # creates the set folders (train, valid, test)
        os.mkdir(f'{args.output_dir}/{s}')

        # get the classes from the dataset the dataset is made of
        labels = os.listdir(f'{args.root_dir}/{s}')

        # creates class folders
        for label in os.listdir(f'{args.root_dir}/{s}'):
            os.mkdir(f'{args.output_dir}/{s}/{label}')

        for label in labels:
            for img_p in os.listdir(f'{args.root_dir}/{s}/{label}'):

                # reads img
                img = cv.imread(f'{args.root_dir}/{s}/{label}/{img_p}')

                # processes the image by given method
                img_pro = method(img)

                # writes the images to new variation dataset. The original image and preprocessed images stays within
                # the same split set
                cv.imwrite(f'{args.output_dir}/{s}/{label}/{img_p}', img_pro)


# program start
if __name__ == '__main__':
    # creates commandline parser
    parser = argparse.ArgumentParser('Create dataset variation', parents=[get_args_parser()])
    args = parser.parse_args()

    # passes the commandline arguments to the create_variation function
    create_variation(args)
