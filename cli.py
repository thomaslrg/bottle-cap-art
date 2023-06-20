import argparse
import os
import sys
import matplotlib.pyplot as plt

from bottle_cap_art import generate_image


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add the command line arguments
    parser.add_argument('--input-image-path', type=str, help='Path to the input image', required=True)
    parser.add_argument('--count-limit', type=bool, help='True if the number of caps is limited (default is False)', required=False)
    parser.add_argument('--nb-rotations', type=int, help='Number of rotations (default is 4)', required=False)
    parser.add_argument('--nb-caps-cols', type=int, help='Number of capsules columns (default is 20)', required=False)
    parser.add_argument('--image-width', type=int, help='Width of the output image (default is 2000)', required=False)

    # TODO: `mode` argument is not handled yet
    mode = "Depuis le centre"
    noise = 100

    # Parse the arguments
    args = parser.parse_args()

    # Load the input image
    if not os.path.isfile(args.input_image_path):
        print(f'Error: {args.input_image_path} is not a valid file')
        sys.exit(1)
    input_img = plt.imread(args.input_image_path)

    # Set the default values
    if args.count_limit is None:
        args.count_limit = False
    if args.nb_rotations is None:
        args.nb_rotations = 4
    if args.nb_caps_cols is None:
        args.nb_caps_cols = 20
    if args.image_width is None:
        args.image_width = 2000

    # Generate the output image
    output_img = generate_image(
        input_img=input_img,
        nb_rotations=args.nb_rotations,
        nb_caps_cols=args.nb_caps_cols,
        image_width=args.image_width,
        count_limit=args.count_limit,
        mode=mode,
        noise=noise,
        from_streamlit=False,
    )

    # Show the output image
    plt.imshow(output_img)
    plt.show()


if __name__ == '__main__':
    main()
