import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.color as colorModule
import imutils
import os
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List


def patch_to_capsule_classifier(
    images_path_list: List[str],
    caps_diameter: int = 50,
    nb_rotations: int = 4,
    count_limit: bool = False,
) -> Tuple[NearestNeighbors, List[np.ndarray]]:
    """Clusterize a list of capsule images, with L2 norm in the LAB space

    Args:
        images_path_list (list): list of paths to capsule crops images
        caps_diameter (int, optional): diameter of a capsule in pixels. Defaults to 50.
        nb_rotations (int, optional): number of rotations to perform on each capsule. Defaults to 4.
        count_limit (bool, optional): wether to limit the number of capsules to the number of capsules in the dataset. Defaults to False.

    Returns:
        NN_clf: A sklearn classifier which takes as input a patch in LAB and returns which capsule to use to represent the patch
        capsule_patches: A list of images of capsules that could be outputed bt the clf
    """
    all_capsules_flat = []
    capsule_patches = []
    caps_patch_white = get_caps_patch(D=caps_diameter, color=np.ones(3), factor=1)[:, :, 0]

    for path in images_path_list:
        patch = plt.imread(path)
        if patch.shape[2] == 4:
            patch[patch[:, :, 3] < 0.5] = 0
            patch = patch[:, :, :3]
        patch = cv2.resize(
            patch, (caps_diameter, caps_diameter), interpolation=cv2.INTER_AREA
        )
        for angle in range(nb_rotations):
            patch_rotated = imutils.rotate(
                patch.copy(), angle=(angle * 360 / nb_rotations)
            )
            if patch_rotated.dtype == "uint8":
                patch_rotated = patch_rotated.astype("float") / 255
            patch_rotated[caps_patch_white == 0] = 0
            capsule_patches.append(patch_rotated)
            patch_lab = colorModule.rgb2lab(
                patch_rotated
            )
            patch_flat = patch_lab.reshape(-1)
            all_capsules_flat.append(patch_flat)

    def patch_L2_dist(patch1_flat_LAB, patch2_flat_LAB):
        p1 = patch1_flat_LAB.reshape((caps_diameter, caps_diameter, 3))
        p2 = patch2_flat_LAB.reshape((caps_diameter, caps_diameter, 3))
        return np.sum(np.sum((p1 - p2) ** 2, axis=2) ** 0.5)

    # clf which associates a new patch with the set of given patches
    # if count_limit is True, the clf will return the all the nearest caps and we will take the first available
    # if count_limit is False, the clf will just return the nearest cap
    clf = NearestNeighbors(
        n_neighbors=len(all_capsules_flat) if count_limit else 1, metric=patch_L2_dist
    )
    clf.fit(all_capsules_flat)

    return clf, capsule_patches


def dnorm(x, mu, sd):
    """Compute the normal distribution

    Args:
        x (float): value
        mu (float): mean
        sd (float): standard deviation

    Returns:
        float: normal distribution
    """
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    """Compute a gaussian kernel

    Args:
        size (int): size of the kernel
        sigma (int, optional): standard deviation. Defaults to 1.

    Returns:
        np.array: gaussian kernel
    """
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.sum()
    return kernel_2D


def get_caps_patch(
    D: int,
    color: np.array,
    factor: float = 0.994
) -> np.array:
    """Get a patch of a capsule

    Args:
        D (int): diameter of a capsule in pixels
        color (np.array): color of the capsule
        factor (float, optional): threshold for the gaussian kernel. Defaults to 0.994.

    Returns:
        np.array: capsule patch
    """
    patch = np.zeros((D, D, len(color)))
    gaussian2D = gaussian_kernel(D, sigma=D)
    lim = 0
    for j in range(D):
        if gaussian2D[0, j] > lim:
            lim = gaussian2D[0, j]
    for i in range(D):
        for j in range(D):
            if gaussian2D[i, j] >= lim * factor:
                patch[i, j] = color
    return patch


def compute_positions_reordering(
    positions_y_x: List[Tuple[int, int]],
    mode: str,
    center_y_x: Tuple[int, int],
    noise: int,
) -> np.ndarray:
    """Compute the order in which the capsules will be generated

    Args:
        positions_y_x (List[Tuple[int, int]]): list of positions of the capsules
        mode (str): mode of generation
        center_y_x (Tuple[int, int]): center of the image

    Returns:
        np.ndarray: order of generation
    """
    if mode == "Double parcours de grille":
        return np.arange(len(positions_y_x), dtype=int)
    elif mode == "Depuis le centre":
        positions_y_x = np.array(positions_y_x)
        dists_to_center = np.sum((positions_y_x - center_y_x) ** 2, axis=1) ** 0.5
        noise_dists = (np.random.rand(*dists_to_center.shape) - 0.5) * noise
        dists_to_center += noise_dists
        order = np.argsort(dists_to_center)
        return order
    else:
        raise NotImplementedError()


def display_text(text: str, for_streamlit: bool = False):
    """Display text in streamlit

    Args:
        text (str): text to display
        for_streamlit (bool, optional): wether to display in streamlit or to print in console
    """
    if for_streamlit:
        st.text(text)
    else:
        print(text)


def generate_image(
    input_img: np.ndarray,
    nb_rotations: int,
    nb_caps_cols: int,
    image_width: int,
    count_limit: bool,
    mode: str,
    noise: int,
    from_streamlit: bool = False,
) -> np.ndarray:
    """Generate a capsule version of the input image.

    Uses the capsule dataset and positions them to reproduce the original image
    according to the given parameters

    Args:
        input_img (np.ndarray): PIL Image loaded in streamlit
        nb_rotations (int): number of rotations to try for each position of each capsule
        nb_caps_cols (int): number of capsules to use to make the width of the image
        image_width (int): number of pixels in width of generated image
        count_limit (bool): limit use of dataset capsules to quantity available
        mode (str): mode for the reordering of the capsules
        noise (int): noise if reordering from center

    Returns:
        np.ndarray: generated image
    """

    # Load input image
    image = np.array(input_img)

    # Load dataset count
    dataset = pd.read_csv("data/my_dataset.csv", sep=";", index_col="id")
    count_list = list(dataset["count"])

    # Load dataset images
    images_dir_path = "data/my_caps/"
    images_path_list = os.listdir(images_dir_path)
    for identifier in dataset.index:
        if str(identifier) + ".png" not in images_path_list:
            print(
                f"Warining: {images_dir_path} does not contain the image {identifier}.png"
            )
    images_path_list = [
        os.path.join(images_dir_path, image) for image in images_path_list
    ]

    # Load classifier
    caps_diameter = int(image_width / (3**0.5 * (nb_caps_cols - 1) + 1))
    clf, all_capsule_patches = patch_to_capsule_classifier(
        images_path_list=images_path_list,
        caps_diameter=caps_diameter,
        nb_rotations=nb_rotations,
        count_limit=count_limit,
    )

    # Calculate capsule positions
    h_step = int(round(3**0.5 * caps_diameter))
    v_step = caps_diameter
    v_step += v_step % 2
    h_step += h_step % 2
    width = (nb_caps_cols - 1) * h_step + caps_diameter
    height = int(image.shape[0] * width / image.shape[1])
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Calculate real size
    # We measured that 38 capsules take 1 meter
    D_cm = 100 / 38
    hauteur_cm = round(height * D_cm / caps_diameter)
    largeur_cm = round(width * D_cm / caps_diameter)
    display_text(f"\nCe bail mesure {hauteur_cm}cm par {largeur_cm}cm.", for_streamlit=from_streamlit)

    long_height = (height - caps_diameter) // v_step + 1
    long_width = (width - caps_diameter) // h_step + 1

    if image.dtype == "uint8":
        image = image.astype("float") / 255

    image_lab = colorModule.rgb2lab(image)

    capsule_image = np.zeros(image.shape)
    liste = []
    patchs = []
    positions_y_x = []
    capsules_epuisees = []
    if count_limit:
        for caps_idx, caps_nb in enumerate(count_list):
            if caps_nb == 0:
                capsules_epuisees.append(1 + caps_idx)

    # Extract capsule patches
    for row in range(long_height):
        for col in range(long_width):
            patch_lab = image_lab[
                v_step * row: v_step * row + caps_diameter,
                h_step * col: h_step * col + caps_diameter,
            ]
            patchs.append(patch_lab.reshape(-1))
            positions_y_x.append(
                (v_step * row + caps_diameter / 2, h_step * col + caps_diameter / 2)
            )
    for row in range(long_height - 1):
        for col in range(long_width - 1):
            patch_lab = image_lab[
                v_step // 2 + v_step * row: v_step // 2 + v_step * row + caps_diameter,
                h_step // 2 + h_step * col: h_step // 2 + h_step * col + caps_diameter,
            ]
            patchs.append(patch_lab.reshape(-1))
            positions_y_x.append(
                (
                    v_step // 2 + v_step * row + caps_diameter / 2,
                    h_step // 2 + h_step * col + caps_diameter / 2,
                )
            )

    # Get order of capsules according to mode
    positions_order = compute_positions_reordering(
        positions_y_x=positions_y_x,
        mode=mode,
        center_y_x=(height / 2, width / 2),
        noise=noise,
    )

    # Change order of coordinates and patches
    positions_y_x = [positions_y_x[positions_order[i]] for i in range(len(positions_order))]
    patchs = [patchs[positions_order[i]] for i in range(len(positions_order))]

    display_text(f"Il va te falloir {long_height*long_width + (long_height-1)*(long_width-1)} capsules pour faire cette dingz.", for_streamlit=from_streamlit)
    display_text("Bon je calcule man, Laisse moi calculer tranquille man...", for_streamlit=from_streamlit)

    # Classify capsules for each patch
    dists, inds = clf.kneighbors(patchs)

    display_text("J'y suis presque...", for_streamlit=from_streamlit)

    # Add capsules to image
    for idx, pos_y_x in enumerate(positions_y_x):
        neighbor_rank = 0
        id_caps = 1 + inds[idx][neighbor_rank] // nb_rotations
        while id_caps in capsules_epuisees:
            neighbor_rank += 1
            if neighbor_rank == len(inds[idx]):
                break
            id_caps = 1 + inds[idx][neighbor_rank] // nb_rotations
        if neighbor_rank != len(inds[idx]):
            capsule_image[
                round(pos_y_x[0] - caps_diameter / 2): round(pos_y_x[0] + caps_diameter / 2),
                round(pos_y_x[1] - caps_diameter / 2): round(pos_y_x[1] + caps_diameter / 2),
            ] += all_capsule_patches[inds[idx][neighbor_rank]]
            if count_limit:
                count_list[id_caps - 1] -= 1
                if count_list[id_caps - 1] == 0:
                    capsules_epuisees.append(id_caps)
        else:
            id_caps = -1
        liste.append(id_caps)

    display_text("Et voilà !", for_streamlit=from_streamlit)

    # Return image
    capsule_image[capsule_image > 1] = 1
    return Image.fromarray((capsule_image * 255).astype("uint8"))
