import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.color as colorModule
from sklearn.neighbors import NearestNeighbors
import imutils


def patch_to_capsule_classifier(
    images_path_list, caps_diameter=50, nb_rotations=4, count_limit=False, show_capsules=False
):
    """Clusterize a list of capsule images, with L2 norm in the LAB space

    Parameters
    ----------
    my_patchs_paths : list
        list of paths to capsule crops images
    D : int, optional
        diameter of a capsule in pixels, by default 50
    nb_angles : list, optional
        list of angles in degree to perform rotations on to add a new rotated capsule in the dictionary
    show_capsules : bool, optional
        wether to display capsule images, by default False

    Returns
    -------
    NN_clf
        A sklearn classifier which takes as input a patch in LAB and returns which capsule to use to represent the patch
    capsule_patches
        A list of images of capsules that could be outputed bt the clf
    """
    all_capsules_flat = []
    capsule_patches = []
    # patch blanc
    caps_patch_white = get_caps_patch(D=caps_diameter, color=np.ones(3), factor=1)[:, :, 0]

    for path in images_path_list:
        # lit une image de capsule
        patch = plt.imread(path)
        if patch.shape[2] == 4:  # channel de transparence
            patch[patch[:, :, 3] < 0.5] = 0
            patch = patch[:, :, :3]
        patch = cv2.resize(patch, (caps_diameter, caps_diameter), interpolation=cv2.INTER_AREA)
        for angle in range(nb_rotations):
            # tourne la capsule
            patch_rotated = imutils.rotate(
                patch.copy(), angle=(angle * 360 / nb_rotations)
            )
            if patch_rotated.dtype == "uint8":
                patch_rotated = patch_rotated.astype("float") / 255
            # met a zero la partie du patch ne contenant pas la capsule
            patch_rotated[caps_patch_white == 0] = 0
            capsule_patches.append(patch_rotated)
            if show_capsules:
                plt.imshow(patch_rotated), plt.show()
            # convertit le patch de RGB a LAB
            patch_lab = colorModule.rgb2lab(
                patch_rotated
            )  # TODO : try other colorspaces
            # vectorise le patch et l'ajoute a la liste pour clustering
            patch_flat = patch_lab.reshape(-1)
            all_capsules_flat.append(patch_flat)

    def patch_L2_dist(patch1_flat_LAB, patch2_flat_LAB):
        # distance L2 entre patchs LAB
        return np.sum(
            np.sum(
                (
                    patch1_flat_LAB.reshape((caps_diameter, caps_diameter, 3))
                    - patch2_flat_LAB.reshape((caps_diameter, caps_diameter, 3))
                )
                ** 2,
                axis=2,
            )
            ** 0.5
        )

    # clf which associates a new patch with the set of given patches
    # if count_limit is True, the clf will return the all the nearest caps and we will take the first available
    # if count_limit is False, the clf will just return the nearest cap
    clf = NearestNeighbors(
        n_neighbors=len(all_capsules_flat) if count_limit else 1,
        metric=patch_L2_dist
    )
    clf.fit(all_capsules_flat)

    return clf, capsule_patches


def capsule_step(D):
    """Get the horizontal and vertical shifts to generate a semi-grid of hexagons (capsules)

    Parameters
    ----------
    D : int
        diameter of a capsule in pixels

    Returns
    -------
    tuple
        (horizontal-step, vertical-step)
    """

    h_step = int(round(3**0.5 * D))
    h_step += h_step % 2  # steps forced to be even

    v_step = D
    v_step += v_step % 2

    return h_step, v_step


def caps_dimensionsApparentes(image_row, image_col, D):
    """Get the dimensions of the output image in number of capsules

    Parameters
    ----------
    image_row : int
        number of pixel rows
    image_col : int
        number of pixel columns
    D : int
        diameter of a capsule in pixels

    Returns
    -------
    tuple
        (number of capsule big-rows ,  number of capsule big-columns)
    """

    h_caps_step, v_caps_step = capsule_step(D)

    return (image_row - D) // v_caps_step + 1, (image_col - D) // h_caps_step + 1


def nb_capsules_necessary(image_row, image_col, D):
    """Get the needed number of capsules

    Parameters
    ----------
    image_row : int
        number of pixel rows
    image_col : int
        number of pixel columns
    D : int
        diameter of a capsule in pixels

    Returns
    -------
    int
        number of capsules
    """

    long_height, long_width = caps_dimensionsApparentes(image_row, image_col, D)

    return long_height * long_width + (long_height - 1) * (long_width - 1)


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.sum()

    if verbose:
        plt.imshow(kernel_2D, interpolation="none", cmap="gray")
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()

    return kernel_2D


def get_caps_patch(D, color, factor=0.994):
    """ Get a patch of a capsule

    Parameters
    ----------
    D : int
        diameter of a capsule in pixels
    factor : float
        threshold for the gaussian kernel

    Returns
    -------
    np.array
        capsule patch
    """
    patch = np.zeros((D, D, len(color)))
    gaussian2D = gaussian_kernel(D, sigma=D, verbose=False)
    lim = 0
    for j in range(D):
        if gaussian2D[0, j] > lim:
            lim = gaussian2D[0, j]
    for i in range(D):
        for j in range(D):
            if gaussian2D[i, j] >= lim * factor:
                patch[i, j] = color
    return patch


####################################################################################################


def compute_positions_reordering(positions_y_x, mode, **kwargs):
    if mode == "double_parcours_de_grille" or mode is None:
        return np.arange(len(positions_y_x), dtype=int)
    # definir une fonction f(x,y) sur les positions et renvoyer le argsort des f(positions)
    elif mode == "from_center":
        center_y_x = np.array(kwargs["center_y_x"])
        noise = np.array(kwargs["noise"])
        positions_y_x = np.array(positions_y_x)  # shape (n,2)
        dists_to_center = np.sum((positions_y_x - center_y_x) ** 2, axis=1) ** 0.5
        noise_dists = (np.random.rand(*dists_to_center.shape) - 0.5) * noise
        dists_to_center += noise_dists
        order = np.argsort(dists_to_center)
        return order
    else:
        raise NotImplementedError()


def capsulify_Real_(
    input_image_path: str,
    images_path_list: list,
    count_list: list,
    nb_rotations: int,
    count_limit: bool,
    params_ordre_remplissage: dict,
    nb_caps_cols: int,
    image_width: int,
    output_image_path: str,
):

    # load image
    image = plt.imread(input_image_path)

    # calculate the diameter of the capsules in pixels
    caps_diameter = int(image_width / (3**0.5 * (nb_caps_cols - 1) + 1))

    # rearrange the sizes found to have beautiful rounds
    # if caps_diameter < 17:
    #     caps_diameter = 14
    # elif caps_diameter < 24:
    #     caps_diameter = 21
    # if caps_diameter == 33:
    #     caps_diameter = 30
    # if caps_diameter == 27:
    #     caps_diameter = 24

    # init a classifier for all the capsules
    clf, all_capsule_patches = patch_to_capsule_classifier(
        images_path_list=images_path_list,
        caps_diameter=caps_diameter,
        nb_rotations=nb_rotations,
        count_limit=count_limit,
    )
    h_step = int(round(3**0.5 * caps_diameter))
    h_step += h_step % 2  # steps forced to be even

    v_step = caps_diameter
    v_step += v_step % 2  # steps forced to be even

    width = (nb_caps_cols - 1) * h_step + caps_diameter
    height = int(image.shape[0] * width / image.shape[1])
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # plt.imshow(image), plt.title("Image"), plt.show()

    D_cm = 100 / 38  # 38 capsules pour faire 1m m'a dit Léo ?
    hauteur_cm = round(height * D_cm / caps_diameter)
    largeur_cm = round(width * D_cm / caps_diameter)
    print(f"\nCe bail mesure {hauteur_cm}cm par {largeur_cm}cm.")

    long_height, long_width = (height - caps_diameter) // v_step + 1, (width - caps_diameter) // h_step + 1

    if image.dtype == "uint8":
        image = image.astype("float") / 255

    image_lab = colorModule.rgb2lab(image)

    capsule_image = np.zeros(image.shape)
    liste = []
    patchs = []
    positions_y_x = []
    capsules_epuisees = []
    if count_limit:
        for caps_idx, caps_nb in enumerate(
            count_list
        ):  # on peut des le debut mettre qu'on a 0 de telles ou telles capsules et ca sera pris en compte
            if caps_nb == 0:
                capsules_epuisees.append(1 + caps_idx)

    # remplit les emplacements de capsules un par un
    for row in range(long_height):
        for col in range(long_width):
            patch_lab = image_lab[
                v_step * row: v_step * row + caps_diameter,
                h_step * col: h_step * col + caps_diameter
            ]
            patchs.append(patch_lab.reshape(-1))
            positions_y_x.append(
                (v_step * row + caps_diameter / 2, h_step * col + caps_diameter / 2)
            )  # retient quel emplacement c'est
    for row in range(long_height - 1):
        for col in range(long_width - 1):
            patch_lab = image_lab[
                v_step // 2 + v_step * row: v_step // 2 + v_step * row + caps_diameter,
                h_step // 2 + h_step * col: h_step // 2 + h_step * col + caps_diameter,
            ]
            patchs.append(patch_lab.reshape(-1))
            positions_y_x.append(
                (v_step // 2 + v_step * row + caps_diameter / 2, h_step // 2 + h_step * col + caps_diameter / 2)
            )  # retient quel emplacement c'est

    positions_order = compute_positions_reordering(
        positions_y_x,
        params_ordre_remplissage.get("mode", None),
        center_y_x=(height / 2, width / 2),
        noise=params_ordre_remplissage.get("noise", None),
    )

    positions_y_x = [
        positions_y_x[positions_order[i]] for i in range(len(positions_order))
    ]
    patchs = [patchs[positions_order[i]] for i in range(len(positions_order))]
    print()
    print(
        f"Il va te falloir {long_height*long_width + (long_height-1)*(long_width-1)} capsules pour faire cette dingz."
    )
    print()
    print("Bon je calcule man, Laisse moi calculer tranquille man...")
    dists, inds = clf.kneighbors(patchs)

    for idx, pos_y_x in enumerate(positions_y_x):
        neighbor_rank = 0
        id_caps = 1 + inds[idx][neighbor_rank] // nb_rotations
        while (
            id_caps in capsules_epuisees
        ):  # TODO : ajouter une contrainte sur dist ou pas ? bof hein
            neighbor_rank += 1
            if neighbor_rank == len(inds[idx]):
                break
            id_caps = 1 + inds[idx][neighbor_rank] // nb_rotations

        if neighbor_rank != len(inds[idx]):  # on met une capsule
            capsule_image[
                round(pos_y_x[0] - caps_diameter / 2): round(pos_y_x[0] + caps_diameter / 2),
                round(pos_y_x[1] - caps_diameter / 2): round(pos_y_x[1] + caps_diameter / 2),
            ] += all_capsule_patches[inds[idx][neighbor_rank]]
            if count_limit:
                # update l'etat des nombres de capsules
                count_list[id_caps - 1] -= 1
                if count_list[id_caps - 1] == 0:
                    capsules_epuisees.append(id_caps)
        else:
            # on laisse noir si on a plus de capsules disponible
            id_caps = -1

        liste.append(id_caps)

    capsule_image[capsule_image > 1] = 1
    plt.imshow(capsule_image), plt.show()

    if output_image_path is not None:
        plt.imsave(output_image_path, capsule_image)

    return image, capsule_image, liste, count_list

