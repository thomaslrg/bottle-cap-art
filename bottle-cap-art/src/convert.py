import numpy as np
from sklearn.cluster import KMeans
import cv2


def get_color_palette(img, n_colors):
    # Calculer la palette de couleurs de l'image
    img_array = np.array(img)
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img_array)
    color_palette = kmeans.cluster_centers_
    return color_palette


def replace_colors(img, palette_a, palette_b):
    # Remplacer les couleurs de l'image selon la palette correspondante
    img_array = np.array(img)
    img_shape = img_array.shape
    img_array = img_array.reshape((img_shape[0] * img_shape[1], 3))
    index_a = np.argmin(np.linalg.norm(img_array[:, None, :] - palette_a[None, :, :], axis=-1), axis=1)
    new_colors = palette_b[index_a]
    new_img_array = new_colors.reshape((img_shape[0] * img_shape[1], 3))
    new_img_array = np.uint8(np.round(new_img_array))
    new_img = new_img_array.reshape((img_shape[0], img_shape[1], 3))
    return new_img


# Charger l'image d'origine
img = cv2.imread("/Users/thomas/Documents/Autres/Perso/bottle-cap-art/data/input.jpeg")

# Calculer la palette de couleurs de l'image d'origine
palette_a = get_color_palette(img, 3)

# Définir la palette de couleurs cible
palette_7 = np.array([
    [8, 16, 63],
    [193, 199, 196],
    [142, 132, 107],
    [154, 19, 15],
    [82, 86, 69],
    [17, 101, 48],
    [8, 8, 7],
])
# palette_20 = np.array([
palette_b = palette_7
palette_b = palette_b[:, [2, 1, 0]]

# Remplacer les couleurs de l'image selon la palette correspondante
new_img = replace_colors(img, palette_a, palette_b)

# Enregistrer l'image transformée
cv2.imwrite("/Users/thomas/Documents/Autres/Perso/bottle-cap-art/data/test.jpeg", new_img)
