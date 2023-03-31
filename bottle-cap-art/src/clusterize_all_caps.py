import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from tqdm import tqdm

n_colors = 7

# # init image array with shape (0, 3)
# all_caps_pixels_array = np.empty((0, 3), float)

# # Load all caps
# dataset = pd.read_csv("/Users/thomas/Documents/Autres/Perso/bottle-cap-art/data/my_dataset.csv", sep=';', index_col='id')
# for identifier, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
#     count = row['count']

#     input_image = cv2.imread(f"/Users/thomas/Documents/Autres/Perso/bottle-cap-art/data/my_caps/tmp/{identifier}.png")
#     input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#     input_image = np.array(input_image, dtype=np.float64) / 255
#     input_image = cv2.resize(input_image, (58, 58), interpolation=cv2.INTER_AREA)
#     w, h, d = tuple(input_image.shape)
#     assert d == 3

#     # Keep only the pixels inside the inscribed circle
#     # TODO: use library get_caps_patch ?
#     center = (w // 2, h // 2)
#     radius = min(w, h) // 2
#     mask = np.zeros((w, h), dtype=np.uint8)
#     cv2.circle(mask, center, radius, 1, thickness=-1)

#     # Append to all_caps_pixels_array only the pixels inside the circle
#     # do it `count` times
#     for _ in range(count):
#         all_caps_pixels_array = np.append(all_caps_pixels_array, input_image[mask == 1], axis=0)

# print(f"all_caps_pixels_array.shape: {all_caps_pixels_array.shape}")

# load all_caps_pixels_array from a file
all_caps_pixels_array = np.load("/Users/thomas/Documents/Autres/Perso/bottle-cap-art/data/all_caps_pixels_array.npy")

# Fitting model on a small sub-sample of the data
image_array_sample = shuffle(all_caps_pixels_array, random_state=0, n_samples=1_000)
kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=0).fit(
    image_array_sample
)

# Get labels for all points
# Predicting color indices on the full image
labels = kmeans.predict(all_caps_pixels_array)

fig, ax = plt.subplots()
# Display palette
palette_color_list = (kmeans.cluster_centers_ * 255).astype(int)
for i, color in enumerate(palette_color_list):
    hexa = '#%s' % ''.join(('%02x' % p for p in color))  # rgb to hex
    print(hexa, f"{100 * np.count_nonzero(labels == i) / labels.size:.2f}%")
    print(color)
    ax.bar(i, 1, color=hexa)
ax.set_axis_off()
plt.title("Dataset color palette")

# plt.show()

# save all_caps_pixels_array to a file
# np.save("/Users/thomas/Documents/Autres/Perso/bottle-cap-art/data/all_caps_pixels_array.npy", all_caps_pixels_array)
