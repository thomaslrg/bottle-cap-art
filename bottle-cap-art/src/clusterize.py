import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

n_colors = 20

# Load input image
input_image = cv2.imread("/Users/thomas/Documents/Autres/Perso/bottle-cap-art/data/input.jpeg")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# input_image = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
input_image = np.array(input_image, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(input_image.shape)
assert d == 3
image_array = np.reshape(input_image, (w * h, d))

# Fitting model on a small sub-sample of the data
image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=0).fit(
    image_array_sample
)

# Get labels for all points
# Predicting color indices on the full image
labels = kmeans.predict(image_array)

# Count how many percent of each color is in the image
for i in range(n_colors):
    print(f"Color {i}: {100 * np.count_nonzero(labels == i) / labels.size:.2f}%")

# Display all

plt.figure(1)
plt.clf()
plt.axis("off")
plt.title("Original image (96,615 colors)")
# Display original image
plt.imshow(input_image)

plt.figure(2)
plt.clf()
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, K-Means)")
# Recreate the (compressed) image from the palette and labels
plt.imshow(kmeans.cluster_centers_[labels].reshape(w, h, -1))

fig, ax = plt.subplots()
# Display palette
palette_color_list = (kmeans.cluster_centers_ * 255).astype(int)
for i, color in enumerate(palette_color_list):
    hexa = '#%s' % ''.join(('%02x' % p for p in color))  # rgb to hex
    ax.bar(i, 1, color=hexa)
ax.set_axis_off()
plt.title("Color palette")

plt.show()
