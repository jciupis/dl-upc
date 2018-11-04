import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from matplotlib import image


def cluster(embeddings, nb_clusters):
    """
    Perform a K-means clustering and plot the results.

    :param embeddings: numpy array, image embeddings
    :param nb_clusters: int, number of clusters
    """
    # Perform K-means clustering
    pred_labels = KMeans(n_clusters=nb_clusters, random_state=0).fit_predict(embeddings)

    # Apply a dimensionality reduction technique to visualize 2 dimensions.
    vis_matrix = TSNE(n_components=2).fit_transform(embeddings)

    # Using matplotlib to create the plot and show it!
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 5)]
    color_list = []
    for label in pred_labels:
        color_list.append(colors[label])

    plt.figure()
    plt.scatter(vis_matrix[:, 0], vis_matrix[:, 1], s=30, c=color_list)
    plt.show()


def nearest_neighbor(embeddings, labels):
    """
    Find the closest image in the embedding space and display them side by side.

    :param embeddings: numpy array, image embeddings
    :param labels: numpy_array, image labels
    """
    # Randomly select an image to compare others to.
    idx = np.random.randint(0, len(labels))
    img_file = labels[idx]
    img_mbed = embeddings[idx]
    img_mbed = img_mbed.reshape(1, -1)

    # Calculate dot product of every embedding with the selected one.
    similarities = cosine_similarity(img_mbed, embeddings)

    # Replace the selected image dot product with itself with a 0.
    similarities[0, idx] = 0

    # Find the image's nearest neighbor in embedding space.
    best_match_file = labels[np.argmax(similarities)]

    # Plot both images together for visual comparison.
    fig = plt.figure()
    subplot = fig.add_subplot(1, 2, 1)
    img = image.imread(img_file)
    plt.imshow(img)
    subplot.set_title('Original image')
    subplot = fig.add_subplot(1, 2, 2)
    img = image.imread(best_match_file)
    plt.imshow(img)
    subplot.set_title('Closest image')
    plt.show()


def check_equation(embeddings, labels):
    img_path_1 = 'IMG_5954_224_224.jpg'
    img_path_2 = 'IMG_5953_224_224.jpg'

    filenames = [os.path.basename(f) for f in labels]

    # Find indexes of the files to subtract.
    if img_path_1 in filenames and img_path_2 in filenames:
        idx_1 = filenames.index(img_path_1)
        idx_2 = filenames.index(img_path_2)
    else:
        return

    # Get embedding of the image difference.
    img_diff_mbed = embeddings[idx_1] - embeddings[idx_2]
    img_diff_mbed = img_diff_mbed.reshape(1, -1)

    # Calculate dot product of every embedding with the selected one.
    similarities = cosine_similarity(img_diff_mbed, embeddings)

    # Find the image's nearest neighbor in embedding space.
    best_match_file = labels[np.argmax(similarities)]

    # Plot both images together with their difference's best match for visual comparison.
    fig = plt.figure()
    subplot = fig.add_subplot(1, 3, 1)
    img = image.imread(labels[idx_1])
    plt.imshow(img)
    subplot.set_title('Image_1  -')
    subplot = fig.add_subplot(1, 3, 2)
    img = image.imread(labels[idx_2])
    plt.imshow(img)
    subplot.set_title('Image_2  =')
    subplot = fig.add_subplot(1, 3, 3)
    img = image.imread(best_match_file)
    plt.imshow(img)
    subplot.set_title('Difference file')
    plt.show()


def main():
    # Load dataset embeddings and labels.
    obj = np.load('img_mbed.npz')
    labels = obj['labels']
    embeddings = obj['embeddings']

    # Reduce number of dimensions with PCA.
    reduced_mbed = PCA(n_components=100).fit_transform(embeddings)

    # Perform selected data analysis
    choice = 0

    if choice == 0:
        nearest_neighbor(reduced_mbed, labels)
    elif choice == 1:
        cluster(reduced_mbed, 5)
    elif choice == 2:
        check_equation(embeddings, labels)


if __name__ == '__main__':
    main()


