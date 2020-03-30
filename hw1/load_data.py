import sys
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import imageio

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):

        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """

        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        all_image_batches = None
        all_label_batches = None

        for i in range(batch_size):

            sampled_classes = np.random.choice(folders, size=self.num_classes)
            sampled_data = get_images(paths=sampled_classes,
                                      labels=range(self.num_classes),
                                      nb_samples=self.num_samples_per_class,
                                      shuffle=False)

            # Shape: K x N x Inp_dim
            images = np.swapaxes(np.stack([image_file_to_array(data[1],
                                    self.dim_input) for data in sampled_data]).reshape(self.num_classes,
                                                                                       self.num_samples_per_class,
                                                                                       -1), 0, 1)
            images = np.expand_dims(images, axis=0)

            one_hot_labels = np.eye(self.num_classes)
            labels = np.swapaxes(np.stack([one_hot_labels[data[0]] for data in sampled_data]).reshape(self.num_classes,
                                                                                          self.num_samples_per_class,
                                                                                          -1), 0, 1)
            labels = np.expand_dims(labels, axis=0)

            # Shuffle data
            for k in range(labels.shape[1]):
                shuffled_indices = np.random.permutation(self.num_classes)
                new_labels = labels[0,k][shuffled_indices]
                labels[0, k] = new_labels
                new_images = images[0, k][shuffled_indices]
                images[0, k] = new_images

            if all_image_batches is None:
                all_image_batches = images
                all_label_batches = labels
            else:
                all_image_batches = np.vstack([all_image_batches, images])
                all_label_batches = np.vstack([all_label_batches, labels])

        return all_image_batches, all_label_batches

# test display 2 batches
def viz_batch(imgs, labels):

    K = imgs.shape[1]
    N = imgs.shape[2]

    fig, axes = plt.subplots(K*2, N, figsize=(10,10))

    for k in range(K*2):

        for n in range(N):

            if k < 2:
                axes[k, n].imshow(imgs[0,k,n,:].reshape(28,28))
                axes[k, n].set_title(f'Label: {labels[0, k, n]}, k:{k}')
            else:
                axes[k, n].imshow(imgs[1, k-2, n, :].reshape(28, 28))
                axes[k, n].set_title(f'Label: {labels[1, k-2, n]}, k:{k}')


    plt.show()




# test
#generator = DataGenerator(3,2)
#imgs, labels = generator.sample_batch('train', 5)
#viz_batch(imgs,labels)