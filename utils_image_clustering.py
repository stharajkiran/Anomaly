import math

import numpy as np
def get_cluster_descriptors(n_clusters, part_labels, part_descriptors):
    """
    :param n_clusters: number of clusters
    :param part_labels: labels of the fg/fg patches
    :param part_descriptors: fg/bg descriptors
    :return: list of descriptors where each index gives descriptors for each label
    """
    cluster_descriptors = []
    for i in range(n_clusters):
        cluster_mask = part_labels == i
        # Get all the indices of the patches from the segment
        cluster_indices = np.unique(np.argwhere(cluster_mask.flatten() == True)[:, 0], axis=0)
        # Find the patch embedding from corresponding indices/ SEGMENT EMBEDDING
        cluster_descriptor = part_descriptors[sorted(cluster_indices)]
        cluster_descriptors.append(cluster_descriptor)

    return cluster_descriptors
def fill_missed_cluster_labels(fg_masks, fg_part_labels):
    """
    Fill the missing bg patches

    :param fg_masks: list of fg masks for each image
    :param fg_part_labels: list of fg cluster labels for each image with bg patches missing
    :return:
    """
    filled_clustering = []
    for _mask, _cluster in zip(fg_masks, fg_part_labels):
        _mask = np.squeeze(_mask)
        _cluster = np.squeeze(_cluster)
        # index of each fg patch in the image
        fg_mask_index = np.unique(np.argwhere(_mask == True)[:, 0], axis=0)
        # convert boolean into 0 and 1
        _mask = _mask * 1
        # in _mask binary image, we replace all 1's by fg labels. eg: 0 fg label becomes 1, 1 becomes 2 and so on.
        # All the 0s in binary image stay 0.
        for j, index in enumerate(fg_mask_index):
            _mask[index] = _cluster[j] + 1
        filled_clustering.append(_mask)

    return filled_clustering
def seperate_clusters(part_descriptor_list, part_labels):
    """
    To separate the mixed cluster labels from  different image

    :param part_descriptor_list: list of fg/bg descriptors where each index gives all fg/bg descriptors for each image
    :param part_labels: list of all fg/bg cluster labels obtained from k images
    :return: list of cluster labels for each image
    """
    new_part_labels = []
    a = 0
    b = 0
    for i in range(len(part_descriptor_list)):
        des_len = np.squeeze(part_descriptor_list[i]).shape[0]
        b = b + des_len
        new_part_labels.append(part_labels[a:b])
        a = b
    return new_part_labels



def getsaliency(num_labels,labels_per_image, saliency_maps_list, thresh):
    votes = np.zeros(num_labels)
    for  image_labels, saliency_map in zip(labels_per_image, saliency_maps_list):
        for label in range(num_labels):
            label_saliency = saliency_map[image_labels[:, 0] == label].mean()
            if label_saliency > thresh:
                votes[label] += 1
    salient_labels = np.where(votes >= np.ceil(75 / 100))[0]
    return salient_labels


def kmeans_clustering(faiss, descriptors_list, sample_interval, n_clusters):
    all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
    # (7921, 384); 89 x 89 = 7921, so each patch is represented by 384 length vector
    normalized_all_descriptors = all_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
    sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
    # (80, 384)
    all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
    normalized_all_sampled_descriptors = all_sampled_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_sampled_descriptors)  # in-place operation

    algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
    algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
    _, cluster_labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)

    return cluster_labels

# def get_normal_distribution_using_cluster(img_segments):
#     # if img_segments has nan, convert it into integer
#     if np.isnan(img_segments).any():
#         img_segments = np.nan_to_num(img_segments, nan=np.unique(img_segments).shape[0] - 1)
#     # Convert into 2d
#     img_segments = np.squeeze(img_segments)
#     return img_segments