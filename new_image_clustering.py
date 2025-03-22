import torch
from pathlib import Path

from torchvision import transforms

from extractor import ViTExtractor
import numpy as np
import faiss
from PIL import Image
from typing import List
from utils_image_clustering import *
import pydensecrf.densecrf as dcrf


def image_clustering(extractor, image_paths: List[str], load_size: int = 224, layer: int = 11,
                     facet: str = 'key', bin: bool = False, thresh: float = 0.065,
                     votes_percentage: int = 75,
                     sample_interval: int = 100, fg_sample_interval: int = 10, low_res_saliency_maps: bool = True,
                     vmax: int = 10,
                     test: bool = True):
    _dir = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # extractor = ViTExtractor('dino_vits8', stride , device=device)
    descriptors_list = []
    saliency_maps_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []
    saliency_extractor = extractor
    num_images = len(image_paths)

    # extract descriptors and saliency maps for each image
    if test:
        for image_path in image_paths:
            image_batch, image_pil = extractor.preprocess(image_path, load_size)
            image_pil_list.append(image_pil)
            # Extract features
            #  descs's shape (1, 1, 7921, 384)
            descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin).cpu().numpy()
            # (89,89) , (360, 360)
            curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
            num_patches_list.append(curr_num_patches)
            load_size_list.append(curr_load_size)
            descriptors_list.append(descs)
        return [], image_pil_list, descriptors_list

    # extract descriptors and saliency maps for each image
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        image_pil_list.append(image_pil)
        # Extract features
        #  descs's shape (1, 1, 7921, 384)
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin).cpu().numpy()
        # (89,89) , (360, 360)
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        descriptors_list.append(descs)
        if low_res_saliency_maps:
            if load_size is not None:
                low_res_load_size = (curr_load_size[0] // 2, curr_load_size[1] // 2)
            else:
                low_res_load_size = curr_load_size
            image_batch, _ = saliency_extractor.preprocess(image_path, low_res_load_size)

        # (1,1936) size
        saliency_map = saliency_extractor.extract_saliency_maps(image_batch.to(device)).cpu().numpy()
        # 44x44 , 180x180
        curr_sal_num_patches, curr_sal_load_size = saliency_extractor.num_patches, saliency_extractor.load_size
        if low_res_saliency_maps:
            reshape_op = transforms.Resize(curr_num_patches, transforms.InterpolationMode.NEAREST)
            saliency_map = np.array(reshape_op(Image.fromarray(saliency_map.reshape(curr_sal_num_patches)))).flatten()
        else:
            saliency_map = saliency_map[0]
        saliency_maps_list.append(saliency_map)

    _dir["image_pil"] = image_pil_list
    _dir["vmax"] = vmax
    _dir["descriptors"] = descriptors_list
    _dir["saliency_map"] = [saliency_maps_list[i].reshape(num_patches_list[i]) for i in range(len(num_patches_list))]

    #####################################################################################################
    # fg/bg seperation

    labels = kmeans_clustering(faiss, descriptors_list, sample_interval, vmax)
    num_labels = np.max(vmax)

    # 7921 descriptors for one image; same as 7921 patches
    num_descriptors_per_image = [num_patches[0] * num_patches[1] for num_patches in num_patches_list]
    # 7921 labels for each image or 7921 labels for each descriptor/patch
    labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image)[:-1])

    _dir["initial_cluster"] = [labels_per_image[i].reshape(num_patches_list[0]) for i in range(len(labels_per_image))]

    # use saliency maps to vote for salient clusters (only original images vote, not augmentations)
    votes = np.zeros(num_labels)
    for image_path, image_labels, saliency_map in zip(image_paths, labels_per_image, saliency_maps_list):
        if not ('_aug_' in Path(image_path).stem):
            for label in range(num_labels):
                # mean saliency map for one label type
                # saliency map: 7921x1; image_labels: 7921x1 (with kmeans label);
                label_saliency = saliency_map[image_labels[:, 0] == label].mean()
                if label_saliency > thresh:
                    votes[label] += 1

    salient_labels = np.where(votes >= np.ceil(num_images * votes_percentage / 100))[0]
    # Filter labels, act as fg
    while True:
        if len(salient_labels) == 0:
            thresh = thresh / 1.1
            salient_labels = getsaliency(num_labels, labels_per_image, saliency_maps_list, thresh)
        elif len(salient_labels) == num_labels:
            thresh = thresh * 1.1
            salient_labels = getsaliency(num_labels, labels_per_image, saliency_maps_list, thresh * 1.1)
        else:
            break

        # cluster all parts using k-means:
        # mask out all the labels that is in salient labels
    fg_masks = [np.isin(labels, salient_labels) for labels in labels_per_image]  # get only foreground descriptors
    _dir["fg_mask"] = fg_masks

    ############################################################################################################################
    # cluster fg patches

    # all the fg descriptors from each image
    fg_descriptor_list = [desc[:, :, fg_mask[:, 0], :] for fg_mask, desc in zip(fg_masks, descriptors_list)]
    all_fg_descriptors = np.ascontiguousarray(np.concatenate(fg_descriptor_list, axis=2)[0, 0])

    # cluster count

    n_clusters = int(all_fg_descriptors.shape[1] / 39)
    if int(all_fg_descriptors.shape[1] / 39) > 30:
        n_clusters = 30
    if int(all_fg_descriptors.shape[1] / 39) < 2:
        n_clusters = 2

    _dir["fg_n_clusters"] = n_clusters

    # cluster the fg descriptors into label
    fg_part_labels = kmeans_clustering(faiss, fg_descriptor_list, fg_sample_interval, n_clusters)

    # separate the fg descriptors as per the labels
    # we need that for gaussian distribution
    _dir["fg_cluster_descriptors"] = get_cluster_descriptors(n_clusters, fg_part_labels, all_fg_descriptors)

    ####################################################################################################
    # For visualization purpose

    # the part labels includes labels from all k images
    # we seperate the labels for each image
    new_fg_part_labels = seperate_clusters(fg_descriptor_list, fg_part_labels)

    # the fg labels do not have bg info. So, we add bg patches to get 2d plot
    _dir["fg_clustering"] = fill_missed_cluster_labels(fg_masks, new_fg_part_labels)

    ###########################################################################################################
    # cluster bg patches

    # Work on the background patches
    bg_masks = [np.invert(_m) for _m in fg_masks]

    # all the bg descriptors from each image
    bg_descriptor_list = [desc[:, :, bg_mask[:, 0], :] for bg_mask, desc in zip(bg_masks, descriptors_list)]
    all_bg_descriptors = np.ascontiguousarray(np.concatenate(bg_descriptor_list, axis=2)[0, 0])

    # cluster count
    n_clusters = 8

    n_clusters = int(all_bg_descriptors.shape[1] / 39)
    if int(all_bg_descriptors.shape[1] / 39) > 10:
        n_clusters = 10
    if int(all_bg_descriptors.shape[1] / 39) < 2:
        n_clusters = 2

    _dir["bg_n_clusters"] = n_clusters

    # cluster the bg descriptors into label
    bg_part_labels = kmeans_clustering(faiss, bg_descriptor_list, fg_sample_interval, n_clusters)

    # separate the bg descriptors as per the labels
    # we need that for gaussian distribution
    _dir["bg_cluster_descriptors"] = get_cluster_descriptors(n_clusters, bg_part_labels, all_bg_descriptors)

    ####################################################################################################################################################################################################################

    # For visualization purpose

    # the part labels includes labels from all k images
    # we seperate the labels for each image
    new_bg_part_labels = seperate_clusters(bg_descriptor_list, bg_part_labels)

    # the bg labels do not have bg info. So, we add bg patches to get 2d plot
    _dir["bg_clustering"] = fill_missed_cluster_labels(bg_masks, new_bg_part_labels)

    return _dir

