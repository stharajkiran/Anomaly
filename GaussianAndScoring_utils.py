
from tqdm import tqdm
import gc
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import PIL
import pickle
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
# from image_clustering import image_clustering
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from skimage import morphology
from skimage.segmentation import mark_boundaries
import torch
from torchvision import transforms as T
import math
from extractor import ViTExtractor
from new_image_clustering import image_clustering
from func_library import _reshape


def get_data(img_path, extractor = None, load_size = 360, layer = 11, facet = "key", vmax = 10,sample_interval = 1, fg_sample_interval = 1, save_dir = None, test = False):
    # Model parameters
    # load_size = 360
    # layer = kwargs['layer'] #11
    # facet = kwargs['facet'] #'token'
    bin = False
    thresh = 0.05
    votes_percentage = 75
    low_res_saliency_maps = False

    with torch.no_grad():
        # computing part cosegmentation
        _dict = image_clustering(extractor, img_path, load_size, layer,
                                facet, bin, thresh,
                                 votes_percentage, sample_interval, fg_sample_interval,
                                low_res_saliency_maps, vmax, test = test)
    if test:
        _ , image_pil_list, descriptors_list = _dict
        img_descriptors = _reshape(descriptors_list)
        return [], image_pil_list[0], img_descriptors
    else:
        return _dict

def get_normal_distribution_using_cluster(cluster_descriptors_list):
    """
    Find gaussian distribution for each cluster where cluster type is given by index

    :param cluster_descriptors_list: list of clusters where each index represents all the descriptors associated to that cluster
    :return: Gaussian of each cluster
    """
    # Each element is a list of distances of each patch from the particular segment's Normal distribution
    gaussians = []
    # Find all gaussian first
    for i in range(len(cluster_descriptors_list)):
        # Get each segment
        cluster_descriptor = cluster_descriptors_list[i]
        # mean value of the segment embedding
        mean = torch.mean(torch.from_numpy(cluster_descriptor), dim=0).numpy()
        I = np.identity(384)
        cov = np.cov(cluster_descriptor, rowvar=False) + 0.01 * I
        conv_inv = np.linalg.inv(cov)
        gaussians.append((mean, conv_inv))
    return gaussians


def get_ground_truth_mask(gt, label, resize=360):
    if label == 0:
        mask = torch.zeros([1,resize, resize])
    else:
        mask = Image.open(gt)
        l = min(mask.size)
        transform_mask = T.Compose([T.CenterCrop(l),
                                    T.Resize(resize, Image.NEAREST),
                                    T.CenterCrop(resize),
                                    T.ToTensor()])
        mask = transform_mask(mask)
    return mask


def get_anomaly_map_with_fg(fg_mask, test_descriptors, gaussians):
    # For each patch
    min_dist_from_gaussian = []
    min_distance_indices = []
    for patch_mask, patch_embed in zip(fg_mask.flatten(), test_descriptors):
        if patch_mask == 0:
            min_dist_from_gaussian.append(0)
            min_distance_indices.append(0)
            continue
        dist_from_gaussian = [mahalanobis(patch_embed, normal[0], normal[1]) for normal in gaussians]
        _min = min(dist_from_gaussian)
        min_dist_from_gaussian.append(_min)
        idx = np.argmin(dist_from_gaussian)
        min_distance_indices.append(idx)
    # Get numpy 2d array
    min_dist_from_gaussian_arr = np.array(min_dist_from_gaussian).reshape(89, 89)
    min_distance_indices_arr = np.array(min_distance_indices).reshape(89, 89)
    return min_dist_from_gaussian_arr, min_distance_indices_arr

def get_anomaly_map(test_descriptors, gaussians):
    # For each patch
    min_dist_from_gaussian = []
    min_distance_indices = []
    for patch_embed in test_descriptors:
        dist_from_gaussian = [mahalanobis(patch_embed, normal[0], normal[1]) for normal in gaussians]
        _min = min(dist_from_gaussian)
        min_dist_from_gaussian.append(_min)
        idx = np.argmin(dist_from_gaussian)
        min_distance_indices.append(idx)
    # Get numpy 2d array
    min_dist_from_gaussian_arr = np.array(min_dist_from_gaussian).reshape(89, 89)
    min_distance_indices_arr = np.array(min_distance_indices).reshape(89, 89)
    return min_dist_from_gaussian_arr, min_distance_indices_arr

def get_anomaly_and_patch_mapping(class_name, train_indices, train_pkl_dir = "datasets/mvtec_dict_embedding",
                                  test_pkl_dir = "datasets/mvtec_dict_test_embedding"):
    # Get all directories for test category
    test_images = []
    anomaly_map_list = []
    patch_mapping_list = []
    saliency_map_list = []
    test_fg_cluster_list = []
    test_part_segmentations = []
    train_dict_filepath = os.path.join(train_pkl_dir, class_name + ".pkl")
    test_dict_filepath = os.path.join(test_pkl_dir, class_name + ".pkl")

    with open(train_dict_filepath, 'rb') as f:
        train_class_dict = pickle.load(f)
    if not os.path.exists(test_dict_filepath):
        return None
    # Find normal gaussian
    fg_mask = np.squeeze(train_class_dict['good']['fg_mask'][0])
    fg_mask_index = np.unique(np.argwhere(fg_mask == True)[:, 0], axis=0)
    fg_mask = fg_mask * 1

    fg_clusters = np.squeeze(train_class_dict['good']['fg_clustering'])
    for i, index in enumerate(fg_mask_index):
        fg_mask[index] = fg_clusters[i] + 1

    normal_segments = fg_mask.reshape(89, 89)
    normal_descriptors = np.squeeze(train_class_dict['good']['descriptors'][0])
    gaussians = get_normal_distribution_using_cluster(normal_segments, normal_descriptors)

    # Now on test image
    with open(test_dict_filepath, 'rb') as f:
        test_class_dict = pickle.load(f)
    # Each test image
    for i in tqdm(train_indices, f"------|__{class_name}__ |-------"):
        # Find category type
        # Get data from dictionary
        fg_mask = np.squeeze(test_class_dict[i]['fg_mask'][0])
        fg_clusters = np.squeeze(test_class_dict[i]['fg_clustering'])
        test_pil_image, test_descriptors = test_class_dict[i]["image_pil"][0], test_class_dict[i]['descriptors']
        test_descriptors = _reshape(test_descriptors)

        fg_mask_index = np.unique(np.argwhere(fg_mask == True)[:, 0], axis=0)
        fg_mask = fg_mask * 1
        for j, index in enumerate(fg_mask_index):
            fg_mask[index] = fg_clusters[j] + 1

        # anomaly_map, patch_mapping are both np array of 89x89 shape
        anomaly_map, patch_mapping = get_anomaly_map_with_fg(fg_mask, test_descriptors, gaussians)
        anomaly_map_list.append(anomaly_map)
        patch_mapping_list.append(patch_mapping)
        test_images.append(np.array(test_pil_image))

        saliency_map_list.append(test_class_dict[i]["saliency_map"])
        test_fg_cluster_list.append(fg_mask.reshape(89,89))

        part_segment = test_class_dict[i]["part_segmentations"][0].reshape(360,360) + 1
        part_segment = np.nan_to_num(part_segment, nan=0)
        test_part_segmentations.append(part_segment)
    return test_images, anomaly_map_list, patch_mapping_list, saliency_map_list, test_fg_cluster_list, test_part_segmentations

def load_dataset_folder(dataset_path, class_name, is_train=False):
    x, y, mask = [], [], []
    phase = 'train' if is_train else 'test'
    img_dir = os.path.join(dataset_path, class_name, phase)
    gt_dir = os.path.join(dataset_path, class_name, 'ground_truth')

    img_types = sorted(os.listdir(img_dir))
    for img_type in img_types:

        # load images
        img_type_dir = os.path.join(img_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                 for f in os.listdir(img_type_dir)
                                 if f.endswith('.png')])
        x.extend(img_fpath_list)

        # load gt labels
        if img_type == 'good':
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
        else:
            y.extend([1] * len(img_fpath_list))
            gt_type_dir = os.path.join(gt_dir, img_type)
            img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
            gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                             for img_fname in img_fname_list]
            mask.extend(gt_fpath_list)
    # merged = list(itertools.chain(*x))
    assert len(x) == len(y), 'number of x and y should be same'

    return list(x), list(y), list(mask)