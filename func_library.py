
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


def plot_padim(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in tqdm(range(num), f"-----------plotting for {class_name}--------------------------"):
        file_dir = os.path.join(save_dir,class_name, class_name + '_{}'.format(i))
        if os.path.isfile(file_dir + ".png"):
            continue
        img = test_img[i]
        heat_map = scores[i] * 255
        mask = scores[i].copy()
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(np.array(img), mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')

        gt = np.squeeze(gts[i])
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')

        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')

        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')

        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        if not os.path.isfile(file_dir + ".png"):
            fig_img.savefig(file_dir, dpi=100)
        plt.close()
        gc.collect()

def plot_test(test_images, scores, patch_matches,saliency_map_list, test_fg_cluster_list,
                                      test_part_segmentations, gts, threshold):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    # for i in range(num):
    i = 0
    img = test_images[i]
    heat_map = scores[i] * 255
    mask = scores[i].copy()
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
    fig_img, ax_img = plt.subplots(3, 3, figsize=(18, 18))
    fig_img.subplots_adjust(right=0.9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for row in ax_img:
        for ax_i in row:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
    ##########################################################################
    ax_img[0][0].imshow(img)
    ax_img[0][0].title.set_text('Image')

    gt = np.squeeze(gts[i])
    ax_img[0][1].imshow(gt, cmap='gray')
    ax_img[0][1].title.set_text('GroundTruth')

    ax = ax_img[0][2].imshow(heat_map, cmap='jet', norm=norm)
    ax_img[0][2].imshow(img, cmap='gray', interpolation='none')
    ax_img[0][2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
    ax_img[0][2].title.set_text('Predicted heat map')
    ##########################################################################
    _list = np.unique(patch_matches[i])
    cmap = "jet"
    # colors = [color_list[i] for i in _list]
    pil_image = Image.fromarray(img)
    ax_img[1][0].imshow(resize_pil(pil_image, 89))
    ax_img[1][0].imshow(patch_matches[i], cmap=cmap, vmin=0, vmax=len(_list), alpha=0.5, interpolation='nearest')
    ax_img[1][0].title.set_text('patch mapping')

    ax_img[1][1].imshow(mask, cmap='gray')
    ax_img[1][1].title.set_text('Predicted mask')

    ax_img[1][2].imshow(vis_img)
    ax_img[1][2].title.set_text('Segmentation result')
    ##########################################################################
    # saliency_map_list, test_fg_cluster_list,test_part_segmentations,
    fg_clusters = test_fg_cluster_list[i]
    _list = np.unique(fg_clusters)
    cmap = "jet"
    # ax_img[2][0].imshow(resize_pil(pil_image, 89))
    ax_img[2][0].imshow(fg_clusters)#, cmap=cmap, vmin=0, vmax=len(_list), alpha=0.5, interpolation='nearest')
    ax_img[2][0].title.set_text('fg clustering')

    ax_img[2][1].imshow(saliency_map_list[i], vmin=0, vmax=1, cmap='jet')
    ax_img[2][1].title.set_text('Saliency map')

    # part_segment = test_part_segmentations[i] + 1
    # part_segment = np.nan_to_num(part_segment, nan= 0)
    part_segment = test_part_segmentations[i]
    _list = np.unique(part_segment)
    ax_img[2][2].imshow(resize_pil(pil_image, 89))
    ax_img[2][2].imshow(part_segment, cmap=cmap, vmin=0, vmax=len(_list), alpha=0.5, interpolation='nearest')
    ax_img[2][2].title.set_text('part segmentation')

    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig_img.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 8,
    }
    cb.set_label('Anomaly Score', fontdict=font)


def plot_fig_during_test_with_details(test_img, scores, patch_matches,saliency_map_list, test_fg_cluster_list,
                                      test_part_segmentations, gts, threshold, save_dir, class_name):
    # color_list = ["red", "yellow", "blue", "lime", "darkviolet", "magenta", "cyan", "brown", "yellow"]
    color_list = ["red", "yellow", "blue", "lime", "darkviolet", "cyan", "magenta", "brown", "pink"]
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        file_dir = os.path.join(save_dir, class_name + '_{}'.format(i))
        if os.path.isfile(file_dir + ".png"):
            continue
        img = test_img[i]
        heat_map = scores[i] * 255
        mask = scores[i].copy()
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(3, 3, figsize=(18, 18))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for row in ax_img:
            for ax_i in row:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
        ##########################################################################
        ax_img[0][0].imshow(img)
        ax_img[0][0].title.set_text('Image')

        gt = np.squeeze(gts[i])
        ax_img[0][1].imshow(gt, cmap='gray')
        ax_img[0][1].title.set_text('GroundTruth')

        ax = ax_img[0][2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[0][2].imshow(img, cmap='gray', interpolation='none')
        ax_img[0][2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[0][2].title.set_text('Predicted heat map')
        ##########################################################################
        _list = np.unique(patch_matches[i])
        cmap = "jet"
        # colors = [color_list[i] for i in _list]
        pil_image = Image.fromarray(img)
        ax_img[1][0].imshow(resize_pil(pil_image, 89))
        ax_img[1][0].imshow(patch_matches[i], cmap=cmap, vmin=0, vmax=len(_list), alpha=0.5, interpolation='nearest')
        ax_img[1][0].title.set_text('patch mapping')

        ax_img[1][1].imshow(mask, cmap='gray')
        ax_img[1][1].title.set_text('Predicted mask')

        ax_img[1][2].imshow(vis_img)
        ax_img[1][2].title.set_text('Segmentation result')
        ##########################################################################
        # saliency_map_list, test_fg_cluster_list,test_part_segmentations,
        fg_clusters = test_fg_cluster_list[i]
        _list = np.unique(fg_clusters)
        cmap = "jet"
        # ax_img[2][0].imshow(resize_pil(pil_image, 89))
        ax_img[2][0].imshow(fg_clusters, cmap=cmap, vmin=0, vmax=len(_list), alpha=0.5, interpolation='nearest')
        ax_img[2][0].title.set_text('fg clustering')

        ax_img[2][1].imshow(saliency_map_list[i], vmin=0, vmax=1, cmap='jet')
        ax_img[2][1].title.set_text('Saliency map')

        # part_segment = test_part_segmentations[i] + 1
        # part_segment = np.nan_to_num(part_segment, nan= 0)
        part_segment = test_part_segmentations[i]
        _list = np.unique(part_segment)
        ax_img[2][2].imshow(resize_pil(pil_image, 360))
        ax_img[2][2].imshow(part_segment, cmap=cmap, vmin=0, vmax=len(_list), alpha=0.5, interpolation='nearest')
        ax_img[2][2].title.set_text('part segmentation')

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        if not os.path.isfile(file_dir + ".png"):
            fig_img.savefig(file_dir, dpi=100)
        plt.close()
        gc.collect()

def plot_fig_during_test(test_img, scores, patch_matches, gts, threshold, save_dir, class_name):
    # color_list = ["red", "yellow", "blue", "lime", "darkviolet", "magenta", "cyan", "brown", "yellow"]
    color_list = ["red", "yellow", "blue", "lime", "darkviolet", "cyan", "magenta", "brown", "pink"]
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        file_dir = os.path.join(save_dir, class_name + '_{}'.format(i))
        if os.path.isfile(file_dir + ".png"):
            continue
        img = test_img[i]
        heat_map = scores[i] * 255
        mask = scores[i].copy()
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(2, 3, figsize=(18, 12))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for row in ax_img:
            for ax_i in row:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
        ax_img[0][0].imshow(img)
        ax_img[0][0].title.set_text('Image')

        gt = np.squeeze(gts[i])
        ax_img[0][1].imshow(gt, cmap='gray')
        ax_img[0][1].title.set_text('GroundTruth')
        ax = ax_img[0][2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[0][2].imshow(img, cmap='gray', interpolation='none')
        ax_img[0][2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[0][2].title.set_text('Predicted heat map')

        _list = np.unique(patch_matches[i])
        cmap = "jet"
        # colors = [color_list[i] for i in _list]
        pil_image = Image.fromarray(img)
        ax_img[1][0].imshow(resize_pil(pil_image, 89))
        ax_img[1][0].imshow(patch_matches[i], cmap=cmap, vmin=0, vmax=len(_list), alpha=0.5, interpolation='nearest')
        ax_img[1][0].title.set_text('patch mapping')

        ax_img[1][1].imshow(mask, cmap='gray')
        ax_img[1][1].title.set_text('Predicted mask')

        ax_img[1][2].imshow(vis_img)
        ax_img[1][2].title.set_text('Segmentation result')

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        if not os.path.isfile(file_dir + ".png"):
            fig_img.savefig(file_dir, dpi=100)
        plt.close()
        gc.collect()


def _reshape(img_descriptors):
    return np.squeeze(img_descriptors[0])


def resize_pil(pil_image, load_size):
    x = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
    return x


def get_normal_distribution_using_cluster(img_segments, descriptors):
    # if img_segments has nan, convert it into integer
    if np.isnan(img_segments).any():
        img_segments = np.nan_to_num(img_segments, nan=np.unique(img_segments).shape[0] - 1)
    else:
        img_segments = np.copy(img_segments)
    # Convert into 2d
    img_segments = np.squeeze(img_segments)

    img_size = img_segments.shape[0]

    patches_count = int(math.sqrt(descriptors.shape[0]))
    patch_size = int(img_size / patches_count)
    # This is to remove border rows and columns to match the sizes of descriptors and image
    pad = int(abs(img_segments.shape[0] - patches_count * patch_size) / 2)
    img_segments = img_segments[pad:360 - pad, pad:360 - pad]
    img_segments = img_segments + 1
    # Each element is a list of distances of each patch from the particular segment's Normal distribution
    gaussians = []
    # Find all gaussian first
    for i in list(np.unique(img_segments).astype(int)):
        # Get each segment
        segment = img_segments == i
        # Get all the indices of the patches from the segment
        patch_indices = np.unique(np.argwhere(segment.flatten() == True)[:, 0], axis=0)
        # Find the patch embedding from corresponding indices/ SEGMENT EMBEDDING
        label_array = descriptors[sorted(patch_indices)]
        # mean value of the segment embedding
        mean = torch.mean(torch.from_numpy(label_array), dim=0).numpy()
        I = np.identity(384)
        cov = np.cov(label_array, rowvar=False) + 0.01 * I
        conv_inv = np.linalg.inv(cov)
        gaussians.append((mean, conv_inv))
    return gaussians

def convert_fig_to_PIL(fig):
    pil_fig = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return pil_fig

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

def get_data(img_path, vmax=10, save_dir=None, test=False):
    # Model parameters
    elbow = 0.975
    load_size = 360
    layer = 11
    facet = 'token'
    bin = False
    thresh = 0.05
    model_type = 'dino_vits8'
    stride = 4
    votes_percentage = 75
    sample_interval = 1
    low_res_saliency_maps = False
    elbow_second_stage = 0.94

    with torch.no_grad():
        # computing part cosegmentation
        _dict = image_clustering(img_path, elbow, load_size, layer,
                                 facet, bin, thresh, model_type,
                                 stride, votes_percentage, sample_interval,
                                 low_res_saliency_maps, vmax, elbow_second_stage,
                                 save_dir=save_dir, test=test)
    if test:
        _, image_pil_list, descriptors_list = _dict
        img_descriptors = _reshape(descriptors_list)
        return _, image_pil_list[0], img_descriptors
    else:
        return _dict

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


def get_ground_truth_mask(image_label, mask_path, resize=412, crop_size=360):
    transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                T.CenterCrop(crop_size),
                                T.ToTensor()])

    if image_label == 0:
        mask = torch.zeros([1, crop_size, crop_size])
    else:
        mask = Image.open(mask_path)
        mask = transform_mask(mask)
    return mask

def get_ground_truth_mask(y, z, resize=412, crop_size=360):
    gt_mask_list = []
    transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                T.CenterCrop(crop_size),
                                T.ToTensor()])

    for i in range(len(z)):
        if y[i] == 0:
            mask = torch.zeros([1, 360, 360])
        else:
            mask = Image.open(z[i])
            mask = transform_mask(mask)
        gt_mask_list.append(mask)
    return gt_mask_list

def get_metrics_pixel(gt_mask, scores):
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    return {
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "rocauc": rocauc
    }

def get_metrics_image(gt_list, img_scores):
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "rocauc": img_roc_auc
    }