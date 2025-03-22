import cv2
import matplotlib
from skimage import morphology
from skimage.segmentation import mark_boundaries

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from torchvision import transforms
import gc

from PIL import Image

##
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

##
import matplotlib.ticker as mtick


def plot_sample_cv2(names, imgs, scores_: dict, gts, save_folder=None):
    # get subplot number
    total_number = len(imgs)

    scores = scores_.copy()
    # normarlisze anomalies
    for k, v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        scores[k] = (scores[k] - min_value) / max_value * 255
        scores[k] = scores[k].astype(np.uint8)
    # draw gts
    mask_imgs = []
    for idx in range(total_number):
        gts_ = gts[idx]
        mask_imgs_ = imgs[idx].copy()
        mask_imgs_[gts_ > 0.5] = (0, 0, 255)
        mask_imgs.append(mask_imgs_)

    # save imgs
    for idx in range(total_number):
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_ori.jpg'), imgs[idx])
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_gt.jpg'), mask_imgs[idx])

        for key in scores:
            heat_map = cv2.applyColorMap(scores[key][idx], cv2.COLORMAP_JET)
            visz_map = cv2.addWeighted(heat_map, 0.5, imgs[idx], 0.5, 0)
            cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_{key}.jpg'),
                        visz_map)

def resize_pil(pil_image, load_size):
    x = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
    return x

def plot_fig_during_test_with_details(names, test_img, scores, patch_matches, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    class_dir = os.path.join(save_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir, exist_ok=True)

    for i in range(num):
        file_dir = os.path.join(save_dir, class_name, class_name + '_{}'.format(i))
        # file_dir = os.path.join(class_dir, names[i])
        if os.path.isfile(file_dir + ".png"):
            continue
        img = test_img[i]
        if type(img) is not np.ndarray:
            img = np.array(img)
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

def plot_training_img(img, saliency, first_cluster, fg_cluster):
    for i in range(num):
        file_dir = os.path.join(save_dir, class_name, class_name + '_{}'.format(i))
        # file_dir = os.path.join(class_dir, names[i])
        if os.path.isfile(file_dir + ".png"):
            continue
        img = test_img[i]
        if type(img) is not np.ndarray:
            img = np.array(img)
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


        if not os.path.isfile(file_dir + ".png"):
            fig_img.savefig(file_dir, dpi=100)
        plt.close()
        gc.collect()

def plot_anomaly_score_distributions(scores: dict, ground_truths_list, save_folder, class_name):
    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 100000

    for k, v in scores.items():
        layer_score = np.stack(v, axis=0)
        normal_score = layer_score[ground_truths == 0]
        abnormal_score = layer_score[ground_truths != 0]

        plt.clf()
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # with plt.style.context(['science', 'ieee', 'no-latex']):
        sns.histplot(np.random.choice(normal_score, N_COUNT), color="green", bins=50, label='${d(p_n)}$',
                     stat='probability', alpha=.75)
        sns.histplot(np.random.choice(abnormal_score, N_COUNT), color="red", bins=50, label='${d(p_a)}$',
                     stat='probability', alpha=.75)

        plt.xlim([0, 3])

        save_path = os.path.join(save_folder, f'distributions_{class_name}_{k}.jpg')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)


valid_feature_visualization_methods = ['TSNE', 'PCA']


def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):
    assert method in valid_feature_visualization_methods
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)

    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')


def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label == l, 0],
                      feat_proj[label == l, 1],
                      feat_proj[label == l, 2], s=5)

    return ax1


def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label == l, 0],
                    feat_proj[label == l, 1], s=5)

    return ax1



