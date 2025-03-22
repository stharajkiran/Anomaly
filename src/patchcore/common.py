import copy
import math
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from patchify import patchify
from scipy.spatial.distance import mahalanobis


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]



class GaussianScorer(object):
    def __init__(self, num_parts: int) -> None:

        self.num_parts = num_parts
        self.gaussians = None

    def fit(self, img_segments,  descriptors) -> None:
        self.gaussians = self.get_normal_distribution(img_segments, descriptors)

    def predict(
        self, test_descriptors):

        assert self.gaussians is not None
        min_dist_from_gaussian_arr, min_distance_indices_arr = self.get_anomaly_map(test_descriptors)
        return min_dist_from_gaussian_arr, min_distance_indices_arr

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def load(self, load_folder: str, prepend: str = "") -> None:
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )

    def get_normal_distribution(self, img_segments, descriptors):
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
            # convert into patches.
            segment_patches = patchify(segment, (4, 4), step=4)
            # Reshape into linear patches(7921,4,4)
            segment_patches = segment_patches.reshape((89 * 89, 4, 4))
            # Using np.argmax
            # Get all the indices of the patches from the segment
            patch_indices = np.unique(np.argwhere(segment_patches == True)[:, 0], axis=0)
            # Find the patch embedding from corresponding indices/ SEGMENT EMBEDDING
            label_array = descriptors[sorted(patch_indices)]
            # mean value of the segment embedding
            mean = torch.mean(torch.from_numpy(label_array), dim=0).numpy()
            I = np.identity(384)
            cov = np.cov(label_array, rowvar=False) + 0.01 * I
            conv_inv = np.linalg.inv(cov)
            gaussians.append((mean, conv_inv))
        return gaussians

    def get_anomaly_map(self, test_descriptors):
        # For each patch
        min_dist_from_gaussian = []
        min_distance_indices = []
        for patch_embed in test_descriptors:
            dist_from_gaussian = [mahalanobis(patch_embed, normal[0], normal[1]) for normal in self.gaussians]
            _min = min(dist_from_gaussian)
            min_dist_from_gaussian.append(_min)
            idx = np.argmin(dist_from_gaussian)
            min_distance_indices.append(idx)
        # Get numpy 2d array
        min_dist_from_gaussian_arr = np.array(min_dist_from_gaussian).reshape(89, 89)
        min_distance_indices_arr = np.array(min_distance_indices).reshape(89, 89)
        return min_dist_from_gaussian_arr, min_distance_indices_arr