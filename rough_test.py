import argparse
import pickle

from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL.Image import Image

from GaussianAndScoring_utils import get_ground_truth_mask, load_dataset_folder
from SeDIM.gaussian_model import GaussianModel

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *
from extractor import ViTExtractor
import torch.nn.functional as F

# test(model, test_dataloader,train_dataloader, device, is_vis=True, img_dir=img_dir, **kwargs)

def test(model,
         dataloader: DataLoader,
         device: str,
         is_vis: bool,
         img_dir: str,
         class_name: str,
         cal_pro: bool,
         train_data: DataLoader,
         resolution: int):

    # change the model into eval mode
    if train_data is not None:
        logger.info('begin build cluster gaussians...')
        img_paths = []
        for (idx, img_path, gt, label, img_name) in train_data:
            for (i, g, l, n) in zip(img_path, gt, label, img_name):
                img_paths.append(i)

        # model.build_gaussian(img_paths)
        logger.info('build cluster gaussians finished.')

    saved_test_info = False
    test_dic_folders= "datasets/test_embedding/412_360"
    test_pkl_dir = f"{test_dic_folders}/facet_{model.facet}_layer_{model.layer}/{class_name}"
    test_dict_filepath = os.path.join(test_pkl_dir, class_name + ".pkl")

    if os.path.exists(test_dict_filepath):
        # Now on test image
        with open(test_dict_filepath, 'rb') as f:
            test_class_dict = pickle.load(f)
            saved_test_info = True
            logger.info(f"{test_dict_filepath} is available")
    else:
        logger.info('Features to be created/stored.')


    indices = []
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    patch_mapping_list = []
    names = []
    _dict = {}

    # img_path, gt, label, img_name
    for (idx, img_path, gt, label, img_name) in tqdm(dataloader):
        for (_id, i, g, l, n) in zip(idx, img_path, gt, label, img_name):
            _id = int(_id)
            mask = get_ground_truth_mask(g, l)
            if saved_test_info:
                # index = x.index(i)
                # test_image_pil, test_descriptors = test_class_dict[_id]["image_pil"][0], test_class_dict[_id]['descriptors']
                test_image_pil, test_descriptors = test_class_dict[_id]
                test_descriptors = np.squeeze(test_descriptors)
                # _ , anomaly_map, patch_mapping = [],[],[]# model(None, test_descriptors)  # data is test_img_dir
                _, anomaly_map, patch_mapping = model(None, test_descriptors)  # data is test_img_dir
            else:
                # test_image_pil, test_descriptors, anomaly_map, patch_mapping = [],[],[],[]
                test_image_pil, test_descriptors, anomaly_map, patch_mapping = model([i]) # data is test_img_dir
                _dict[_id] = (test_image_pil,test_descriptors)
            gt_mask_list.append(mask)
            gt_list.append(l)
            # scores.append(anomaly_map)
            # patch_mapping_list.append(patch_mapping)
            # test_imgs.append(test_image_pil)
            names.append(img_name)
            indices.append(_id)

    # if not saved_test_info:
    #     if not os.path.exists(test_dict_filepath):
    #         if not os.path.exists(test_pkl_dir):
    #             os.makedirs(test_pkl_dir, exist_ok=True)
    #         with open(test_dict_filepath, 'wb') as f:
    #             pickle.dump(_dict, f)
    #
    # scores = np.array(scores)
    # scores = torch.tensor(scores)
    # scores = F.interpolate(scores.unsqueeze(1), size=360, mode='bilinear',
    #                           align_corners=False).squeeze().numpy()
    #
    # # apply gaussian smoothing on the score map
    # for i in range(scores.shape[0]):
    #     scores[i] = gaussian_filter(scores[i], sigma=4)
    #
    # max_score = scores.max()
    # min_score = scores.min()
    # scores = (scores - min_score) / (max_score - min_score)

    scores = np.random.random((len(gt_list), 360, 360))
    scores_info = {}
    # scores_info['max_score'] = max_score
    # scores_info['min_score'] = min_score
    # scores_info['scores'] = scores
    # scores_info['patch_mapping_list'] = patch_mapping_list


    # test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))
    result_dict = metric_cal(scores, gt_list, gt_mask_list, cal_pro=cal_pro)

    if is_vis:
        plot_fig_during_test_with_details(names, test_imgs, scores, patch_mapping_list, gt_mask_list, result_dict['p_threshold'], img_dir, class_name)

    return result_dict, scores_info


def main(args):
    kwargs = vars(args)

    logger.info('==========running parameters=============')
    for k, v in kwargs.items():
        logger.info(f'{k}: {v}')
    logger.info('=========================================')

    seeds = [111, 333, 999]
    kwargs['seed'] = seeds[kwargs['experiment_indx']]
    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device


    # prepare the experiment dir
    model_dir, img_dir, logger_dir, model_name, csv_path, scores_dir = get_dir_from_args(**kwargs)

    ################### check here #################################################

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)


    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    ################################# fix the model issue ####################################

    # Do not need to initialize anything, just set all the parameters
    # get the model
    extractor = ViTExtractor(kwargs['model_type'], kwargs['stride'], device=kwargs['device'])

    layer = kwargs['layer']
    facet = kwargs['facet']
    model = GaussianModel(extractor, layer, facet)
    model = model.to(device)
    model.set_model_dir(model_dir)


    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics, scores_info = test(model, test_dataloader, device, is_vis=True, img_dir=img_dir,
                   class_name=kwargs['class_name'], cal_pro=kwargs['cal_pro'], train_data=train_dataloader,
                   resolution=kwargs['resolution'])

    logger.info(f"\n")

    for k, v in metrics.items():
        logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path, scores_info, scores_dir)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, default='candle')

    parser.add_argument('--img-resize', type=int, default=412)
    parser.add_argument('--img-cropsize', type=int, default=360)
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--facet', type=str, default="token")
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result_SeDIM")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=True)
    parser.add_argument("--experiment_indx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--fg_sample_interval', type=int, default=1)

    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--model_type", type=str, default="dino_vits8")
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
