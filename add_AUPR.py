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
         is_vis: bool,
         img_dir: str,
         train_data: DataLoader,
         **kwargs):

    class_name = kwargs['class_name']

    gt_list = []
    gt_mask_list = []
    _dict = {}

    # img_path, gt, label, img_name
    for (idx, img_path, gt, label, img_name) in tqdm(dataloader):
        for (_id, i, g, l, n) in zip(idx, img_path, gt, label, img_name):
            _id = int(_id)
            mask =  get_ground_truth_mask(g, l)

            gt_mask_list.append(mask)
            gt_list.append(l)


    scores_dir = kwargs["scores_dir"]
    scores_pkl_path = f"{scores_dir}/visa-{class_name}"
    scores_dict_filepath = os.path.join(scores_pkl_path, "scores.pkl")
    if os.path.exists(scores_dict_filepath) :
        try:
            # Now on test image
            with open(scores_dict_filepath, 'rb') as f:
                scores_info = pickle.load(f)
        except:
            logger.info(f"{scores_dict_filepath} is not available")

    else:
        scores_dict_filepath = os.path.join(scores_pkl_path, "new_scores.pkl")
        try:
            # Now on test image
            with open(scores_dict_filepath, 'rb') as f:
                scores_info = pickle.load(f)
        except:
            logger.info(f"{scores_dict_filepath} is not available")

    # Load the dict and scores
    scores = scores_info['scores']

    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list, dtype=int)
    precision, recall, _ = precision_recall_curve(gt_list, img_scores)
    aupr = auc(recall, precision)

    return {'image_AUPR': aupr * 100}, None


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
    kwargs["scores_dir"] = scores_dir
    ################### check here #################################################

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)


    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)


    ################################# fix the model issue ####################################


    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics, scores_info = test(None, test_dataloader, is_vis=True, img_dir=img_dir, train_data=train_dataloader,
                                **kwargs)

    logger.info(f"\n")

    for k, v in metrics.items():
        logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")

    metrics, total_classes, class_name, dataset, csv_path, scores_info, scores_dir = (metrics, dataset_classes[kwargs['dataset']],
                                                                                      kwargs['class_name'],kwargs['dataset'], csv_path, scores_info, scores_dir)

    if dataset != 'mvtec':
        for indx in range(len(total_classes)):
            total_classes[indx] = f"{dataset}-{total_classes[indx]}"
        class_name = f"{dataset}-{class_name}"
    write_results(metrics, class_name, total_classes, csv_path)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, default='macaroni1')

    parser.add_argument('--resize', type=int, default=412)
    parser.add_argument('--load_size', type=int, default=360)
    parser.add_argument("--root-dir", type=str, default="./result_SeDIM")
    parser.add_argument('--facet', type=str, default="token")

    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--experiment_indx", type=int, default=0)


    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)
    # method related parameters
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--fg_sample_interval', type=int, default=1)


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
