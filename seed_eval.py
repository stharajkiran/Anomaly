import argparse

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *
from extractor import ViTExtractor
import torch.nn.functional as F


def main(args):
    kwargs = vars(args)

    # logger.info('==========running parameters=============')
    # for k, v in kwargs.items():
    #     logger.info(f'{k}: {v}')
    # logger.info('=========================================')

    seeds = [111, 333, 999, 444,555]
    kwargs['seed'] = seeds[kwargs['experiment_indx']]
    setup_seed(kwargs['seed'])


    # prepare the experiment dir
    model_dir, img_dir, logger_dir, model_name, csv_path, scores_dir = get_dir_from_args(**kwargs)

    kwargs["scores_dir"] = scores_dir

    ################### check here #################################################

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)
    paths = train_dataset_inst.img_paths
    paths = [path.split("\\")[-1] for path in paths]
    logger.info(f" k size confirmation for seed {kwargs['seed']}: {paths} \n")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, default='candle')

    parser.add_argument('--resize', type=int, default=412)
    parser.add_argument('--load_size', type=int, default=360)
    parser.add_argument("--root-dir", type=str, default="./result_seed_confirmation")
    parser.add_argument('--facet', type=str, default="token")

    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--experiment_indx", type=int, default=0)

    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument("--gpu-id", type=int, default=0)



    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    # setup_seed(args.experiment_indx)
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)