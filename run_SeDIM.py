import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)
    # datasets = ['mvtec','visa']
    datasets = ['visa']
    for dataset in datasets:

        classes = dataset_classes[dataset]
        for layer in [7]:
            for facet in ["token"]:
                for exp_index in [0, 1, 2,3, 4]:
                    for first_k in [5,20,40, 60,80]:
                        for k in [1,2,4]:
                            for cls in classes[:]:
                                dir = f"result_224/facet_{facet}_layer_{layer}/visa-k-{k}-exp{exp_index}-first_k{first_k}/scores_info/visa-{cls}/scores.pkl"
                                if not os.path.exists(dir):
                                    sh_method = f'python eval_SeDIM.py ' \
                                                f'--dataset {dataset} ' \
                                                f'--class-name {cls} ' \
                                                f'--k-shot {k} ' \
                                                f'--layer {layer} ' \
                                                f'--facet {facet} ' \
                                                f'--first_k {first_k} ' \
                                                f'--experiment_indx {exp_index} '

                                    print(sh_method)
                                    pool.apply_async(os.system, (sh_method,))


    pool.close()
    pool.join()