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
                for exp_index in [0, 1,2,3,4]:
                    for k in [1,2,4]:
                        for cls in classes[:]:
                            sh_method = f'python seed_eval.py ' \
                                        f'--dataset {dataset} ' \
                                        f'--class-name {cls} ' \
                                        f'--k-shot {k} ' \
                                        f'--layer {layer} ' \
                                        f'--facet {facet} ' \
                                        f'--experiment_indx {exp_index} '

                            print(sh_method)
                            pool.apply_async(os.system, (sh_method,))



    pool.close()
    pool.join()