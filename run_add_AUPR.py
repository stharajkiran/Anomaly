import os
from datasets import dataset_classes
from multiprocessing import Pool
import pandas as pd

if __name__ == '__main__':



    pool = Pool(processes=1)
    # datasets = ['mvtec','visa']
    datasets = ['visa']
    cols = ['category', 'i_roc', 'p_roc', 'p_pro', 'i_f1', 'p_f1', 'r_f1', 'p_threshold', "image_AUPR"]

    for dataset in datasets:

        classes = dataset_classes[dataset]
        for layer in [7]:
            for facet in ["token"]:
                for exp_index in [0,1,2]:
                    for k in [1,2,3,4]:
                        csv_path = f"result_SeDIM/facet_{facet}_layer_{layer}/csv/visa-k-{k}--exp{exp_index}.csv"
                        if not os.path.exists(csv_path):
                            print("path missing")
                            print(csv_path)
                            print(" ")
                        # df = pd.read_csv(csv_path)
                        # df = df.rename(columns={'Unnamed: 0': 'category'})
                        # # if df.shape[1] > 8:
                        #     rows = df[df[cols].isnull().any(axis=1)]
                        #     nan_rows = rows.index.values.tolist()
                        # else:
                        #     nan_rows = df.index.values.tolist()
                        for cls in classes:
                            # print(nan_rows)
                            # cls = classes[cls]
                            sh_method = f'python add_AUPR.py ' \
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

    # "C:\Users\kiran\OneDrive - Tulane University\semesters\2022 Fall\cv\codes\Segmentation\Segmentation\dino-vit-features-main\SeDIM_new\"