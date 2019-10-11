import pandas as pd
import os


def standard_data(root, model):
    if model != 'test':
        all_files, all_labels = [], []
        for root, dirs, files in os.walk(os.path.join(root, model)):
            for file in files:
                all_files.append(os.path.join(root, file))
                all_labels.append(os.path.join(root, file).split("/")[-2])
        # print(pd.DataFrame({"filename": all_files, "label": all_labels}))
        return pd.DataFrame({"filename": all_files, "label": all_labels})
    elif model == 'test':
        all_files = []
        for root, dirs, files in os.walk(os.path.join(root, model)):
            for file in files:
                all_files.append(os.path.join(root, file))
        return pd.DataFrame({"filename": all_files})
    else:
        assert model == 'train' and model == 'val' and model == 'test', 'input data model error'

# standard_data('/mnt/HD_2TB/hua/github/learning_torch/cnn/data/256_ObjectionCategoties', 'train')
