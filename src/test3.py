import numpy as np
import pandas as pd
from load import load_GSD, load_data
from utils import try_gpu


print("Importing human protein sequences...", end = "")
data_path = "../data/"
dataset_path = "../data/dataset/"
uniprot = pd.read_csv(data_path + "uniprot.tsv", sep="\t")
print(" Done.")

dataset_train, dataset_test = load_GSD(uniprot, dataset_path)
AC_train = dataset_train[:, :2]
X_train = dataset_train[:, 3:].astype(np.int64)
Y_train = dataset_train[:, 2].astype(np.int64).reshape([-1, 1])
AC_test = dataset_test[:, :2]
X_test = dataset_test[:, 3:].astype(np.int64)
Y_test = dataset_test[:, 2].astype(np.int64).reshape([-1, 1])
X_all = np.concatenate([X_train, X_test])
Y_all = np.concatenate([Y_train, Y_test])
print(uniprot,data_path)
features, adj_norm, adj_label = load_data(uniprot, data_path, is_CT = True)
features = features.to(device=try_gpu())
adj_norm = adj_norm.to(device=try_gpu())
adj_label = adj_label.to(device=try_gpu())
print(features)
print(features.size)
print(adj_label)
print(adj_norm)