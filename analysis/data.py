from sklearn.model_selection import RepeatedStratifiedKFold
from tabicl.prior.dataset import PriorDataset
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from .load_mm_data import (
    load_multiomics_benchmark_shamir,
    load_multiomics_benchmark_ds,
    ALL_MULTIOMICS_DATASETS,
    ALL_MULTIOMICS_DATASETS_SHAMIR,
)
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_openml

from .utils import feature_reduction_agglomeration

from tabpfnwide.config import PriorDatasetConfig, PriorDataLoaderConfig


def load_prior_dataloader(
    dataset: PriorDataset,
    config_dataset: PriorDatasetConfig,
    config_dataloader: PriorDataLoaderConfig,
):
    return DataLoader(dataset(**config_dataset.__dict__), **config_dataloader.__dict__)


def get_wide_validation_datasets(
    device, dataset_name="gbm", n_splits=5, n_repeats=1, reduced_features=0, omics=["mrna"]
):
    """
    Returns a generator that yields validation datasets the specified dataset.
    """
    print(f"\n=== DEBUG: get_wide_validation_datasets ===")
    print(f"Dataset: {dataset_name}, omics: {omics}, reduced_features: {reduced_features}")

    if dataset_name in ALL_MULTIOMICS_DATASETS:
        ds = load_multiomics_benchmark_ds(dataset_name, preprocessing="Original")
    elif dataset_name in ALL_MULTIOMICS_DATASETS_SHAMIR:
        ds = load_multiomics_benchmark_shamir(dataset_name, normalize=True, subtype_labels=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not found in multiomics datasets.")

    x_list = [ds[omic] for omic in omics]
    X, y = pd.concat(x_list, axis=1), ds["labels"]

    print(f"Before feature reduction - X shape: {X.shape}, y shape: {y.shape}")

    if reduced_features > X.shape[-1]:
        print(
            f"Skipping {dataset_name} with {reduced_features} features, not enough features in dataset"
        )
        yield from ()
    else:
        if reduced_features > 0:
            X = feature_reduction_agglomeration(X, n_features=reduced_features).values
            print(f"After feature reduction - X shape: {X.shape}")
        else:
            X = X.values
            print(f"No feature reduction - X shape: {X.shape}")

        X = X.astype(np.float32)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"After numpy conversion - X shape: {X.shape}, X dtype: {X.dtype}")
        yield from generate_tensor_folds(X, y, device, n_splits=n_splits, n_repeats=n_repeats)


def generate_tensor_folds(X, y, device, n_splits=5, n_repeats=1):
    print(f"\n=== DEBUG: generate_tensor_folds ===")
    print(f"Input X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Input y shape: {y.shape}, dtype: {y.dtype}")

    for train_idx, test_idx in RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    ).split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\nFold split - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.int8).unsqueeze(1).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.int8).unsqueeze(1).to(device)

        print(f"After unsqueeze - X_train_tensor shape: {X_train_tensor.shape}")
        print(f"After unsqueeze - y_train_tensor shape: {y_train_tensor.shape}")
        print(f"WARNING: X_train_tensor is 3D! This will cause issues with sklearn/TabPFN")

        yield X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def get_wide_validation_data(device, validation_datasets, omics_combinations):
    for dataset_name in validation_datasets:
        for omic_list in omics_combinations:
            print(f"Using omics: {omic_list} for dataset {dataset_name}")
            yield from get_wide_validation_datasets(
                device, dataset_name=dataset_name, omics=omic_list
            )
