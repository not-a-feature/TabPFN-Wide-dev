import os
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from analysis.load_mm_data import load_multiomics
from tabpfnwide.classifier import TabPFNWideClassifier
import warnings

warnings.filterwarnings("ignore")
import argparse


def main(
    dataset_name,
    output_file,
    checkpoint_path,
    device="cuda:0",
    omic="mrna",
):
    """
    Extracts and saves attention maps from a trained transformer-based model on multi-omics data.
    Parameters:
        dataset_name (str): Name of the multi-omics dataset to use.
        output_file (str): Path to save the extracted attention maps (as a torch file).
        checkpoint_path (str): Path to the model checkpoint to load weights from.

        device (str, optional): Device to run the model on (default: "cuda:0").
        omic (str, optional): Omics data type to use from the dataset (default: "mrna").
    Description:
        - Loads multi-omics data and encodes labels.
        - Loads a TabPFN-Wide model.
        - Configures the model to save attention maps during inference.
        - Runs inference to obtain predictions and attention maps.
        - Saves the extracted attention maps to the specified output file.
    """
    ds_dict, labels = load_multiomics(dataset_name)
    mrna = ds_dict[omic]
    X, y = mrna.values, labels
    y = LabelEncoder().fit_transform(y)
    print(X.shape)

    clf = TabPFNWideClassifier(
        model_name=checkpoint_path,
        device=device,
        n_estimators=1,
        features_per_group=1,
        save_attention_maps=True,
    )

    model = clf.model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    clf.predict_proba(X_test)

    maps = clf.get_attention_maps().mean(maps, axis=0)
    maps = torch.from_numpy(maps)
    torch.save(maps, f"{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="Path to data directory (unused but kept for compatibility)",
    )
    parser.add_argument("output_file", type=str, help="Path to save the attention maps")
    parser.add_argument(
        "--dataset", dest="dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the model checkpoint"
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--omic", type=str, default="mrna", help="Omic type to use (default: mRNA)")

    args = parser.parse_args()
    main(
        args.dataset_name,
        args.output_file,
        args.checkpoint_path,
        args.device,
        args.omic,
    )
