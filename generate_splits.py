import os
import json
import numpy as np
import argparse
from src.datasets.adult_income import Adult
from src.datasets.aloi import Aloi
from src.datasets.blog import Blog
from src.datasets.california_housing import California
from src.datasets.helena import Helena
from src.datasets.higgs import Higgs
from src.datasets.jannis import Jannis
from src.datasets.mnist import MNIST

DATASET_NAME_TO_DATASET_MAP = {
    "adult": Adult,
    "aloi": Aloi,
    "california": California,
    "helena": Helena,
    "higgs": Higgs,
    "jannis": Jannis,
    "mnist": MNIST,
    "blog": Blog,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets in specified structure."
    )
    parser.add_argument(
        "--dataset", required=True, help="Name of the dataset to process"
    )
    parser.add_argument("--data_path", default="./datasets", help="Path to datasets")
    parser.add_argument(
        "--output_path", default="data", help="Output directory for processed datasets"
    )
    args = parser.parse_args()

    # Get dataset class from the map
    if args.dataset not in DATASET_NAME_TO_DATASET_MAP:
        raise ValueError(
            f"Dataset '{args.dataset}' is not supported. Choose from {list(DATASET_NAME_TO_DATASET_MAP.keys())}."
        )

    # Load the dataset
    dataset_class = DATASET_NAME_TO_DATASET_MAP[args.dataset]
    dataset = dataset_class(argparse.Namespace(**{"data_path": args.data_path}))
    dataset.load()
    X, y = dataset.X, dataset.y

    # Split the dataset
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.375, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.555, random_state=42
    )

    # Output directory
    output_dir = os.path.join(args.output_path, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Save splits as .npy files
    np.save(os.path.join(output_dir, "N_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "N_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "N_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Save test indices
    idx_test = np.arange(len(y))[len(y_train) + len(y_val) :]
    np.save(os.path.join(output_dir, "idx_test.npy"), idx_test)

    # Save info.json
    info = {
        "name": f"{args.dataset}___0",
        "basename": args.dataset,
        "split": 0,
        "task_type": "multiclass",
        "n_num_features": X.shape[1],
        "n_cat_features": 0,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "n_classes": len(np.unique(y)),
    }

    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"Processed dataset '{args.dataset}' saved at {output_dir}")


if __name__ == "__main__":
    main()
