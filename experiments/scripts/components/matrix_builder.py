import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import LabelEncoder

from scripts import logger
from scripts.utils import save_pkl, read_pkl
from scripts.components.base import BaseComponent
from scripts.settings import get_matrix_builder_component_config


class MatrixBuilderComponent(BaseComponent):
    """
    Builds and saves encoders for user, item and item-related features
    IDs, user-items matrix and item-features matrices.

    Attributes:
        config (MatrixBuilderComponentConfig):
            The configuration parameters for the MatrixBuilderComponent
            class.
        print_info (bool):
            Whether to print a matrix info (e.g. sparsity).
    """

    def __init__(self, print_info: bool = False) -> None:
        """
        Initializes the MatrixBuilder class.
        """
        self.config = get_matrix_builder_component_config()
        self.print_info = print_info
        self._path_to_script = Path(__file__)

    def fit_encoders(self) -> None:
        """
        Fits and saves the encoders for user and item IDs based on the
        unique IDs found in the events_train dataset. The process is
        performed iteratively using a batch-wise approach.

        Raises:
            FileNotFoundError:
                If no parquet files are found for the events_train
                dataset.
        """

        # Get all parquet files within events_train directory
        files = list(
            Path(
                self.config.source_path,
                self.config.events_filenames["train"],
            ).glob("*.parquet")
        )
        if len(files) == 0:
            msg = "No parquet files found for events_train dataset"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Initialize sets to store the ids
        user_ids = set()
        item_ids = set()

        # Iterate over all partitions
        for file in tqdm(
            files, total=len(files), desc="Processing partitions"
        ):

            # Iterate over batches within a file
            for batch in pq.ParquetFile(
                Path(
                    self.config.source_path,
                    self.config.events_filenames["train"],
                    file,
                )
            ).iter_batches(
                batch_size=self.config.batch_size,
                columns=[
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ],
            ):

                # Update the sets with unique user and item IDs
                user_ids.update(
                    pa.compute.unique(
                        batch.column(self.config.fields_id["user"])
                    ).to_pylist()
                )
                item_ids.update(
                    pa.compute.unique(
                        batch.column(self.config.fields_id["item"])
                    ).to_pylist()
                )
        logger.info("Processed all partitions")

        # Fit and save the item id encoder
        encoder_item = LabelEncoder()
        encoder_item.fit(list(item_ids))
        logger.info("Fitted an encoder for item ID")
        save_pkl(
            path=Path(
                self.config.destination_path,
                self.config.encoders_filenames["item"],
            ),
            model=encoder_item,
        )

        # Fit the user id encoder
        encoder_user = LabelEncoder()
        encoder_user.fit(list(user_ids))
        logger.info("Fitted an encoder for user ID")
        save_pkl(
            path=Path(
                self.config.destination_path,
                self.config.encoders_filenames["user"],
            ),
            model=encoder_user,
        )

    def build_user_items_matrix(self) -> None:
        """
        Builds and saves the user-item matrix. The process is performed
        iteratively using a batch-wise approach.

        Raises:
            FileNotFoundError:
                If no parquet files are found for the events_train
                dataset.
        """

        # Read the encoders
        encoder_user = read_pkl(
            path=Path(
                self.config.destination_path,
                self.config.encoders_filenames["user"],
            )
        )
        encoder_item = read_pkl(
            path=Path(
                self.config.destination_path,
                self.config.encoders_filenames["item"],
            )
        )

        # Initialize an empty sparse matrix
        user_items_matrix = csr_matrix(
            (len(encoder_user.classes_), len(encoder_item.classes_)),
            dtype=np.uint8,
        )
        logger.info("Initialized empty user-items sparse matrix")

        # Get all files in the events_train directory
        files = list(
            Path(
                self.config.source_path,
                self.config.events_filenames["train"],
            ).glob("*.parquet")
        )
        if len(files) == 0:
            msg = "No parquet files found for events_train dataset"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Iterate over all files
        for file in tqdm(
            files,
            total=len(files),
            desc="Processing partitions",
        ):

            # Load entire parquet file into a DataFrame
            df = pd.read_parquet(
                Path(
                    self.config.source_path,
                    self.config.events_filenames["train"],
                    file,
                ),
                columns=[
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ],
            )

            # Transform user and item IDs
            user_indices = encoder_user.transform(
                df[self.config.fields_id["user"]]
            )
            item_indices = encoder_item.transform(
                df[self.config.fields_id["item"]]
            )

            # Fill in the sparse matrix
            user_items_matrix[user_indices, item_indices] = 1

        logger.info("Build user-items sparse matrix")

        # Save the user_items matrix
        save_npz(
            file=Path(
                self.config.destination_path,
                self.config.user_items_matrix_filename,
            ),
            matrix=user_items_matrix,
        )

        if self.print_info:
            n_users = user_items_matrix.shape[0]
            n_items = user_items_matrix.shape[1]
            n_events = user_items_matrix.nnz
            sparsity = round((1 - n_events / (n_users * n_items)) * 100, 2)
            logger.info(
                f"user-items matrix info: "
                f"number of users - {n_users}, "
                f"number of items - {n_items}, "
                f"number of events - {n_events}, "
                f"sparsity - {sparsity}%"
            )

    def build_item_features_matrices(self) -> None:
        """
        Builds and saves the item-feature matrices for the given item
        features.
        """

        # Read the item ID encoder
        encoder_item = read_pkl(
            path=Path(
                self.config.destination_path,
                self.config.encoders_filenames["item"],
            )
        )

        # Process each feature separately
        for feat, filename in tqdm(
            self.config.item_features_filenames.items(),
            total=len(self.config.item_features_filenames),
            desc="Processing features",
        ):

            # Read the item-feature data
            df = pd.read_parquet(
                Path(
                    self.config.source_path,
                    filename,
                )
            )
            logger.info(f"Read item-{feat} dataframe")

            # Filter data to only include items that exist in the
            # encoder (i.e. in the training set)
            size = len(df)
            df = df[
                df[self.config.fields_id["item"]].isin(encoder_item.classes_)
            ]
            rate = round(len(df) / size * 100, 2)
            logger.info(f"Left with {len(df)} ({rate}%) training items")

            # Encode item IDs
            df[self.config.fields_id["item"]] = encoder_item.transform(
                df[self.config.fields_id["item"]]
            )

            # Fit and save an encoder to encode the feature IDs
            encoder = LabelEncoder()
            df[self.config.fields_id[feat]] = encoder.fit_transform(
                df[self.config.fields_id[feat]]
            )
            logger.info(f"Fitted an encoder for feature '{feat}'")
            save_pkl(
                path=Path(
                    self.config.destination_path,
                    self.config.encoders_filenames[feat],
                ),
                model=encoder,
            )

            # Build the item-feature matrix
            item_features_matrix = csr_matrix(
                (
                    np.ones(len(df)),
                    (
                        df[self.config.fields_id["item"]],
                        df[self.config.fields_id[feat]],
                    ),
                ),
                shape=(
                    len(encoder_item.classes_),
                    len(encoder.classes_),
                ),
                dtype=np.uint8,
            )
            logger.info(f"Built item-{feat} sparse matrix")

            # Save the item-feature matrix
            save_npz(
                file=Path(
                    self.config.destination_path,
                    self.config.item_feature_matrix_filenames[feat],
                ),
                matrix=item_features_matrix,
            )

            if self.print_info:
                n_items = item_features_matrix.shape[0]
                n_features = item_features_matrix.shape[1]
                data_count = item_features_matrix.nnz
                sparsity = round(
                    (1 - data_count / (n_items * n_features)) * 100, 4
                )
                logger.info(
                    f"item-{feat} matrix info: "
                    f"number of items - {n_items}, "
                    f"number of features - {n_features}, "
                    f"number of data points - {data_count}, "
                    f"sparsity - {sparsity}%"
                )

    def build(self, log: bool = False) -> None:
        """
        Builds and saves the user-items matrix and the item-features
        matrices.
        """

        # Fit user ID encoder and itemID encoder
        self.fit_encoders()

        # Build user-item matrix
        self.build_user_items_matrix()

        # Build item-feature matrices
        self.build_item_features_matrices()

        # Log the run if requested
        if log:
            self.log()

        logger.info("Finished building all matrices")


if __name__ == "__main__":

    MatrixBuilderComponent(print_info=True).build(log=True)

    logger.info(f"MatrixBuilderComponent completed successfully")
