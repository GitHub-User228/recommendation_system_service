import warnings

warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from scipy.sparse import load_npz

from scripts import logger
from scripts.utils import read_pkl
from scripts.components.base import BaseModelComponent
from scripts.settings import get_top_items_model_component_config


class TopItemsModelComponent(BaseModelComponent):
    """
    Implements a model that recommends the top N most popular items.

    Attributes:
        config (TopItemsModelComponentConfig):
            The configuration parameters for the TopItemsModelComponent
            class.
    """

    def __init__(self) -> None:
        """
        Initializes the TopItemsModelComponent class with the
        configuration settings.
        """
        self.config = get_top_items_model_component_config()
        self._path_to_script = Path(__file__)

    def fit(self) -> None:
        """
        Retrieves the top N most popular items
        """

        # Read item encoder
        item_encoder = read_pkl(
            Path(
                self.config.source_path2,
                self.config.encoders_filenames["item"],
            )
        )

        # Read user_items matrix and calculate item popularity
        popularity = load_npz(
            Path(
                self.config.source_path2,
                self.config.user_items_matrix_filename,
            )
        )
        popularity = popularity.sum(axis=0).A1 / popularity.nnz

        # Retrieve the most popular items
        top_items = np.argpartition(
            -popularity,
            kth=self.config.top_n_items,
        )[: self.config.top_n_items]
        top_scores = popularity[top_items].tolist()

        # Save the top items to a dataframe
        df = pd.DataFrame(
            {
                self.config.fields_id["item"]: item_encoder.inverse_transform(
                    top_items
                ),
                self.config.score_col: top_scores,
            }
        )
        logger.info(
            f"Retrieved top {self.config.top_n_items} most popular items"
        )

        # Save the data
        df.to_parquet(
            Path(
                self.config.destination_path,
                self.config.top_items_filename,
            )
        )
        logger.info("Saved top items data")

    def recommend(self, subset: Literal["all", "target", "test"]) -> None:
        """
        Generate recommendations for the specified subset.

        Args:
            subset (Literal["all", "target", "test"]):
                The subset of data for which recommendations are to be
                generated.
        """

        # Retrieve users
        if subset in ["test", "target"]:
            user_ids = (
                pd.read_parquet(
                    Path(
                        self.config.source_path,
                        self.config.events_filenames[subset],
                    ),
                    columns=[self.config.fields_id["user"]],
                )[self.config.fields_id["user"]]
                .unique()
                .tolist()
            )
        elif subset == "all":
            user_ids = read_pkl(
                Path(
                    self.config.source_path2,
                    self.config.encoders_filenames["user"],
                )
            ).classes_.tolist()
        logger.info(f"Retrieved user IDs for subset '{subset}'")

        # Read the top items data keeping only top `n_recommendations`
        # items
        top_items = pd.read_parquet(
            Path(
                self.config.destination_path,
                self.config.top_items_filename,
            )
        ).iloc[: self.config.n_recommendations]
        logger.info("Loaded top items data")

        # Recommendations
        recommendations = pd.concat(
            [top_items] * len(user_ids), ignore_index=True
        )
        recommendations[self.config.fields_id["user"]] = np.repeat(
            user_ids, len(top_items)
        )

        # Save recommendations
        recommendations.to_parquet(
            Path(
                self.config.destination_path,
                self.config.recommendations_filenames[subset],
            )
        )
        logger.info(f"Saved recommendations for subset '{subset}'")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--stage",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.stage == 1:
        TopItemsModelComponent().fit()
    elif args.stage == 2:
        TopItemsModelComponent().recommend(subset="target")
    elif args.stage == 3:
        TopItemsModelComponent().recommend(subset="test")
    elif args.stage == 4:
        TopItemsModelComponent().recommend(subset="all")
    elif args.stage == 5:
        TopItemsModelComponent().evaluate()
    elif args.stage == 6:
        TopItemsModelComponent().log()
    else:
        raise ValueError("Invalid --stage. Must be 1, 2, 3, 4, 5, 6")

    logger.info(
        f"TopItemsModelComponent stage {args.stage} completed successfully"
    )


if __name__ == "__main__":
    main()
