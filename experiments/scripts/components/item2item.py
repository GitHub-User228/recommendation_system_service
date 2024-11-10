import os
import warnings

warnings.filterwarnings("ignore")

import argparse
import cloudpickle
import pandas as pd
from pathlib import Path
from typing import Literal
from scipy.sparse import load_npz, hstack


from scripts import models, logger
from scripts.utils import read_pkl
from scripts.components.base import BaseModelComponent
from scripts.settings import get_item2item_model_component_config


class Item2ItemModelComponent(BaseModelComponent):
    """
    Implements the Item2Item model component for the recommendation
    system.

    Attributes:
        config (Item2ItemModelComponentConfig):
            Configuration for the Item2Item model component.
    """

    def __init__(self) -> None:
        """
        Initializes the Item2ItemModelComponent class with the
        configuration settings.
        """
        self.config = get_item2item_model_component_config()
        self._path_to_script = Path(__file__)

    def fit(self) -> None:
        """
        Fits the Item2Item model to the item-features data, saves
        the trained model.
        """

        # Reading the encoders and matrices
        item_encoder = read_pkl(
            Path(
                self.config.source_path2,
                self.config.encoders_filenames["item"],
            )
        )
        user_encoder = read_pkl(
            Path(
                self.config.source_path2,
                self.config.encoders_filenames["user"],
            )
        )
        user_items_matrix = load_npz(
            Path(
                self.config.source_path2,
                self.config.user_items_matrix_filename,
            )
        )
        filenames = self.config.item_feature_matrix_filenames.values()
        item_features_matrix = hstack(
            [
                load_npz(
                    Path(
                        self.config.source_path2,
                        filename,
                    )
                )
                for filename in filenames
            ]
        )

        # Fit the model
        model = models.Item2ItemModel(
            min_users_per_item=self.config.min_users_per_item,
            n_neighbors=self.config.n_neighbors,
            n_components=self.config.n_components,
            similarity_criteria=self.config.similarity_criteria,
        )
        model.fit(
            item_features_matrix=item_features_matrix,
            user_items_matrix=user_items_matrix,
            item_id_encoder=item_encoder,
            user_id_encoder=user_encoder,
            n_jobs=os.cpu_count(),
        )

        # Save the model
        with open(
            Path(
                self.config.destination_path,
                self.config.model_filename,
            ),
            "wb",
        ) as f:
            cloudpickle.register_pickle_by_value(models)
            cloudpickle.dumps(models.BaseRecommender)
            cloudpickle.dumps(models.Item2ItemModel)
            cloudpickle.dump(model, f)

    def recommend(self, subset: Literal["target", "test"]) -> None:
        """
        Generate recommendations for the specified subset.

        Args:
            subset (Literal["target", "test"]):
                The subset of data for which recommendations are to be
                generated.
        """

        # Loading user_items_matrix
        user_items_matrix = load_npz(
            Path(
                self.config.source_path2,
                self.config.user_items_matrix_filename,
            )
        )
        logger.info("Loaded user_items_matrix")

        # Retrieve users
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
        logger.info(f"Retrieved user IDs for subset '{subset}'")

        # Loading the trained model
        model = read_pkl(
            Path(self.config.destination_path, self.config.model_filename)
        )

        model.recommend(
            user_ids=user_ids,
            user_items_matrix=user_items_matrix,
            n_recommendations=self.config.n_recommendation,
            user_id_col=self.config.fields_id["user"],
            item_id_col=self.config.fields_id["item"],
            score_col=self.config.score_col_name,
            batch_size=self.config.batch_size,
            save_path=Path(
                self.config.destination_path,
                self.config.recommendations_filenames[subset],
            ),
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
        Item2ItemModelComponent().fit()
    elif args.stage == 2:
        Item2ItemModelComponent().recommend(subset="target")
    elif args.stage == 3:
        Item2ItemModelComponent().recommend(subset="test")
    elif args.stage == 4:
        Item2ItemModelComponent().evaluate()
    elif args.stage == 5:
        Item2ItemModelComponent().log()
    else:
        raise ValueError("Invalid --stage. Must be 1, 2, 3, 4, 5")

    logger.info(
        f"Item2ItemModelComponent stage {args.stage} completed successfully"
    )


if __name__ == "__main__":
    main()
