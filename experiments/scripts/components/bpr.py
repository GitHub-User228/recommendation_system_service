import warnings

warnings.filterwarnings("ignore")

import argparse
import cloudpickle
import pandas as pd
from pathlib import Path
from typing import Literal
from scipy.sparse import load_npz

from scripts import logger
from scripts import models
from scripts.utils import read_pkl
from scripts.components.base import BaseModelComponent
from scripts.settings import get_bpr_model_component_config


class BPRModelComponent(BaseModelComponent):
    """
    Implements the Bayesian Personalized Ranking (BPR) model component
    for the recommendation system.

    Attributes:
        config (BPRModelComponentConfig):
            Configuration for the BPR model component.
    """

    def __init__(self) -> None:
        """
        Initializes the BPRModelComponent class with the configuration
        settings.
        """
        self.config = get_bpr_model_component_config()
        self._path_to_script = Path(__file__)

    def fit(self) -> None:
        """
        Fits an Bayesian Personalized Ranking (BPR) model to the
        user-item interaction data, saves the trained model and
        generated similar items data.
        """

        # Reading the encoders and user_items matrix
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

        # Creating the BPR model and fitting it to the data
        model = models.BPR(
            min_users_per_item=self.config.min_users_per_item,
            factors=self.config.factors,
            learning_rate=self.config.learning_rate,
            iterations=self.config.iterations,
            regularization=self.config.regularization,
            verify_negative_samples=self.config.verify_negative_samples,
            random_state=self.config.random_state,
        )
        model.fit(
            user_items_matrix=user_items_matrix,
            item_id_encoder=item_encoder,
            user_id_encoder=user_encoder,
        )
        logger.info("Trained BPR model")

        # Saving the trained BPR model using cloudpickle
        with open(
            Path(self.config.destination_path, self.config.model_filename),
            "wb",
        ) as f:
            cloudpickle.register_pickle_by_value(models)
            cloudpickle.dumps(models.BaseRecommender)
            cloudpickle.dumps(models.BPR)
            cloudpickle.dump(model, f)
        logger.info("Saved BPR model")

        # Generating similar items
        similar_items_df = model.get_similar_items(
            max_similar_items=self.config.max_similar_items,
            item_id_col=self.config.fields_id["item"],
            item_id_col_similar=f'similar_{self.config.fields_id["item"]}',
            score_col=self.config.score_col,
        )
        logger.info("Generated similar items")

        # Saving data
        similar_items_df.to_parquet(
            Path(
                self.config.destination_path,
                self.config.similar_items_filename,
            )
        )
        logger.info("Saved similar items data")

    def recommend(self, subset: Literal["all", "target", "test"]) -> None:
        """
        Generate recommendations for the specified subset.

        Args:
            subset (Literal["all", "target", "test"]):
                The subset of data for which recommendations are to be
                generated.
        """

        # Read the user_items matrix
        user_items_matrix = load_npz(
            Path(
                self.config.source_path2,
                self.config.user_items_matrix_filename,
            )
        )
        logger.info("Loaded user_items_matrix")

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

        # Load the trained BPR model
        model = read_pkl(
            Path(self.config.destination_path, self.config.model_filename)
        )

        # Recommendations
        recommendations = model.recommend(
            user_ids=user_ids,
            user_items_matrix=user_items_matrix,
            n_recommendations=self.config.n_recommendations,
            user_id_col=self.config.fields_id["user"],
            item_id_col=self.config.fields_id["item"],
            score_col=self.config.score_col,
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
        BPRModelComponent().fit()
    elif args.stage == 2:
        BPRModelComponent().recommend(subset="target")
    elif args.stage == 3:
        BPRModelComponent().recommend(subset="test")
    elif args.stage == 4:
        BPRModelComponent().recommend(subset="all")
    elif args.stage == 5:
        BPRModelComponent().evaluate()
    elif args.stage == 6:
        BPRModelComponent().log()
    else:
        raise ValueError("Invalid --stage. Must be 1, 2, 3, 4, 5, 6")

    logger.info(f"BPRModelComponent stage {args.stage} completed successfully")


if __name__ == "__main__":
    main()
