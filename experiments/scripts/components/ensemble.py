import warnings

warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Literal
from scipy.sparse import load_npz
from catboost import CatBoostClassifier

from scripts import logger
from scripts.components.base import BaseModelComponent
from scripts.utils import read_pkl, save_pkl, save_yaml
from scripts.settings import get_ensemble_model_component_config


class EnsembleModelComponent(BaseModelComponent):

    def __init__(self) -> None:
        """
        Initializes the EnsembleModelComponent class with the
        configuration settings.
        """
        self.config = get_ensemble_model_component_config()
        self._path_to_script = Path(__file__)

    def _merge_data(
        self, subset: Literal["train_df", "test_df"]
    ) -> pd.DataFrame:
        """
        Merges data from multiple base models for the specified subset.

        Args:
            subset (Literal["train_df", "test_df"]):
                The data subset to merge.

        Returns:
            pd.DataFrame:
                The merged DataFrame containing data from all base models.
        """
        if subset not in ["train_df", "test_df"]:
            raise ValueError(f"Invalid subset: {subset}")

        path_dict = self.config.train_data_path
        if subset == "test_df":
            path_dict = self.config.test_data_path

        id_cols = set(
            [
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ]
        )
        df = None
        for base_model in tqdm(
            self.config.base_models, desc=f"Merging {subset} data"
        ):
            df_base = pd.read_parquet(path_dict[base_model])
            feature_col = list(set(df_base.columns) - id_cols)[0]
            df_base[feature_col] = df_base[feature_col].astype(np.float32)
            df = (
                df_base
                if df is None
                else df.merge(
                    df_base,
                    on=[
                        self.config.fields_id["user"],
                        self.config.fields_id["item"],
                    ],
                    how="outer",
                )
            )
        logger.info(f"Merged {subset} data.")

        return df

    def _add_top_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds data from top items model to the provided DataFrame.
        """

        # Load item_encoder and user_encoder
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

        # Read top items data
        top_items = pd.read_parquet(self.config.top_items_path).iloc[
            : self.config.n_recommendations
        ]
        score_col = list(
            set(top_items.columns) - set([self.config.fields_id["item"]])
        )[0]
        top_items_dict = top_items.set_index(
            self.config.fields_id["item"]
        ).to_dict()[score_col]
        item_ids_encoded = item_encoder.transform(
            top_items[self.config.fields_id["item"]]
        )

        # Retrieve user IDs from the input dataframe
        user_ids = df[self.config.fields_id["user"]].unique().tolist()
        user_ids_encoded = user_encoder.transform(user_ids)

        # TODO
        df2 = pd.DataFrame(
            zip(
                *np.where(
                    load_npz(
                        Path(
                            self.config.source_path2,
                            self.config.user_items_matrix_filename,
                        )
                    )[:, item_ids_encoded][user_ids_encoded, :].toarray()
                    == 0
                )
            ),
            columns=[
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ],
        )

        # Decoding item ID
        df2[self.config.fields_id["item"]] = df2[
            self.config.fields_id["item"]
        ].map(dict(zip(list(range(len(item_ids_encoded))), item_ids_encoded)))
        df2[self.config.fields_id["item"]] = item_encoder.inverse_transform(
            df2[self.config.fields_id["item"]]
        )

        # Decoding user ID
        df2[self.config.fields_id["user"]] = df2[
            self.config.fields_id["user"]
        ].map(dict(zip(list(range(len(user_ids_encoded))), user_ids_encoded)))
        df2[self.config.fields_id["user"]] = user_encoder.inverse_transform(
            df2[self.config.fields_id["user"]]
        )

        # Adding score column
        df2[score_col] = df2[self.config.fields_id["item"]].map(top_items_dict)

        # # Create a dataframe containing top items for each user
        # df2 = pd.concat([top_items] * len(user_ids), ignore_index=True)
        # df2[self.config.fields_id["user"]] = np.repeat(
        #     user_ids, len(top_items)
        # )
        # df2[score_col] = df2[self.config.fields_id["item"]].map(top_items_dict)

        # Fill the new column with scores for (user ID, item ID)
        # pairs that are in the dataframe
        mask = df[self.config.fields_id["item"]].isin(
            top_items[self.config.fields_id["item"]]
        )
        df.loc[mask, score_col] = df.loc[mask][
            self.config.fields_id["item"]
        ].map(top_items_dict)

        # Remove (user ID, item ID) pairs from df2 that are in df
        df2 = (
            df2.merge(
                df.loc[mask][
                    [
                        self.config.fields_id["user"],
                        self.config.fields_id["item"],
                    ]
                ],
                on=[
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ],
                how="left",
                indicator=True,
            )
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
        )

        # Merge with the input dataframe
        df = pd.concat([df, df2], ignore_index=True)

        logger.info("Added top items data.")

        return df

    def _add_target(
        self, df: pd.DataFrame, subset: Literal["train_df", "test_df"]
    ) -> pd.DataFrame:
        """
        Adds target variable to the provided DataFrame.

        Args:
            df (pd.DataFrame):
                The DataFrame to add target variable to.
            subset (Literal["train_df", "test_df"]):
                The data subset to add target variable to.

        Returns:
            pd.DataFrame:
                The DataFrame with the added target variable.
        """
        if subset not in ["train_df", "test_df"]:
            raise ValueError(f"Invalid subset: {subset}")

        # Reading real data
        df_real = pd.read_parquet(
            Path(
                self.config.source_path,
                self.config.events_filenames[
                    "target" if subset == "train_df" else "test"
                ],
            )
        )
        df_real[self.config.target_col] = 1

        # Merging with recommendations
        df = df.merge(
            df_real,
            on=[
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ],
            how="left",
        )
        df[self.config.target_col].fillna(0, inplace=True)
        df[self.config.target_col] = df[self.config.target_col].astype(
            np.uint8
        )

        logger.info(f"Added target variable for {subset} data.")

        # Calculate the number of correct recommendations
        pos_class_cnt = df[self.config.target_col].value_counts()[1]
        ratio = round(pos_class_cnt / len(df) * 100, 2)
        logger.info(
            f"Number of positive samples for {subset} data: "
            f"{pos_class_cnt} ({ratio}%) "
        )

        return df

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the data so that all positive samples and only few
        negative samples are left for each user.

        Args:
            df (pd.DataFrame):
                The DataFrame to be filtered.

        Returns:
            pd.DataFrame:
                The filtered DataFrame.
        """

        orig_size = len(df)

        # Leave only users with at least one positive target
        mask = (
            df.groupby(self.config.fields_id["user"])[
                self.config.target_col
            ].transform("sum")
            > 0
        )
        df = df[mask]
        filtered_size = len(df)
        rate = round(filtered_size / orig_size * 100, 2)
        logger.info(
            f"Left with {filtered_size} ({rate}%) samples after "
            f"filtering users with no positive targets"
        )

        # Separate positive and negative targets
        positives = df.query(f"{self.config.target_col} == 1")
        df = df.query(f"{self.config.target_col} == 0")

        # Sampling negative targets (this way is faster)
        df = df.assign(
            rand=np.random.default_rng(self.config.sampling_seed).random(
                len(df)
            )
        ).sort_values([self.config.fields_id["user"], "rand"])
        df["rank"] = df.groupby(self.config.fields_id["user"]).cumcount() + 1
        df = df.query(f"rank <= {self.config.negative_samples_per_user}").drop(
            columns=["rand", "rank"]
        )

        # Merging positive and negative targets
        df = pd.concat([positives, df], axis=0, ignore_index=True)
        del positives

        filtered_size = len(df)
        rate = round(filtered_size / orig_size * 100, 2)
        logger.info(
            f"Left with {filtered_size} ({rate}%) samples after "
            f"sampling {self.config.negative_samples_per_user} negative "
            f"samples per user"
        )

        pos_class_cnt = df[self.config.target_col].value_counts()[1]
        ratio = round(pos_class_cnt / len(df) * 100, 2)
        logger.info(f"Number of positive samples: {pos_class_cnt} ({ratio}%)")

        return df

    def _add_features(
        self, df: pd.DataFrame, kind: Literal["user", "item"]
    ) -> pd.DataFrame:
        """
        Adds user or item features to the provided DataFrame.

        Args:
            df (pd.DataFrame):
                The DataFrame to add features to.
            kind (Literal["user", "item"]):
                Specifies whether to add user or item features.

        Returns:
            pd.DataFrame:
                The DataFrame with the added features.
        """

        if kind == "user":
            path = self.config.user_features_path
        elif kind == "item":
            path = self.config.item_features_path
        else:
            raise ValueError("Invalid kind. Must be 'user' or 'item'.")

        if (path != None) and path.exists() and path.is_file():
            df_user_features = pd.read_parquet(path)
            df = df.merge(
                df_user_features,
                on=self.config.fields_id[kind],
                how="left",
            )
            logger.info(f"Added `{kind}` features")

        return df

    def prepare_data(self) -> None:
        """
        Prepares the train and test data for the ensemble model by
        merging the base models' data, adding user and item features.
        The resulting data is saved.
        """
        for subset in ["train_df", "test_df"]:
            df = self._merge_data(subset)
            if self.config.include_top_items:
                df = self._add_top_items(df)
            df = self._add_target(df, subset)
            if subset == "train":
                df = self._filter_data(df)
            df = self._add_features(df, kind="user")
            df = self._add_features(df, kind="item")
            df.to_parquet(
                Path(
                    self.config.destination_path,
                    self.config.recommendations_filenames[subset],
                ),
            )
            logger.info(f"Prepared and saved {subset} data.")

    def fit(self) -> None:
        """
        Trains and saves the ensemble model.
        Also saves the imporance values for features
        """

        # Reading train data
        df_train = pd.read_parquet(
            Path(
                self.config.destination_path,
                self.config.recommendations_filenames["train_df"],
            ),
        )
        logger.info("Read train data")

        # Separating names of columns with features
        features = list(
            set(df_train.columns)
            - set(
                [
                    self.config.target_col,
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ]
            )
        )

        # Initialize an ensemble model
        model = globals()[self.config.model_class_name](
            **self.config.model_params
        )
        logger.info(
            f"Initialized ensemble model as a {self.config.model_class_name}"
        )

        # Fit the model
        model.fit(
            X=df_train[features],
            y=df_train[self.config.target_col],
        )
        logger.info("Fitted the ensemble model")

        # Saving the model
        save_pkl(
            path=Path(
                self.config.destination_path, self.config.model_filename
            ),
            model=model,
        )

        # Retreiving feature importances and saving them
        importances = dict(
            zip(
                model.feature_names_,
                list(map(float, model.feature_importances_)),
            )
        )
        importances = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )
        logger.info(f"Feature importances: {importances}")
        save_yaml(
            path=Path(
                self.config.destination_path,
                self.config.feature_importances_filename,
            ),
            data=importances,
        )
        logger.info("Saved feature importances")

    def recommend(self) -> None:
        """
        Ranks recommendations on the test data
        """

        # Read test data
        df = pd.read_parquet(
            Path(
                self.config.destination_path,
                self.config.recommendations_filenames["test_df"],
            ),
        )
        logger.info("Read test data")

        # Separate names of columns with features
        features = list(
            set(df.columns)
            - set(
                [
                    self.config.target_col,
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ]
            )
        )

        # Load the trained ensemble model
        model = read_pkl(
            path=Path(self.config.destination_path, self.config.model_filename)
        )

        # Predicting on the test data
        df[self.config.score_col] = model.predict_proba(df[features])[:, 1]
        logger.info("Ranked recommendations on the test data")

        # Omit columns with features
        df.drop(columns=features, inplace=True)

        # Leaving top recommendations
        df = (
            df.sort_values(
                by=[self.config.fields_id["user"], self.config.score_col],
                ascending=[True, False],
            )
            .groupby(self.config.fields_id["user"])
            .head(self.config.n_recommendations)
        )
        logger.info(
            f"Left only top {self.config.n_recommendations} recommendations"
        )

        # Saving recommendations
        df.to_parquet(
            Path(
                self.config.destination_path,
                self.config.recommendations_filenames["test"],
            ),
        )
        logger.info("Saved recommendations")


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
        EnsembleModelComponent().prepare_data()
    elif args.stage == 2:
        EnsembleModelComponent().fit()
    elif args.stage == 3:
        EnsembleModelComponent().recommend()
    elif args.stage == 4:
        EnsembleModelComponent().evaluate()
    elif args.stage == 5:
        EnsembleModelComponent().log()
    else:
        raise ValueError("Invalid --stage. Must be 1, 2, 3, 4, 5")

    logger.info(
        f"EnsembleModelComponent stage {args.stage} completed successfully"
    )


if __name__ == "__main__":
    main()
