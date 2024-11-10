import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from scripts import logger
from scripts.components.base import BaseComponent
from scripts.settings import get_preprocessing_component_config
from scripts.utils import get_spark_session, spark_parquet_to_single_parquet


class PreprocessingComponent(BaseComponent):
    """
    Preprocesses the music data to be used in the downstream tasks.

    Attributes:
        config (PreprocessingComponentConfig):
            The configuration parameters for the PreprocessingComponent
            class.
        spark (SparkSession):
            The Spark session.
        print_info (bool):
            Whether to print info about data.
    """

    def __init__(self, print_info: bool = False) -> None:
        """
        Initializes the PreprocessingComponent class.

        Args:
            print_info (bool):
                Whether to print info about data.
        """
        self.config = get_preprocessing_component_config()
        self.spark = get_spark_session()
        self.print_info = print_info
        self._path_to_script = Path(__file__)

    def read_parquet(
        self, filename: str, spark: bool = True
    ) -> DataFrame | pd.DataFrame:
        """
        Reads a Parquet file from the configured source path and
        returns a Spark DataFrame.

        Args:
            filename (str):
                The name of the Parquet file to read.

        Returns:
            DataFrame | pd.DataFrame:
                The Spark DataFrame containing the data.
        """
        if spark:
            df = self.spark.read.parquet(
                str(Path(self.config.source_path, filename))
            )
        else:
            df = pd.read_parquet(Path(self.config.source_path, filename))
        logger.info(f"Read {filename} from {self.config.source_path}.")
        return df

    def save_parquet(
        self, df: DataFrame | pd.DataFrame, filename: str
    ) -> None:
        """
        Writes a Spark DataFrame to a Parquet file at the configured
        destination path.

        Args:
            df (DataFrame | pd.DataFrame):
                The Spark DataFrame to write to Parquet.
            filename (str):
                The name of the Parquet file to write.
        """
        if isinstance(df, DataFrame):
            df.write.mode("overwrite").parquet(
                str(Path(self.config.destination_path, filename)),
            )
        else:
            df.to_parquet(Path(self.config.destination_path, filename))
        logger.info(f"Saved {filename} to {self.config.destination_path}.")

    def print_table_info(self, df: DataFrame | pd.DataFrame) -> None:
        """
        Prints the info and statistics of a DataFrame.

        Args:
            df (DataFrame | pd.DataFrame):
                The Spark DataFrame or Pandas DataFrame to print
                info and statistics of.
        """
        if isinstance(df, DataFrame):
            print("-- schema --")
            df.printSchema()
            print("-- sample --")
            df.show(5)
            print("-- missing data count --")
            df.select(
                [
                    F.count(F.when(F.col(c).isNull(), c)).alias(c)
                    for c in df.columns
                ]
            ).show()
        else:
            print("-- info --")
            print(df.info(), end="\n\n")
            print("-- sample --")
            try:
                print(df.head().to_markdown(), end="\n\n")
            except:
                print(df.head(), end="\n\n")
            print("-- missing data count --")
            print(df.isnull().sum().to_markdown(), end="\n\n")

    def preprocess_catalog_names(self) -> None:
        """
        Preprocesses the catalog names data by performing the following steps:
        1. Prints information about the catalog names data
        2. For each feature (e.g. 'genre', 'album', etc.):
           - Extracts the feature names data from the catalog names data.
           - Renames the columns to match the expected format.
           - Saves the resulting dataframe to a Parquet file.
           - Prints information about the feature names dataframe.
        """

        catalog_names = self.read_parquet(
            self.config.catalog_names_filename, spark=False
        )
        catalog_names["id"] = catalog_names["id"].astype(np.uint32)

        if self.print_info:
            print("----- catalog_names dataframe info -----\n")

            # General info
            self.print_table_info(catalog_names)

            # Duplicates info along `id` and `type`
            print("-- duplicates count --")
            n_old = catalog_names.shape[0]
            n_dups = catalog_names.duplicated(subset=["id", "type"]).sum()
            rate = round(n_dups / n_old * 100, 2)
            print(f"along ('id', 'type'): {n_dups} ({rate}%)")

            # Duplicates info along `name` for each `type`
            for group in catalog_names["type"].unique():
                n_old = catalog_names.query("type == @group").shape[0]
                n_dups = (
                    catalog_names.query("type == @group")
                    .duplicated(subset=["name"])
                    .sum()
                )
                rate = round(n_dups / n_old * 100, 2)
                print(f'along "name" for "type" {group}: {n_dups} ({rate}%)')
            print("\n")

            # `None-like` values info
            print("-- `None-like` values count -- ")
            print(
                catalog_names[
                    catalog_names["name"]
                    .str.lower()
                    .str.contains(r"^none$", regex=True)
                ]
                .groupby(["type", "name"])
                .count()
                .to_markdown(),
                end="\n\n",
            )

        for feature, filename in self.config.name_df_filenames.items():

            feature_names_df = (
                catalog_names.query(
                    f'type == "{self.config.fields_name[feature]}"'
                )
                .rename(
                    columns={
                        "id": self.config.fields_id[feature],
                        "name": self.config.fields_name[feature],
                    }
                )
                .drop(columns="type")
            )

            self.save_parquet(
                df=feature_names_df,
                filename=filename,
            )

            if self.print_info:
                print(f"\n----- {feature}_names dataframe info -----")

                # IDs range info
                print(
                    feature_names_df[self.config.fields_id[feature]]
                    .agg(["min", "max", pd.Series.nunique])
                    .to_markdown(),
                    end="\n\n",
                )

    def preprocess_tracks(self) -> None:
        """
        Preprocesses the tracks data by performing the following steps:
        1. Prints information about the tracks data
        2. Splits the tracks data depending on the track features
            (album, artist, genre).
        3. For each feature (album, artist, genre):
            - Extracts the feature IDs from the tracks data by exploding
            the corresponding column.
            - Saves the resulting dataframe to a parquet file.
            - Prints information about the dataframe
        """

        items = self.read_parquet(self.config.tracks_filename, spark=False)
        items_count = items.shape[0]
        items[self.config.fields_id["item"]] = items[
            self.config.fields_id["item"]
        ].astype(np.uint32)

        if self.print_info:
            print("----- 'tracks' dataframe info -----\n")

            # General info
            self.print_table_info(items)

            # Duplicates count
            print("-- duplicates count --")
            n_dups = items.duplicated(
                subset=self.config.fields_id["item"]
            ).sum()
            rate = round(n_dups / items_count * 100, 2)
            print(f"{n_dups} ({rate}%)\n")

        for feature in ["album", "artist", "genre"]:

            item_features_df = (
                items[
                    [
                        self.config.fields_id["item"],
                        self.config.groups[feature],
                    ]
                ]
                .apply(pd.Series.explode)
                .rename(
                    columns={
                        self.config.groups[feature]: self.config.fields_id[
                            feature
                        ]
                    }
                )
                .dropna()
            )
            item_features_df[self.config.fields_id[feature]] = (
                item_features_df[self.config.fields_id[feature]].astype(
                    np.uint32
                )
            )

            self.save_parquet(
                df=item_features_df,
                filename=self.config.item_features_filenames[feature],
            )

            if self.print_info:
                print(f"\n----- track_{feature} dataframe info -----\n")

                # General info
                self.print_table_info(item_features_df)

                # Duplicates info
                print("-- duplicates count --")
                n_old = item_features_df.shape[0]
                n_dups = item_features_df.duplicated().sum()
                rate = round(n_dups / n_old * 100, 2)
                print(f"{n_dups} ({rate}%)\n")

                # Number of tracks per feature
                print(f"-- number of items per {feature} --")
                tracks_count2 = item_features_df[
                    self.config.fields_id["item"]
                ].nunique()
                rate = round(tracks_count2 / items_count * 100, 4)
                print(f"{tracks_count2} ({rate}%)\n")

    def preprocess_interactions(self) -> None:
        """
        Preprocesses the interactions data by performing the following steps:
        1. Splits the events into train and test sets based on the
            `train_test_split_date` configuration.
        2. Retrieves the users that are in both the train and test sets,
            and the users that are only in the test set.
        3. Splits the test set into a target set and a remaining test
            set based on the `target_test_split_date` configuration.
        4. Saves the preprocessed data to parquet files.
        5. Prints information about the interactions data, the train set,
            and the test set.
        """

        # Read the parquet file with events
        interactions = self.read_parquet(self.config.interactions_filename)

        # Train test split
        events_train = interactions.filter(
            F.col(self.config.date_col) < self.config.train_test_split_date
        )
        events_test = interactions.filter(
            F.col(self.config.date_col) >= self.config.train_test_split_date
        )

        # Retrieve users that are in boyj the train and test sets
        users_train = events_train.select(
            self.config.fields_id["user"]
        ).distinct()
        users_test = events_test.select(
            self.config.fields_id["user"]
        ).distinct()
        users_test_only = users_test.join(
            users_train, on=self.config.fields_id["user"], how="left_anti"
        )

        # Target test split
        events_target = events_test.filter(
            F.col(self.config.date_col) < self.config.target_test_split_date
        ).select(
            self.config.fields_id["user"],
            self.config.fields_id["item"],
        )
        events_test2 = events_test.filter(
            F.col(self.config.date_col) >= self.config.target_test_split_date
        ).select(
            self.config.fields_id["user"],
            self.config.fields_id["item"],
        )

        # Saving data
        self.save_parquet(
            df=events_train,
            filename=self.config.events_filenames["train"],
        )
        self.save_parquet(
            df=events_target.coalesce(1),
            filename=self.config.events_filenames["target"],
        )
        spark_parquet_to_single_parquet(
            path=self.config.destination_path,
            filename=self.config.events_filenames["target"],
        )
        self.save_parquet(
            df=events_test2.coalesce(1),
            filename=self.config.events_filenames["test"],
        )
        spark_parquet_to_single_parquet(
            path=self.config.destination_path,
            filename=self.config.events_filenames["test"],
        )

        if self.print_info:
            print("\n----- 'interactions' dataframe info -----\n")

            # General info
            self.print_table_info(interactions)

            # Number of events, users and items
            events_total = interactions.count()
            users_total = (
                interactions.select(self.config.fields_id["user"])
                .distinct()
                .count()
            )
            items_total = (
                interactions.select(self.config.fields_id["item"])
                .distinct()
                .count()
            )
            sparsity = round(
                (1 - events_total / (users_total * items_total)) * 100, 4
            )
            print(
                f"Number of:  events - {events_total}, "
                f"users - {users_total}",
                f"items - {items_total}",
                f"sparsity - {sparsity}%\n",
            )

            print("----- 'events_train' dataframe info -----")

            # Number of events, users
            events_count = events_train.count()
            rate1 = round(events_count / events_total * 100, 2)
            users_count = users_train.count()
            rate2 = round(users_count / users_total * 100, 2)
            items_count = (
                events_train.select(self.config.fields_id["item"])
                .distinct()
                .count()
            )
            rate3 = round(items_count / items_total * 100, 2)
            sparsity = round(
                (1 - events_count / (users_count * items_count)) * 100, 4
            )
            print(
                f"Number of: events - {events_count} ({rate1}%), "
                f"users - {users_count} ({rate2}%)",
                f"items - {items_count} ({rate3}%)\n",
                f"sparsity - {sparsity}%\n",
            )

            print("----- 'events_test' dataframe info -----")

            # Number of events, users
            events_count = events_test.count()
            rate1 = round(events_count / events_total * 100, 2)
            users_count = users_test.count()
            rate2 = round(users_count / users_total * 100, 2)
            users_count2 = users_test_only.count()
            rate3 = round(users_count2 / users_total * 100, 2)
            items_count = (
                events_test.select(self.config.fields_id["item"])
                .distinct()
                .count()
            )
            rate4 = round(items_count / items_total * 100, 2)
            sparsity = round(
                (1 - events_count / (users_count * items_count)) * 100, 4
            )
            print(
                f"Number of: events - {events_count} ({rate1}%), "
                f"users - {users_count} ({rate2}%), ",
                f"items - {items_count} ({rate4}%), ",
                f"sparsity - {sparsity}%\n",
                f"users (test only) - {users_count} ({rate3}%)\n",
            )

    def preprocess(self, log: bool = False) -> None:
        """
        Preprocesses the music data by performing the following steps:
        1. Preprocesses the catalog names.
        2. Preprocesses the tracks.
        3. Preprocesses the interactions.

        Args:
            log (bool):
                Whether to log the results. Defaults to False.
        """

        self.preprocess_catalog_names()
        self.preprocess_tracks()
        self.preprocess_interactions()
        if log:
            self.log()
        logger.info("Finished preprocessing")


if __name__ == "__main__":

    PreprocessingComponent(print_info=True).preprocess(log=True)

    logger.info(f"PreprocessingComponent completed successfully")
