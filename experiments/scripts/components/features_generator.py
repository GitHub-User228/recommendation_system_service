import warnings

warnings.filterwarnings("ignore")

import argparse
import pandas as pd
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime, timedelta

from scripts import logger
from scripts.components.base import BaseComponent
from scripts.settings import get_features_generator_component_config
from scripts.utils import get_spark_session, spark_parquet_to_single_parquet


class FeaturesGeneratorComponent(BaseComponent):

    def __init__(self) -> None:
        """
        Initializes the FeaturesGeneratorComponent class.
        """
        self.config = get_features_generator_component_config()
        self.spark = get_spark_session()
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

    def _col1_per_col2(
        self, df: DataFrame, col1: str, col2: str, col_name: str, lag: int = 0
    ) -> DataFrame:
        """
        Calculates the number of times a value in column 1 occurs
        per value in column 2. Optionally, a lag can be specified, so
        that the calculation are performed since last `lag` days.

        Args:
            df (DataFrame):
                The Spark DataFrame to calculate the values from.
            col1 (str):
                The name of the column containing the values to count.
            col2 (str):
                The name of the column containing the values to count
                per value in column 1.
            lag (int, optional):
                The number of days to lag the calculation by.
                Defaults to 0.

        Returns:
            DataFrame:
                The Spark DataFrame with the calculated values.
        """
        if lag == 0:
            return (
                df.select(col1, col2)
                .groupBy(col2)
                .agg(F.count(col1).alias(col_name))
            )
        elif lag > 0:
            ref_date = (
                datetime.strptime(self.config.reference_date, "%Y-%m-%d")
                - timedelta(days=lag)
            ).strftime("%Y-%m-%d")
            return (
                df.select(
                    col1,
                    col2,
                    self.config.date_col,
                )
                .filter(F.col(self.config.date_col) >= F.lit(ref_date))
                .groupBy(col2)
                .agg(F.count(col1).alias(f"{col_name}_last_{lag}"))
            )
        else:
            raise ValueError(
                "Invalid lag value. It must be a non-negative integer."
            )

    def _days_since_first_last_interaction(
        self, df: DataFrame, col: str, col_name1: str, col_name2: str
    ) -> DataFrame:
        """
        Calculates the number of days since the first and last
        interaction for user or item.
        """
        return (
            df.select(
                col,
                self.config.date_col,
            )
            .groupBy(col)
            .agg(
                F.datediff(
                    F.to_date(F.lit(self.config.reference_date)),
                    F.min(self.config.date_col),
                ).alias(col_name1),
                F.datediff(
                    F.to_date(F.lit(self.config.reference_date)),
                    F.max(self.config.date_col),
                ).alias(col_name2),
            )
        )

    def _generate_default_features(
        self, df: DataFrame, for_user: bool = False
    ) -> DataFrame:
        """
        Generates the default features.
        """
        col1 = "user" if for_user else "item"
        col2 = "item" if for_user else "user"

        # col1 per col2
        df2 = self._col1_per_col2(
            df=df,
            col1=self.config.fields_id[col2],
            col2=self.config.fields_id[col1],
            col_name=f"{col2}s_per_{col1}",
        )

        # col1 per col2 with lags
        for lag in [7, 30]:
            df2 = df2.join(
                self._col1_per_col2(
                    df=df,
                    col1=self.config.fields_id[col2],
                    col2=self.config.fields_id[col1],
                    lag=lag,
                    col_name=f"{col2}s_per_{col1}",
                ),
                on=self.config.fields_id[col1],
            )

        # days since the last and the first interaction
        df2 = df2.join(
            self._days_since_first_last_interaction(
                df=df,
                col=self.config.fields_id[col1],
                col_name1=f"days_since_{col1}_first_interaction",
                col_name2=f"days_since_{col1}_last_interaction",
            ),
            on=self.config.fields_id[col1],
        )

        return df2

    def generate_user_features(self) -> None:
        """
        Generates and saves the user features.
        """

        # Read the events train data
        df = self.read_parquet(
            filename=self.config.events_filenames["train"], spark=True
        )

        # Generating default features
        user_features = self._generate_default_features(df=df, for_user=True)

        # Saving the features
        self.save_parquet(
            df=user_features.coalesce(1),
            filename=self.config.user_features_filename,
        )
        spark_parquet_to_single_parquet(
            path=self.config.destination_path,
            filename=self.config.user_features_filename,
        )
        logger.info(
            f"Generated user features saved to {self.config.destination_path}."
        )

    def generate_item_features(self) -> None:
        """
        Generates and saves the item features.
        """

        # Read the events train data
        df = self.read_parquet(
            filename=self.config.events_filenames["train"], spark=True
        )

        # Generating default features
        item_features = self._generate_default_features(df=df, for_user=False)

        # Saving the features
        self.save_parquet(
            df=item_features.coalesce(1),
            filename=self.config.item_features_filename,
        )
        spark_parquet_to_single_parquet(
            path=self.config.destination_path,
            filename=self.config.item_features_filename,
        )
        logger.info(
            f"Generated item features saved to {self.config.destination_path}."
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stage", type=int, required=True)
    args = parser.parse_args()

    if args.stage == 1:
        FeaturesGeneratorComponent().generate_user_features()
    elif args.stage == 2:
        FeaturesGeneratorComponent().generate_item_features()
    elif args.stage == 3:
        FeaturesGeneratorComponent().log()
    else:
        raise ValueError("Invalid --stage. Must be 1, 2, 3")

    logger.info(
        f"FeaturesGeneratorComponent stage {args.stage} completed successfully"
    )


if __name__ == "__main__":
    main()
