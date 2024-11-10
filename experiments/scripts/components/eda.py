import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from scripts import logger
from scripts.components.base import BaseComponent
from scripts.plotters import custom_hist_multiplot
from scripts.settings import get_eda_component_config
from scripts.utils import get_spark_session, get_experiment_id

sns.set_style("dark")
sns.set_theme(style="darkgrid", palette="deep")


class EDAComponent(BaseComponent):
    """
    Performs exploratory data analysis (EDA) on the training data, including:
    - user analysis
    - item analysis (tracks)
    - item category analysis (genres, albums, artists)

    Attributes:
        config (EDAComponentConfig):
            Class containing the configuration parameters.
        spark (SparkSession):
            The Spark session.
    """

    def __init__(self, show_graphs: bool = True) -> None:
        """
        Initializes the EDAComponent class with the necessary configuration and Spark session.

        Args:
            show_graphs (bool):
                Whether to display graphs during the analysis.
        """
        self.config = get_eda_component_config()
        self.spark = get_spark_session()
        self.show_graphs = show_graphs
        self._path_to_script = Path(__file__)

    def read_parquet(
        self, filename: str, spark: bool = True, verbose: bool = True
    ) -> DataFrame | pd.DataFrame:
        """
        Reads a Parquet file from the configured source path and
        returns a Spark DataFrame.

        Args:
            filename (str):
                The name of the Parquet file to read.
            spark (bool):
                Whether to use Spark for reading the file.
            verbose (bool):
                Whether to log the progress of reading the file.

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
        if verbose:
            logger.info(f"Read {filename} from {self.config.source_path}.")
        return df

    def default_analysis(self, events: DataFrame, user: bool = True) -> None:
        """
        Performs a default analysis on a given DataFrame of events, including:
        - Number of items/users per user/item
        - Cumulative number of items/users over time
        - Number of items/users per month

        Args:
            events (DataFrame):
                The DataFrame of events to analyze.
            user (bool):
                Whether to analyze users or items.
                Defaults to True.
        """

        feat1 = "user" if user else "item"
        feat2 = "item" if user else "user"

        ### Number of tracks per user (users per track)
        print(f"----- {feat2.upper()}S PER {feat1.upper()} -----\n\n")
        col = f"{feat2}s_per_{feat1}"
        df = (
            events.groupBy(self.config.fields_id[feat1])
            .agg({self.config.fields_id[feat2]: "count"})
            .withColumnRenamed(f"count({self.config.fields_id[feat2]})", col)
            .orderBy(col, ascending=False)
            .toPandas()
            .set_index(self.config.fields_id[feat1])
        )
        if feat1 == "item":
            df = df.merge(
                self.read_parquet(
                    self.config.name_df_filenames["item"],
                    spark=False,
                    verbose=False,
                ),
                left_index=True,
                right_on=self.config.fields_id["item"],
                how="left",
            ).drop(columns=self.config.fields_id["item"])
        print(
            f"Most {'active' if feat1 == 'user' else 'popular'} {feat1}s",
        )
        print(
            df.head(10).to_markdown(),
            end="\n\n",
        )
        df.head(10).to_csv(
            Path(self.config.data_path, f"{feat1}s_per_{feat2}_top.csv")
        )
        print(
            f"Least {'active' if feat1 == 'user' else 'popular'} {feat1}s",
        )
        print(
            df.tail(10).to_markdown(),
            end="\n\n",
        )
        print(f"Number of unique {feat1}s: {df.shape[0]}\n\n")
        print(f"Quantiles")
        q = (
            df[col]
            .quantile(self.config.quantiles)
            .reset_index()
            .rename(columns={"index": "quantiles"})
        )
        print(q.to_markdown(), end="\n\n")
        q.to_csv(
            Path(self.config.data_path, f"{feat1}s_per_{feat2}_quantiles.csv"),
            index=False,
        )
        custom_hist_multiplot(
            data=df,
            columns=[col],
            stat="count",
            title=f"Number of {feat2}s per {feat1}",
            width_factor=0.5,
            height_factor=1,
            title_fontsize=16,
            kde=False,
            yscale="log",
            savepath=Path(
                self.config.assets_path, f"{feat2}s_per_{feat1}_hist.png"
            ),
            show=self.show_graphs,
        )

        ### Cumulative number of users (items) over time
        col = f"number_of_{feat1}"
        min_date = events.select(F.min(self.config.date_col)).first()[0]
        max_date = events.select(F.max(self.config.date_col)).first()[0]
        df = (
            (
                self.spark.createDataFrame(
                    [(min_date, max_date)], ["start_date", "end_date"]
                )
                .select(
                    F.explode(
                        F.sequence(F.col("start_date"), F.col("end_date"))
                    ).alias("date")
                )
                .alias("dates")
                .join(
                    events.groupBy(self.config.fields_id[feat1])
                    .agg(F.min(self.config.date_col).alias("first_date"))
                    .alias("feats"),
                    F.col("feats.first_date") <= F.col("dates.date"),
                    how="left",
                )
            )
            .groupBy("date")
            .agg(F.count(self.config.fields_id[feat1]).alias(col))
            .orderBy("date")
            .toPandas()
        )
        fig = plt.figure(figsize=(8, 4))
        sns.lineplot(data=df, x="date", y=col, lw=5)
        plt.title(f"Cumulative number of {feat1}s over time", fontsize=16)
        if self.show_graphs:
            plt.show()
        fig.savefig(
            Path(self.config.assets_path, f"cumulative_{feat1}_count.png"),
            bbox_inches="tight",
        )

        ### Number of users (items) per month
        df = (
            events.withColumn(
                "date", F.date_format(F.col(self.config.date_col), "yyyy-MM")
            )
            .groupBy("date")
            .agg(F.countDistinct(self.config.fields_id[feat1]).alias(col))
            .orderBy("date")
            .toPandas()
        )
        df["date"] = pd.to_datetime(df["date"])

        fig = plt.figure(figsize=(8, 4))
        sns.lineplot(data=df, x="date", y=col, lw=5)
        plt.title(f"Number of {feat1}s per month", fontsize=16)
        if self.show_graphs:
            plt.show()
        fig.savefig(
            Path(self.config.assets_path, f"{feat1}_count_per_month.png"),
            bbox_inches="tight",
        )

    def item_analysis(self) -> None:
        """
        Performs an item-level analysis on the training events data.
        """
        events = self.read_parquet(
            filename=self.config.events_filenames["train"], spark=True
        )

        self.default_analysis(events, user=False)
        logger.info(f"Finished item analysis")

    def user_analysis(self) -> None:
        """
        Performs a user-level analysis on the training events data.
        """
        events = self.read_parquet(
            filename=self.config.events_filenames["train"], spark=True
        )
        self.default_analysis(events, user=True)
        logger.info(f"Finished user analysis")

    def category_analysis(
        self,
        category_name: Literal["genre", "album", "artist"],
        yscale: Literal["linear", "log"] = "linear",
    ) -> None:
        """
        Performs item-related item_category analysis.

        Args:
            category_name (Literal['genre', 'album', 'artist']):
                The name of the category to analyze.
            yscale (Literal['linear', 'log']):
                The scale for the y-axis. Defaults to "linear".
        """

        df = self.read_parquet(
            filename=self.config.item_features_filenames[category_name],
            spark=False,
        )

        print(f"----- {category_name.upper()}S PER ITEM -----", end="\n\n")
        cat_count = f"{category_name}_count"
        df2 = (
            df.groupby(self.config.fields_id["item"])
            .agg({self.config.fields_id[category_name]: "count"})
            .rename(columns={self.config.fields_id[category_name]: cat_count})
            .reset_index()
            .merge(
                self.read_parquet(
                    self.config.name_df_filenames["item"],
                    spark=False,
                    verbose=False,
                ),
                on=self.config.fields_id["item"],
                how="left",
            )
            .sort_values(
                by=[cat_count, self.config.fields_name["item"]],
                ascending=[False, True],
            )
            .set_index(self.config.fields_id["item"])
        )
        df2.head(10).to_csv(
            Path(self.config.data_path, f"{category_name}s_per_item_top.csv")
        )
        print(df2.head(10).to_markdown(), end="\n\n")
        print("Quantiles")
        q = (
            df2[cat_count]
            .quantile(self.config.quantiles)
            .reset_index()
            .rename(columns={"index": "quantiles"})
        )
        filename = f"quantiles_{category_name}s_per_item.csv"
        q.to_csv(Path(self.config.data_path, filename))
        print(q.to_markdown(), end="\n\n")
        custom_hist_multiplot(
            data=df2,
            columns=[cat_count],
            stat="count",
            title=f"Number of {category_name}s per item",
            width_factor=(1 if (df2[cat_count].nunique() < 20) else 0.5),
            height_factor=(0.7 if (df2[cat_count].nunique() < 20) else 1),
            title_fontsize=12,
            features_kind=(
                "cat" if (df2[cat_count].nunique() < 20) else "num"
            ),
            cat_orient="h",
            kde=False,
            yscale="linear" if (df2[cat_count].nunique() < 20) else yscale,
            savepath=Path(
                self.config.assets_path,
                f"{category_name}s_per_item_hist.png",
            ),
            show=self.show_graphs,
        )

        print(f"----- ITEMS PER {category_name.upper()} -----", end="\n\n")
        df2 = (
            df.groupby(self.config.fields_id[category_name])
            .agg({self.config.fields_id["item"]: "count"})
            .rename(columns={self.config.fields_id["item"]: "item_count"})
            .reset_index()
            .merge(
                self.read_parquet(
                    self.config.name_df_filenames[category_name],
                    spark=False,
                    verbose=False,
                ),
                on=self.config.fields_id[category_name],
                how="left",
            )
            .sort_values(
                by=["item_count", self.config.fields_name[category_name]],
                ascending=[False, True],
            )
            .set_index(self.config.fields_id[category_name])
        )
        print(f"Number of unique {category_name}s: {df2.shape[0]}", end="\n\n")
        print(f"Most broad {category_name}s")
        print(df2.head(10).to_markdown(), end="\n\n")
        df2.head(10).to_csv(
            Path(self.config.data_path, f"items_per_{category_name}_top.csv")
        )
        print(f"Least broad {category_name}s")
        print(df2.tail(10).to_markdown(), end="\n\n")
        print("Quantiles")
        q = (
            df2["item_count"]
            .quantile(self.config.quantiles)
            .reset_index()
            .rename(columns={"index": "quantiles"})
        )
        filename = f"quantiles_items_per_{category_name}.csv"
        q.to_csv(Path(self.config.data_path, filename))
        print(q.to_markdown(), end="\n\n")
        custom_hist_multiplot(
            data=df2,
            columns=["item_count"],
            stat="count",
            title=f"Number of items per {category_name}",
            width_factor=0.5,
            height_factor=1,
            title_fontsize=12,
            yscale=yscale,
            kde=False,
            savepath=Path(
                self.config.assets_path,
                f"items_per_{category_name}_hist.png",
            ),
            show=self.show_graphs,
        )
        logger.info(f"Finished {category_name} analysis")

    def analyze(self, log: bool = False) -> None:
        """
        Performs a full analysis on the training data.
        Args:
            log (bool):
                Whether to log the results. Defaults to False.
        """
        self.item_analysis()
        self.user_analysis()
        for category_name in ["genre", "album", "artist"]:
            self.category_analysis(category_name)
        if log:
            self.log()
        logger.info(f"Finished EDA")


if __name__ == "__main__":

    EDAComponent(show_graphs=False).analyze(log=True)

    logger.info(f"EDAComponent completed successfully")
