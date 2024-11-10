import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List
from pydantic import BaseModel, Field, model_validator

from scripts.env import env_vars
from scripts.utils import read_yaml

# Reading config data
config = read_yaml(Path(env_vars.config_dir, "components.yaml"))


def inherit_field(
    base_model: BaseModel, field_name: str, **overrides
) -> Field:
    """
    Inherits a field from a base model and applies any overrides.

    Args:
        base_model (BaseModel):
            The base model to inherit the field from.
        field_name (str):
            The name of the field to inherit.
        **overrides:
            Any overrides to apply to the inherited field.

    Returns:
        Field:
            The inherited field with any overrides applied.
    """
    base_field = base_model.model_fields[field_name]
    return Field(
        default=overrides.get("default", base_field.default),
        description=base_field.description,
        **{k: v for k, v in overrides.items() if k != "default"},
    )


class BaseConfig(BaseModel):
    """
    Defines the `BaseConfig` class, which represents
    common configuration parameters for all components.
    """

    experiment_name: str = Field(
        default=config["experiment_name"],
        description="Name of the experiment.",
    )

    uri: str = Field(
        default=(
            f"http://{env_vars.tracking_server_host}:"
            f"{env_vars.tracking_server_port}"
        ),
        description="URI of the MLflow registry and tracking servers.",
    )
    source_path: Path = Field(
        description="Path to the source data.",
    )
    destination_path: Path = Field(
        description="Path to the destination data.",
    )

    date_col: str = "started_at"

    fields_id: Dict[str, str] = Field(
        default={
            "item": "track_id",
            "album": "album_id",
            "artist": "artist_id",
            "genre": "genre_id",
            "user": "user_id",
        },
        description="Maps fields to corresponding names of ID column",
    )

    fields_name: Dict[str, str] = Field(
        default={
            "item": "track",
            "album": "album",
            "artist": "artist",
            "genre": "genre",
            "user": "user",
        },
        description="Maps fields to a corresponding name",
    )

    name_df_filenames: Dict[str, str] = {
        "item": "track_names.parquet",
        "album": "album_names.parquet",
        "artist": "artist_names.parquet",
        "genre": "genre_names.parquet",
    }

    item_features_filenames: Dict[str, str] = {
        "album": "track_albums.parquet",
        "artist": "track_artists.parquet",
        "genre": "track_genres.parquet",
    }

    events_filenames: Dict[str, str] = {
        "train": "events_train.parquet",
        "target": "events_target.parquet",
        "test": "events_test.parquet",
    }

    encoders_filenames: Dict[str, str] = {
        "item": "encoder_item.pkl",
        "user": "encoder_user.pkl",
        "album": "encoder_album.pkl",
        "artist": "encoder_artist.pkl",
        "genre": "encoder_genre.pkl",
    }

    @model_validator(mode="after")
    def path_handler(self) -> "BaseConfig":
        """
        Validates path parameters

        Returns:
            BaseConfig:
                The updated configuration.
        """

        # Checking if source_path exists
        if not self.source_path.exists():
            raise ValueError(
                f"The source path {self.source_path} does not exist."
            )

        # Checking if source_path is not empty
        if not any(self.source_path.iterdir()):
            raise ValueError(f"The source path {self.source_path} is empty.")

        # Creating destination directory if not exists
        if not self.destination_path.exists():
            self.destination_path.mkdir(parents=True, exist_ok=True)

        return self


class PreprocessingComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `PreprocessingComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    source_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="source_path",
        default=env_vars.artifacts_dir,
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "preprocessing"),
    )
    catalog_names_filename: str = Field(
        description="Filename of the catalog names data.",
    )
    tracks_filename: str = Field(
        description="Filename of the tracks data.",
    )
    interactions_filename: str = Field(
        description="Filename of the interactions data.",
    )
    groups: Dict[str, str] = {
        "item": "tracks",
        "album": "albums",
        "artist": "artists",
        "genre": "genres",
    }
    train_test_split_date: str = Field(
        description="Date for splitting the data into train and test sets.",
    )
    target_test_split_date: str = Field(
        description="Date for splitting the test data into target and test sets.",
    )


class MatrixBuilderComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `MatrixBuilderComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "matrices"),
    )
    user_items_matrix_filename: str = Field(
        default="user_items_matrix.npz",
        description="Filename of the user-items matrix.",
    )
    item_feature_matrix_filenames: Dict[str, str] = Field(
        default={
            "album": "item_albums_matrix.npz",
            "artist": "item_artists_matrix.npz",
            "genre": "item_genres_matrix.npz",
        },
        description="Filenames of files with item features matrices.",
    )
    batch_size: int = Field(
        ge=1,
        description="Batch size for matrix construction.",
    )


class FeaturesGeneratorComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `FeaturesGeneratorComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "features"),
    )
    user_features_filename: str = Field(
        default="user_features.parquet",
        description="Filename of the user features data.",
    )
    item_features_filename: str = Field(
        default="item_features.parquet",
        description="Filename of the item features data.",
    )
    reference_date: str = Field(
        description="Reference date for feature generation.",
    )


class EDAComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `EDAComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )

    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "eda"),
    )

    assets_path: Path = Field(
        default=Path("assets"),
        description="Directory for storing EDA assets.",
    )

    data_path: Path = Field(
        default=Path("data"),
        description="Directory for storing EDA files with data.",
    )

    quantiles: List[float] = Field(
        default=list(np.arange(0.1, 1, 0.1)) + [0.95, 0.99, 0.999],
        description="Quantiles for computing the distribution of the data.",
    )

    @model_validator(mode="after")
    def path_handler2(self) -> "EDAComponentConfig":
        """
        Handles path parameters

        Returns:
            EDAComponentConfig:
                The updated configuration.
        """

        # Creating assets directory if not exists
        self.assets_path = Path(self.destination_path, self.assets_path)
        if not self.assets_path.exists():
            self.assets_path.mkdir(parents=True, exist_ok=True)

        # Creating data directory if not exists
        self.data_path = Path(self.destination_path, self.data_path)
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        return self


class ALSModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `ALSModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )

    user_items_matrix_filename: str = Field(
        description="Filename of the user-items matrix.",
    )
    source_path2: Path = Field(
        description="Secondary source path",
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "modelling/als"),
    )
    model_filename: str = Field(
        default="als.pkl",
        description="Filename of the ALS model.",
    )
    similar_items_filename: str = Field(
        default="similar_items_als.parquet",
        description="Filename of the similar items data.",
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="score_als",
        description="Name of the score column",
    )

    # ALS model fields
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )
    min_users_per_item: int = Field(
        description="Minimum number of users per item.",
    )
    max_similar_items: int = Field(
        ge=1,
        description="Maximum number of similar items to return.",
    )
    factors: int = Field(
        ge=1,
        description="Number of factors.",
    )
    iterations: int = Field(
        ge=1,
        description="Number of iterations",
    )
    regularization: float = Field(
        ge=1e-8,
        le=1.0,
        description="The regularization factor",
    )
    alpha: float | None = Field(
        ge=1.0,
        le=50,
        description="The weight to give to positive examples",
    )
    calculate_training_loss: bool = Field(
        description="Whether to calculate the training loss.",
    )
    random_state: int = Field(
        description="Random state",
    )


class BPRModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `BPRModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )

    user_items_matrix_filename: str = Field(
        description="Filename of the user-items matrix.",
    )
    source_path2: Path = Field(
        description="Secondary source path",
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "modelling/bpr"),
    )
    model_filename: str = Field(
        default="bpr.pkl",
        description="Filename of the BPR model.",
    )
    similar_items_filename: str = Field(
        default="similar_items_bpr.parquet",
        description="Filename of the similar items data.",
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="score_bpr",
        description="Name of the score column",
    )

    # BPR model fields
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )
    min_users_per_item: int = Field(
        description="Minimum number of users per item.",
    )
    max_similar_items: int = Field(
        ge=1,
        description="Maximum number of similar items to return.",
    )
    factors: int = Field(
        ge=1,
        description="Number of factors",
    )
    iterations: int = Field(
        ge=1,
        description="Number of iterations",
    )
    learning_rate: float = Field(
        ge=1e-8,
        le=1.0,
        description="Learning rate",
    )
    regularization: float = Field(
        ge=1e-8,
        le=1.0,
        description="Regularization parameter",
    )
    verify_negative_samples: bool = Field(
        description="Whether to verify negative samples.",
    )
    random_state: int = Field(
        description="Random state for the ALS model.",
    )


class Item2ItemModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `Item2ItemModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    user_items_matrix_filename: str = Field(
        description="Filename of the user-items matrix.",
    )
    item_feature_matrix_filenames: Dict[str, str] = Field(
        description="Filenames of files with item features matrices.",
    )
    source_path2: Path = Field(
        description="Secondary data source path",
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "modelling/item2item"),
    )
    model_filename: str = Field(
        default="item2item.pkl",
        description="Filename of the Item2ItemModel model.",
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="score_item2item",
        description="Name of the score column",
    )

    # Item2Item model fields
    batch_size: int = Field(
        ge=1,
        description="Batch size for the Item2ItemModel model.",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )
    min_users_per_item: int = Field(
        ge=1,
        description="Minimum number of users per item.",
    )
    n_neighbors: int = Field(
        ge=1,
        description="Number of nearest neighbors",
    )
    n_components: int = Field(
        ge=1,
        description="Number of components of SVD model.",
    )
    similarity_criteria: str = Field(
        description="Similarity criteria for nearest neighbors",
    )


class TopItemsModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `TopItemsModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    user_items_matrix_filename: str = Field(
        description="Filename of the user-items matrix.",
    )
    source_path2: Path = Field(
        description="Secondary data source path",
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "modelling/top_items"),
    )
    top_items_filename: str = Field(
        default="top_items.parquet",
        description="Filename of the top items data.",
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="item_popularity",
        description="Name of the score column",
    )
    top_n_items: int = Field(
        ge=1,
        description="Number of top items to retrieve",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations to make",
    )


class EnsembleModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `EnsembleModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )

    base_models: List[str] = Field(description="List of base models aliases")
    train_data_path: Dict[str, str] = Field(
        default={},
        description=(
            "Maps base models alias to the corresponding training data path"
        ),
    )
    test_data_path: Dict[str, str] = Field(
        default={},
        description=(
            "Maps base models alias to the corresponding test data path"
        ),
    )
    user_features_path: Path | None = Field(
        default=None,
        description="Path to the user features data",
    )
    item_features_path: Path | None = Field(
        default=None,
        description="Path to the item features data",
    )
    user_items_matrix_filename: str = Field(
        description="Filename of the user-items matrix.",
    )
    source_path2: Path = Field(
        description="Secondary data source path",
    )
    include_top_items: bool = Field(
        default=False,
        description="Whether to include top items in the ensemble",
    )
    top_items_path: str | None = Field(
        default=None,
        description="Path to the top items data",
    )

    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "modelling/ensemble"),
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "train_df": "train_df.parquet",
            "test_df": "test_df.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    feature_importances_filename: str = Field(
        default="feature_importances.yaml",
        description="Filename of the feature importances file.",
    )
    target_col: str = Field(
        default="target",
        description="Name of the target column",
    )
    score_col: str = Field(
        default="score_cb_ensemble",
        description="Name of the score column",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )

    # Sampling fields
    negative_samples_per_user: int = Field(
        default=5,
        ge=1,
        description="Number of negative samples per user",
    )
    sampling_seed: int = Field(
        default=42,
        description="Seed for sampling negative samples",
    )

    # Ensemble model fields
    model_class_name: str = Field(
        description="Name of the ensemble model class",
    )
    model_params: Dict[str, Any] = Field(
        description="Parameters for the ensemble model",
    )
    model_filename: str = Field(
        default="ensemble.pkl",
        description="Filename of the Ensemble model.",
    )

    @model_validator(mode="after")
    def get_train_test_data_path(self) -> "EnsembleModelComponentConfig":
        """
        Retrieves the path to the training data

        Returns:
            EnsembleModelComponentConfig:
                The updated configuration.
        """

        config_aliases = {
            "als": get_als_model_component_config,
            "bpr": get_bpr_model_component_config,
            "item2item": get_item2item_model_component_config,
        }

        for alias in self.base_models:
            model_config = config_aliases[alias]()
            self.train_data_path[alias] = Path(
                model_config.destination_path,
                model_config.recommendations_filenames["target"],
            )
            self.test_data_path[alias] = Path(
                model_config.destination_path,
                model_config.recommendations_filenames["test"],
            )

        return self

    @model_validator(mode="after")
    def get_top_items_data(self) -> "EnsembleModelComponentConfig":
        """
        Retrieves the path to the top items data if necessary.

        Returns:
            EnsembleModelComponentConfig:
                The updated configuration.
        """

        if self.include_top_items:
            config1 = get_top_items_model_component_config()
            self.top_items_path = Path(
                config1.destination_path, config1.top_items_filename
            )

        return self


@lru_cache
def get_preprocessing_component_config() -> PreprocessingComponentConfig:
    """
    Returns the preprocessing component configuration.
    """
    return PreprocessingComponentConfig(
        **config["PreprocessingComponentConfig"]
    )


@lru_cache
def get_matrix_builder_component_config() -> MatrixBuilderComponentConfig:
    """
    Returns the matrix builder component configuration.
    """
    config1 = get_preprocessing_component_config()
    return MatrixBuilderComponentConfig(
        source_path=config1.destination_path,
        **config["MatrixBuilderComponentConfig"],
    )


@lru_cache
def get_features_generator_component_config() -> (
    FeaturesGeneratorComponentConfig
):
    """
    Returns the features generator component configuration.
    """
    config1 = get_preprocessing_component_config()
    return FeaturesGeneratorComponentConfig(
        source_path=config1.destination_path,
        **config["FeaturesGeneratorComponentConfig"],
    )


@lru_cache
def get_eda_component_config() -> EDAComponentConfig:
    """
    Returns the eda component configuration
    """
    config1 = get_preprocessing_component_config()
    return EDAComponentConfig(
        source_path=config1.destination_path, **config["EDAComponentConfig"]
    )


@lru_cache
def get_als_model_component_config() -> ALSModelComponentConfig:
    """
    Returns the collaborative model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return ALSModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        user_items_matrix_filename=config2.user_items_matrix_filename,
        **config["ALSModelComponentConfig"],
    )


@lru_cache
def get_bpr_model_component_config() -> BPRModelComponentConfig:
    """
    Returns the bpr model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return BPRModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        user_items_matrix_filename=config2.user_items_matrix_filename,
        **config["BPRModelComponentConfig"],
    )


@lru_cache
def get_item2item_model_component_config() -> Item2ItemModelComponentConfig:
    """
    Returns the lightfm model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return Item2ItemModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        user_items_matrix_filename=config2.user_items_matrix_filename,
        item_feature_matrix_filenames=config2.item_feature_matrix_filenames,
        **config["Item2ItemModelComponentConfig"],
    )


@lru_cache
def get_top_items_model_component_config() -> TopItemsModelComponentConfig:
    """
    Returns the top items model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return TopItemsModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        user_items_matrix_filename=config2.user_items_matrix_filename,
        **config["TopItemsModelComponentConfig"],
    )


@lru_cache
def get_ensemble_model_component_config() -> EnsembleModelComponentConfig:
    """
    Returns the ensemble model component configuration.
    """

    config1 = get_preprocessing_component_config()
    config2 = get_features_generator_component_config()
    config3 = get_matrix_builder_component_config()
    return EnsembleModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config3.destination_path,
        user_items_matrix_filename=config3.user_items_matrix_filename,
        user_features_path=Path(
            config2.destination_path, config2.user_features_filename
        ),
        item_features_path=Path(
            config2.destination_path, config2.item_features_filename
        ),
        **config["EnsembleModelComponentConfig"],
    )
