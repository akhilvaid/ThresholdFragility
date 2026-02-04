# General purpose config
import collections

# Module level definitions
DataSplit = collections.namedtuple(
    "DataSplit",
    ["train", "test", "test_prospective", "external", "external_prospective"],
)
Experiment = collections.namedtuple(
    "Experiment", ["cutoff", "external_site", "delta", "model", "pretrained"]
)


class Config:
    # Bootstrap iterations
    n_bootstrap = 500

    # General switches
    DEBUG = False
    RANDOM_STATE = 42

    # Hyperparameters - Classifier
    # Image size and learning rates are set within the dataset.
    ce_loss = True
    ft_epochs = 20

    batch_size = 40
    n_workers = 20

    # Model iterable
    models = {
        "resnet50": {"weights": "ResNet50_Weights"},
    }


class ResultFiles:
    # Threshold drift
    summary = "FinalResults/ThresholdDriftSummary.parquet"
