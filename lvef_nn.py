# Neural network
import os
import time
import datetime

import tqdm
import torch
import torchvision
import pandas as pd
import polars as pl
import numpy as np

from PIL import Image
from sklearn import metrics

from config import Config, DataSplit

torch.manual_seed(Config.RANDOM_STATE)
torch.cuda.manual_seed(Config.RANDOM_STATE)
np.random.seed(Config.RANDOM_STATE)
torch.backends.cudnn.benchmark = True


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, df, return_image_path=False, pretraining=True):
        # Create one iterable that can be __getitemed__
        self.df = df
        self.return_image_path = return_image_path

        # Imagenet norms
        mean = [0.485, 0.456, 0.406] if pretraining else [0.5, 0.5, 0.5]
        std = [0.229, 0.224, 0.225] if pretraining else [0.5, 0.5, 0.5]

        # Transforms
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((448, 448)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        record = self.df.iloc[index]
        filepath = os.path.join(Config.ecg_image_location, record["ECGID"] + ".png")

        image = Image.open(filepath).convert("RGB")
        image_tensor = self.transforms(image)

        label = torch.tensor(record["LABEL"], dtype=torch.long)

        if self.return_image_path:
            return image_tensor, label, filepath

        return image_tensor, label


class Finetune:
    def __init__(
        self,
        dataset_splits: DataSplit,
        model_identifier,
        pretraining,
        output_qualifier,
    ):
        # List of available ECGs
        self.ecg_list = [
            filename.replace(".png", "")
            for filename in os.listdir(Config.ecg_image_location)
        ]

        # Parameters
        self.dataset_splits = dataset_splits
        self.model_identifier = model_identifier
        self.pretraining = pretraining
        self.output_qualifier = output_qualifier

        # Build dataloaders for all splits
        self.dataloaders = DataSplit(
            train=self.create_dataloader(dataset_splits.train, is_training=True),
            test=self.create_dataloader(dataset_splits.test, is_training=False),
            test_prospective=self.create_dataloader(
                dataset_splits.test_prospective, is_training=False
            ),
            external=self.create_dataloader(dataset_splits.external, is_training=False),
            external_prospective=self.create_dataloader(
                dataset_splits.external_prospective, is_training=False
            ),
        )

        self.n_classes = 2
        self.patience = 5
        self.best_auroc_internal = 0.0
        self.epochs_no_improve = 0
        self.stopped_early = False

    def create_dataloader(self, df: pl.DataFrame, is_training: bool = True):
        df = (
            df.select(["ECGID", "OUTCOME"])
            .rename({"OUTCOME": "LABEL"})
            .filter(pl.col("ECGID").is_in(self.ecg_list))
            .to_pandas()
        )
        if len(df) == 0:
            return None

        batch_size = Config.batch_size
        if is_training:
            training_samples = len(df)
            if training_samples < batch_size:
                batch_size = min(Config.batch_size, training_samples)

        dataset = ECGDataset(
            df, return_image_path=not is_training, pretraining=self.pretraining
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=Config.n_workers,
            drop_last=is_training,
            pin_memory=True,
        )

        return dataloader

    def calculate_auc_vals(self, df_pred: pd.DataFrame):
        auroc = metrics.roc_auc_score(y_true=df_pred["TRUE"], y_score=df_pred["PRED"])
        auprc = metrics.average_precision_score(
            y_true=df_pred["TRUE"], y_score=df_pred["PRED"]
        )
        return auroc, auprc

    def eval_model(self, dataloader, model, split_name: str):
        if dataloader is None or len(dataloader.dataset) == 0:
            # Nothing to evaluate on this split
            return 0, np.nan, np.nan, pd.DataFrame()

        n_samples = len(dataloader.dataset)

        all_preds = []
        all_labels = []
        all_files = []

        model.eval()
        for images, labels, paths in tqdm.tqdm(dataloader, desc=f"{split_name}"):
            all_files.extend(paths)

            with torch.autocast(device_type="cuda"):
                with torch.no_grad():
                    images = images.cuda()
                    labels = labels.cuda()

                    outputs = model(images)

                    probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                    all_preds.extend(probs)
                    all_labels.extend(labels.cpu().numpy())

        df_pred = pd.DataFrame(
            {"FILES": all_files, "TRUE": all_labels, "PRED": all_preds}
        )
        df_pred["TRUE"] = df_pred["TRUE"].astype(int)

        try:
            auroc, auprc = self.calculate_auc_vals(df_pred)
        except ValueError:
            auroc, auprc = np.nan, np.nan

        return n_samples, auroc, auprc, df_pred

    def gaping_maw(self, dataloaders: DataSplit, model):
        # Housekeeping
        result_dir = os.path.join("Results", self.output_qualifier)
        os.makedirs(result_dir, exist_ok=True)

        success_file = os.path.join(result_dir, "run_successful.txt")
        if os.path.exists(success_file):
            print("Run already completed. Skipping.")
            return

        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.GradScaler(device="cuda")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=Config.ft_epochs,
            steps_per_epoch=len(dataloaders.train),
        )

        all_results = []

        for epoch in range(Config.ft_epochs + 5):
            epoch_loss = 0.0

            model.train()
            for images, labels in tqdm.tqdm(dataloaders.train):
                with torch.autocast(device_type="cuda"):
                    images = images.cuda()
                    labels = labels.cuda()

                    # Same as optim.zero_grad()
                    for param in model.parameters():
                        param.grad = None

                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), labels)
                    epoch_loss += loss.item() * outputs.shape[0]

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if epoch < Config.ft_epochs:
                        scheduler.step()

            training_loss = epoch_loss / len(dataloaders.train.dataset)
            lr = scheduler.get_last_lr()[0]

            # Sample counts
            training_samples = len(dataloaders.train.dataset)

            # Evaluate on all evaluation splits
            # Internal test set
            testing_samples, auroc_testing, aupr_testing, df_pred_testing = (
                self.eval_model(dataloaders.test, model, "Internal")
            )

            # Prospective internal test set
            (
                testing_prospective_samples,
                auroc_testing_prospective,
                aupr_testing_prospective,
                df_pred_testing_prospective,
            ) = self.eval_model(  # noqa: E501
                dataloaders.test_prospective, model, "Internal Prospective"
            )

            # External test set
            external_samples, auroc_external, aupr_external, df_pred_external = (
                self.eval_model(dataloaders.external, model, "External")
            )

            # Prospective external test set
            (
                external_prospective_samples,
                auroc_external_prospective,
                aupr_external_prospective,
                df_pred_external_prospective,
            ) = self.eval_model(
                dataloaders.external_prospective, model, "External Prospective"
            )

            # Aggregate results for this epoch
            all_results.append(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "samples_training": training_samples,
                    "samples_testing": testing_samples,
                    "samples_testing_prospective": testing_prospective_samples,
                    "samples_external": external_samples,
                    "samples_external_prospective": external_prospective_samples,
                    "loss_training": training_loss,
                    "auroc_testing": auroc_testing,
                    "aupr_testing": aupr_testing,
                    "auroc_testing_prospective": auroc_testing_prospective,
                    "aupr_testing_prospective": aupr_testing_prospective,
                    "auroc_external": auroc_external,
                    "aupr_external": aupr_external,
                    "auroc_external_prospective": auroc_external_prospective,
                    "aupr_external_prospective": aupr_external_prospective,
                }
            )

            # Print epoch summary
            print(
                f"Epoch {epoch}: Test AUROC={auroc_testing:.4f}, AUPRC={aupr_testing:.4f}"
            )

            df_results = pd.DataFrame(all_results)
            df_results.to_pickle(os.path.join(result_dir, "results.pickle"))

            # Save prediction probabilities for this epoch for all eval splits
            df_pred_testing.to_pickle(
                os.path.join(result_dir, f"testing_{epoch}.pickle")
            )
            df_pred_testing_prospective.to_pickle(
                os.path.join(result_dir, f"testing_prospective_{epoch}.pickle")
            )
            df_pred_external.to_pickle(
                os.path.join(result_dir, f"external_{epoch}.pickle")
            )
            df_pred_external_prospective.to_pickle(
                os.path.join(result_dir, f"external_prospective_{epoch}.pickle")
            )

            # Early stopping based on internal retrospective test AUROC
            improved = auroc_testing > self.best_auroc_internal
            if improved:
                self.best_auroc_internal = auroc_testing
                self.epochs_no_improve = 0

                model_filename = os.path.join(result_dir, "best_model.pth")
                torch.save(model.state_dict(), model_filename)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    self.stopped_early = True
                    with open(success_file, "a") as f:
                        f.write(
                            f"Early stopping: no AUROC improvement for {self.patience} "
                            f"epochs (best={self.best_auroc_internal:.4f}) @ "
                            f"{datetime.datetime.now().isoformat(timespec='seconds')}.\n"
                        )
                    break

        if not self.stopped_early:
            with open(success_file, "w") as f:
                f.write(f"Run completed successfully @ {datetime.datetime.now()}.\n")

    def create_model(self):
        if self.pretraining:
            weight_identifier = Config.models[self.model_identifier]["weights"]
            weights = eval(f"torchvision.models.{weight_identifier}.IMAGENET1K_V1")
            model = eval(f"torchvision.models.{self.model_identifier}(weights=weights)")
        else:
            model = eval(f"torchvision.models.{self.model_identifier}()")

        num_classes = self.n_classes

        match self.model_identifier:
            case "resnet50" | "resnet152":
                model.fc = torch.nn.Linear(2048, num_classes)
            case "densenet121":
                model.classifier = torch.nn.Linear(1024, num_classes)
            case "densenet201":
                model.classifier = torch.nn.Linear(1920, num_classes)
            case "convnext_small" | "convnext_base" | "convnext_large":
                if self.model_identifier == "convnext_small":
                    model.classifier[2] = torch.nn.Linear(768, num_classes)
                elif self.model_identifier == "convnext_base":
                    model.classifier[2] = torch.nn.Linear(1024, num_classes)
                elif self.model_identifier == "convnext_large":
                    model.classifier[2] = torch.nn.Linear(1536, num_classes)
            case _:
                raise ValueError(
                    f"No specific modifications for model {self.model_identifier}"
                )

        return model

    def hammer_time(self):
        try:
            model = self.create_model()
            self.gaping_maw(self.dataloaders, model)
        except IndexError as e:
            with open("Errors.log", "a") as outfile:
                outfile.write(f"{time.ctime()} {self.model_identifier} {e}\n")
