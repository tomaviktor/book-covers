import numpy as np
import torch
import lightning as L
from .config import Config
from .metric import Accuracy
from transformers import AutoModelForImageClassification
from .data import BookCovers
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from .resnet import ResNetForImageClassification


class Trainer(L.LightningModule):
    def __init__(self, model, config: Config):
        super().__init__()
        self.model = model
        self.config = config
        self.validation_step_outputs = {}
        self.save_hyperparameters(config.dict())
        self.metric = Accuracy()

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        self.log("loss/train", loss.item(), batch_size=batch["pixel_values"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        logits = output.logits

        self.validation_step_outputs["labels"].append(batch["labels"].detach().cpu().numpy())
        self.validation_step_outputs["logits"].append(logits.detach().cpu().numpy())
        self.validation_step_outputs["loss"].append(loss.item())
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = {"loss": [], "logits": [], "labels": []}

    def on_validation_epoch_end(self):
        prediction = self.validation_step_outputs["logits"]
        if isinstance(prediction, list):
            prediction = np.vstack(prediction)
        labels = self.validation_step_outputs["labels"]
        if isinstance(labels, list):
            labels = np.vstack(labels)
        scores = self.metric.compute_metrics((prediction, labels))
        loss = np.mean(self.validation_step_outputs["loss"])

        self.log("loss/validation", loss.item())
        for key, value in scores.items():
            try:
                self.log(f"metrics/{key}", value.item())
            except AttributeError:
                self.log(f"metrics/{key}", value)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.initial_learning_rate, betas=self.config.betas)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            pct_start=self.config.perc_warmup_steps,
            anneal_strategy="cos",
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def init_and_fit_trainer(
        token: str,
        image_size: int,
        grayscale: bool,
        augmentation: bool,
        debug: bool,
        batch_size: int,
        learning_rate: float,
        seed: int = 0,
        cache_dir: str = None,
        device: str = 'gpu',
        num_labels: int = 207572,
        model_name: str = 'microsoft/resnet-50',
        softmax_function: str = 'softmax'
):
    L.seed_everything(seed)
    config = Config(
        token=token,
        seed=seed,
        learning_rate=learning_rate,
        model_name=model_name,
        batch_size=batch_size,
        image_size=image_size,
        grayscale=grayscale,
        augmentation=augmentation,
        debug=debug,
        cache_dir=cache_dir,
        softmax_function=softmax_function
    )
    dataset = BookCovers(
        config.token,
        config.image_size,
        config.grayscale,
        config.augmentation,
        config.debug,
        config.cache_dir,
        config.batch_size
    )
    config.steps_per_epoch = len(dataset) // config.batch_size
    model = ResNetForImageClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.config.softmax_function = config.softmax_function
    trainer_module = Trainer(model, config)
    logger = WandbLogger('book-covers')
    trainer = L.Trainer(
        deterministic=True,
        accelerator=device,
        log_every_n_steps=10,
        logger=logger,
        min_epochs=config.num_epochs,
        max_epochs=config.num_epochs,
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=10.0,
        check_val_every_n_epoch=1,
        callbacks=[LearningRateMonitor(logging_interval="step")]
    )
    trainer.fit(
        trainer_module,
        datamodule=dataset
    )
