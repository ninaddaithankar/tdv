import torch
import pytorch_lightning as pl

from copy import deepcopy

from eval.probes.module import ProbeLightningModule


class ProbeEvalCallback(pl.Callback):
    def __init__(self, hparams, datamodule, probe_type="linear"):
        super().__init__()
        self.hparams = hparams
        self.datamodule = datamodule
        self.num_classes = hparams.num_classes
        self.devices = hparams.gpus
        self.probe_type = probe_type
        self.strategy = "auto"
        self.max_epochs = hparams.probe_eval_max_epochs
        self.log_prefix = probe_type + "-probe"
        self.eval_once_before_training_start = hparams.eval_once_before_training_start


    def on_fit_start(self, trainer, pl_module):
        """
        Called at the start of training. We'll run probe here if configured to do so.
        """
        if self.eval_once_before_training_start:
            print("Running probe evaluation once before training starts...")

            self.on_validation_end(trainer, pl_module)

            # -- set eval_once_before_training_start to False to avoid running it in the middle of training
            self.eval_once_before_training_start = False


    def on_validation_end(self, trainer, pl_module):
        pl_module.eval()

        # deepcopy the encoder and freeze
        encoder = pl_module.model.teacher_frame_encoder if self.hparams.use_ema_for_frame_encoder else pl_module.model.frame_encoder

        frozen_encoder = deepcopy(encoder)
        frozen_encoder.eval()

        # build probe module
        probe_module = ProbeLightningModule(
            encoder=frozen_encoder,
            encoder_name=self.hparams.backbone_type,
            input_dim=self.hparams.vit_backbone_dim,
            num_classes=self.num_classes,
            probe_type = self.probe_type,
        )

        # init the trainer
        probe_trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            check_val_every_n_epoch=self.max_epochs,
            devices=self.devices,
            accelerator="gpu",
            strategy=self.strategy,
            logger=False,  # reuse main logger
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        print(f"ONLINE EVAL: Executing {self.probe_type}-probe on eval dataset at epoch {trainer.current_epoch} #############################")

        # fit probe
        probe_trainer.fit(probe_module, self.datamodule)

        # get validation accuracy
        val_top1 = probe_trainer.callback_metrics.get("val_top1", None)
        val_top5 = probe_trainer.callback_metrics.get("val_top5", None)
        
        if trainer.is_global_zero and val_top5 and val_top1:
            print(f"ONLINE EVAL: Finished {self.probe_type}-probe. Observed below results: ")
            print(f"{self.log_prefix} Validation Top 1 Acc: {val_top1.item() * 100}")
            print(f"{self.log_prefix} Validation Top 5 Acc: {val_top5.item() * 100}")

            trainer.logger.log_metrics({f"{self.log_prefix}/val_top1": val_top1.item() * 100})
            trainer.logger.log_metrics({f"{self.log_prefix}/val_top5": val_top5.item() * 100})

        pl_module.train()
