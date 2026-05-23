from copy import deepcopy

from pytorch_lightning import Callback, Trainer
from eval.tracking.deepsort.module import DeepSORTModule


class DeepSORTEvalCallback(Callback):
    def __init__(self, hparams, datamodule, eval_every_n_epochs=1):
        self.hparams = hparams
        self.datamodule = datamodule
        self.eval_every_n_epochs = eval_every_n_epochs
        self.devices = hparams.gpus
        self.strategy = "auto"
        self.max_epochs = hparams.mot_eval_max_epochs
        self.eval_once_before_training_start = hparams.eval_once_before_training_start

    def on_fit_start(self, trainer, pl_module):
        """
        Called at the start of training. We'll run eval if configured to do so.
        """
        if self.eval_once_before_training_start:
            print("Running MOT evaluation once before training starts...")

            self.on_validation_end(trainer, pl_module)

            # -- set eval_once_before_training_start to False to avoid running it in the middle of training
            self.eval_once_before_training_start = False

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return

        pl_module.eval()

        # deepcopy the encoder and freeze
        # encoder = pl_module.model.teacher_frame_encoder if self.hparams.use_ema_for_frame_encoder else pl_module.model.frame_encoder
        encoder = pl_module.model.frame_encoder

        frozen_encoder = deepcopy(encoder)
        frozen_encoder.eval()

        frozen_encoder = frozen_encoder.to(pl_module.device)

        # init MOT evaluator model
        mot_model = DeepSORTModule(
            encoder=frozen_encoder,
            encoder_name=self.hparams.backbone_type,
            log_metrics=trainer.logger.log_metrics,
            step=trainer.global_step,
            config=self.hparams
        )

        # eval trainer (reuses wandb logger)
        eval_trainer = Trainer(
            max_epochs=self.max_epochs,
            logger=trainer.logger,
            accelerator='gpu',
            devices=self.devices,            # using only one GPU for now as evaluating on single sequence
            strategy=self.strategy,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

        # run validation pass over MOT data
        eval_trainer.validate(mot_model, datamodule=self.datamodule)

        pl_module.train()
