import gc
import torch

import pytorch_lightning as pl
import torchvision.transforms as T

from data.cv.imagenet_dataloader import ImageNetDataset
from eval.knn.knn_imp.distributed_knn import eval_knn_with_model


class KNNEvalCallback(pl.Callback):
    def __init__(self, args):
        """
        Expected args:
            data_dir: Path to the dataset.
            output_dir: Where to save final KNN results.
            nb_knn: Which k values to use in KNN.
            temperature: Softmax temperature for the KNN weighting.
            batch_size: Batch size for feature extraction.
            num_workers: Number of workers for DataLoader.
            gather_on_cpu: If True, gather features on CPU (slower, but less GPU mem)
        """
        super().__init__()
        self.data_dir = args.knn_eval_data_dir
        self.nb_knn = args.knn_k_values
        self.pool = args.knn_pooling
        self.temperature = args.knn_temperature
        self.batch_size = 128
        self.num_workers_per_gpu = 20
        self.n_per_class = [-1]  # -1 means all samples
        self.gather_on_cpu = False
        self.train_dataset_samples_per_class = args.eval_train_num_samples_per_class
        self.val_dataset_samples_per_class = args.eval_val_num_samples_per_class
        self.eval_once_before_training_start = args.eval_once_before_training_start
        self.run_knn_on_student = args.run_eval_on_student
        self.model_name = args.backbone_type
        self.hparams = args

        # -- define transforms & datasets
        # self.train_transform = T.Compose([
        #     T.Resize((args.image_dim[0], args.image_dim[1])),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        # self.val_transform = T.Compose([
        #     T.Resize((args.image_dim[0], args.image_dim[1])),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        # copied from dinov1 knn eval
        self.transform = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(args.image_dim[0]),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # -- create the datasets
        self.train_dataset = ImageNetDataset(args, split = 'train', transform=self.transform, dataset_dir=self.data_dir, n_samples_per_class=self.train_dataset_samples_per_class)
        self.val_dataset = ImageNetDataset(args, split = 'val', transform=self.transform, dataset_dir=self.data_dir, n_samples_per_class=self.val_dataset_samples_per_class)
        

    def on_fit_start(self, trainer, pl_module):
        """
        Called at the start of training. We'll run k-NN here if configured to do so.
        """
        if self.eval_once_before_training_start:
            print("Running k-NN evaluation once before training starts...")

            self._knn_wrapper(trainer, pl_module)

            # -- set eval_once_before_training_start to False to avoid running it in the middle of training
            self.eval_once_before_training_start = False
        

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called after every validation epoch. We'll run k-NN here.
        """
        print("Running k-NN evaluation after validation epoch...")
        self._knn_wrapper(trainer, pl_module)


    def _knn_wrapper(self, trainer, pl_module):
        # -- set eval mode
        pl_module.eval()  

        encoder = pl_module.model.teacher_frame_encoder if self.hparams.use_ema_for_frame_encoder else pl_module.model.frame_encoder
        encoder = encoder.to(pl_module.device)
        self.run_evaluation_with_encoder(trainer, encoder)

        if self.run_knn_on_student:
            encoder = pl_module.model.frame_encoder
            encoder = encoder.to(pl_module.device)
            print("Running k-NN evaluation on STUDENT model...")
            self.run_evaluation_with_encoder(trainer, encoder, is_student=True)

        # -- back to train
        pl_module.train()


    def run_evaluation_with_encoder(self, trainer, encoder, is_student=False):
        """
        Run k-NN evaluation.
        """            
        # -- currently configured for TDV models 
        with torch.no_grad():
            results_dict = eval_knn_with_model(
                trainer=trainer,
                model=encoder,
                model_name=self.model_name,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                nb_knn=self.nb_knn,
                n_per_class_list=self.n_per_class,
                temperature=self.temperature,
                batch_size=self.batch_size,
                num_workers=self.num_workers_per_gpu * trainer.world_size,
                gather_on_cpu=self.gather_on_cpu,
                pool=self.pool,
            )

            if trainer.is_global_zero:
                prefix = "student_" if is_student else ""

                # Log results to the trainer's logger
                if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                    for k, v in results_dict.items():
                        trainer.logger.experiment.log({prefix + k: v}, step=trainer.global_step)

        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
        
