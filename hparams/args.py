from argparse import ArgumentParser
from model.model_utils import model_sizes

def get_args():
	parser = ArgumentParser()

	# SUPER IMPORTANT HPARAMS ############################################################################

	parser.add_argument("--run_name", help="run name, should include model name in it for important runs (see slurm scripts)", default="test")
	parser.add_argument("--modality", help="is the model being trained for NLP, CV, ARC (arc is considered a modality), or SYN (synthetic data). default is CV, and most of the code has been designed around CV.", type=str, default="CV")
	parser.add_argument("--model_name", help="model_type/name", default='tdv')

	# CV SPECIFIC PARAMS #################################################################################

	parser.add_argument("--vit_backbone_size", help="small, base, large or giant", type=str, default="base")
	parser.add_argument("--backbone_type", help="backbone type for CV encoder", choices=["dinov1", "dinov2", "vae", "mae"], type=str, default="dinov2")
	parser.add_argument("--time_between_frames", help="time (seconds) between frames of video", type=float, default=1.0)
	parser.add_argument("--sampling_rate", help="how many frames to draw a single video frame from. e.g. sampling rate=4 will have three frames in between each sample. overrides time_between_frames", type=float, default=0.0)
	parser.add_argument("--use_raw_framerate", help="whether to use dataset default framerate with a sampling rate of 1. this overrides the time_between_frames and sampling_rate parameters.", action="store_true", default=False)
	parser.add_argument("--vae_normalization", help="whether to use the commonly used vae normalization of [-1, 1].", action="store_true", default=False)

	# MODEL AND ARCHITECTURE #############################################################################

	parser.add_argument("--context_length", help="total number of frames supported by the model", type=int, default=0)
	parser.add_argument("--num_transformer_blocks", help="number of transformer blocks", type=int, default=12)
	parser.add_argument("--change_encoder_type", help="architecture type for the change encoder (currently supports resnets and vits from torchvision.models)", type=str, default="")
	parser.add_argument("--embedding_dim", help="embedding dimension for transformers, if are using a model size this is automatically set", type=int, default=0)

	# HARDWARE ###########################################################################################

	parser.add_argument("--gpus", help="number of gpus or gpus list, -1 uses all GPUs. use -1 for multinode, if want to specify which GPUs to use specify as comma seperated str with brackets e.g. [0, 1]", default="-1")
	parser.add_argument("--distributed_strategy", help="distributed strategy - ddp_spawn, ddp, fsdp_native, or None", default='ddp')

	# TRAINING ###########################################################################################

	parser.add_argument("-ep", "--epochs", help="total number of epochs", type=int, default=-1)
	parser.add_argument("--peak_learning_rate", help="peak learning rate after warm up", type=float, default=0.02)
	parser.add_argument("--batch_size_per_device", help="batch size (PER DEVICE!!!, not effective). to get effective_batch_size is num_gpus * batch_size_per_device * accumulate_grad_batches)", type=int, default=2)
	parser.add_argument("--accumulate_grad_batches", help="number of batches to accumulate before stepping optimizer, 1 is regular", type=int, default=1)
	parser.add_argument("--gradient_clip_val", help="maximum value for gradient clipping", type=float, default=1.0)
	parser.add_argument("--is_random_seed", help="is_random_seed", action="store_true", default=False)
	parser.add_argument("--deterministic", help="ensures results are determnistic, may run a bit slower, sets flag on pl trainer and sets workers for dataloader", action="store_true", default=False)
	parser.add_argument("--execution_mode", type=str, choices=["pretrain", "finetune"], default="pretrain")
	parser.add_argument("--finetuning_model_ckpt", help="model ckpt when finetuning", type=str, default=None)

	# METRICS ############################################################################################
	# -- reported metrics in wandb will be averages over the time span they are being computed,
	# -- so the final one in an epoch will be the most accurate

	parser.add_argument("--metrics_list", help = "list of metrics to use", nargs='+', default = [])
	parser.add_argument("--num_classes", help="the number of classes, must set if using metrics, see torchmetrics docs for more info", type=int, default=-1)
	parser.add_argument("--metrics_task", help="the type of task being done ['binary', 'multiclass' or 'multilabel'], see torchmetrics docs for more info, must set if using metrics_list", type=str, default="")

	# OPTIMIZER AND LR SCHEDULER #########################################################################

	parser.add_argument("--weight_decay", help="weight decay to use", type=float, default=0.01)
	parser.add_argument("--beta1", help="exponential decay rate for first moment estimate", type=float, default=0.9)
	parser.add_argument("--beta2", help="exponential decay rate for second movement estimate", type=float, default=0.999)
	parser.add_argument("--lr_scaling_rule", help="the LR will be scaled according to the rule LR = base_lr * effective_batch_size / 256. is useful for prototyping and is popular in vision SSL. effective_batch_size is based off bs * num_gpus * accumulate_grad_batches", action="store_true", default=False)
	parser.add_argument("--min_lr_scale", help="the most the lr will be scaled down during cosine decay", type=int, default=10)
	parser.add_argument("--max_steps", help="max number of steps for training", type=int, default=1000000)
	parser.add_argument("--max_scheduling_steps", help="max number of steps used for lr/other hparam scheduling. in general should be the same as max_steps but can be different if dont want to do a full run but want the lr and other things to be scheduled the same. similar to V jepa paper. if is not set will default to max_steps value", type=int, default=-1)
	parser.add_argument("--warm_up_steps", help="number of steps to increase the LR linearly over before hitting specified peak LR", type=int, default=10000)
	parser.add_argument("--warm_up_base_lr_divider", help="lr divider for when doing linear learning rate warm up. if is set to -1 then does warm up from 0", type=float, default=-1)
	parser.add_argument("--optimizer", help="used to turn on different optimizers. current options include adamw (default), lars, stableadamw", type=str, default="adamw")
	parser.add_argument("--lars_trust_coeff", help="exponential decay rate for second movement estimate", type=float, default=0.001)
	parser.add_argument("--lars_exclude_bias_bn_wd", help="excludes bias and batch norm from Lars adaptation and weight decay", action="store_true", default=False)

	# DATASET AND DATALOADER #############################################################################

	parser.add_argument("--num_workers", help="num_workers per GPU. idea to do per GPU gotten from https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/", type=int, default=4)
	parser.add_argument("--dataset_name", help="dataset name", default="ucf101")
	parser.add_argument("--dataset_dir", help="dataset base directory", default="")
	parser.add_argument("--aggr_datasets_list", help="list of dataset identifiers to use for the aggregated dataset", default="")
	parser.add_argument("--aggr_dataset_dirs", help="list of dataset directories for the aggregated dataset", default="")
	parser.add_argument("--image_dim", help="List of image dimensions", nargs='+', type=int, default = [224, 224])
	parser.add_argument("--patch_size", help="Patch size for ViT", type=int, default = 14)
	parser.add_argument("--dino_v1_checkpoint_key", help="key to extract weights from DINOv1 checkpoint", type=str, default="teacher", choices=["teacher", "student"])
	parser.add_argument("--no_randomness_dataloader", help="makes dataloader have no randomness by only sampling from start", action="store_true", default=False)
	parser.add_argument("--preprocess_data", help="center crops images - helps to avoid black bars on sides which cause issues with unstable gradient", action="store_true", default=False)
	parser.add_argument("--crop_all_samples", help="if preprocess data is enabled this will crop all samples", action="store_true", default=False)
	parser.add_argument("--custom_image_normalization", help="whether to normalize data according to each dataset's std and mean", action="store_true", default=False)
	parser.add_argument("--ffprobe_path", help="path to ffprobe binary", default="")
	parser.add_argument("--test_split_pct", help="if training, validation, and test set are being split - which percent is test. by default this is not used", type=float, default=0)
	parser.add_argument("--train_data_pct", help="fraction of training data to use (e.g. 0.01, 0.1, 1.0). subset is fixed at setup time so the model sees the same samples every epoch", type=float, default=1.0)
	parser.add_argument("--limit_train_batches", help="percent of training dataset to use, or if > 1 is num batches", type=float, default=1.0)
	parser.add_argument("--limit_val_batches", help="percent of validation dataset to use, or if > 1 is num batches", type=float, default=1.0)
	parser.add_argument("--limit_test_batches", help="percent of testing dataset to use, or if > 1 is num batches", type=float, default=1.0)
	parser.add_argument("--val_sanity", help="number of sanity validation steps", type=int, default=0)
	parser.add_argument("--val_check_interval", help="interval to do validation per training epoch, 1 means once per epoch. useful if epochs are too large to wait to do validation", type=float, default=1.0)
	parser.add_argument("--check_val_every_n_epoch", help="interval to do validation every n epochs", type=int, default=1)
	parser.add_argument("--preencode_dataset", help="whether to encode the dataset and save it for faster loading later. NOTE: this will take a while and requires a lot of disk space. also, if you change any hparams related to the dataloading you might need to reencode the dataset", action="store_true", default=False)
	parser.add_argument("--use_preencoded_dataset", help="whether to use the preencoded dataset. NOTE: if you change any hparams related to the dataloading you might need to reencode the dataset", action="store_true", default=False)
	parser.add_argument("--use_preprocessed_finevideo", help="use preprocessed finevideo dataset", action="store_true", default=False)
	parser.add_argument("--use_preprocessed_ego4d", help="use preprocessed ego4d dataset", action="store_true", default=False)
	parser.add_argument("--hdf5_file_path", help="path to preencoded dataset hdf5 file", type=str, default="")

	# TESTING ############################################################################################

	parser.add_argument("--only_test", help="just testing no training, used passed in model checkpoint", action="store_true", default=False)
	parser.add_argument("--only_test_model_ckpt", help="model ckpt when only testing", type=str, default=None)

	# WANDB ##############################################################################################

	parser.add_argument("--no_wandb", help="no wandb", action="store_true", default=False)
	parser.add_argument("--wandb_project", help="wandb project name", default='testing')
	parser.add_argument("--wandb_tags", help="wandb tags to add", nargs='+', default=None)
	parser.add_argument("--wandb_offline", help="set wandb to offline mode", action="store_true", default=False)
	parser.add_argument("--wandb_watch", help="turns on watch mode for wandb - expensive so only use for debugging", action="store_true", default=False)
	parser.add_argument("--wandb_watch_log_freq", help="number of steps to log for wandb watch. is higher since is a bit expensive", type = int, default=1000)

	# LOGGING ############################################################################################

	parser.add_argument("--console_log_filename", help="filename of log", default='console.log')
	parser.add_argument("--log_model_archi", help="log model architecture", action="store_true", default=False)
	parser.add_argument("--log_gradients", help="logs gradients at every step to wandb to debug them", action="store_true", default=False)
	parser.add_argument("--log_every_n_steps", help="turns on logger freq via pl, not advised to use this", type=int, default=50)

	# CHECKPOINTING ######################################################################################

	parser.add_argument("--resume_training_ckpt", help="checkpoint to resume training from, use absolute",type=str, default="")
	parser.add_argument("--checkpoint_monitor_string", help="string to use to monitor for saving checkpoint. ", type=str, default="valid/loss")
	parser.add_argument("--checkpoint_monitor_mode", help="monitoring mode for checkpoint_monitor_string, either ['min', 'max']. if is loss do min, if is a metric like accuracy do max", type=str, default="min")

	# PRECISION ##########################################################################################

	parser.add_argument("--set_matmul_precision", help="set math mult precision - \"medium\", \"high\", or \"highest\" ", default=None)
	parser.add_argument("--float_precision", help="float precision, pl recommends 16-mixed/bf16-mixed, also has by default 32-true", type=str, default="32-true")

	# SPEED ##############################################################################################

	parser.add_argument("--compile_model", help="compiles the model using torch.compile", action="store_true", default=False)

	# SLURM ##############################################################################################

	parser.add_argument("--is_slurm_run", help="please set to true if doing slurm run, as of now just stops capturing console logs", action="store_true", default=False)

	# DEBUGGING ##########################################################################################

	parser.add_argument("--debug_mode", help="turns debug mode on where dataset returned is very small, no_wandb is on and detect anomaly is on", action="store_true", default=False)
	parser.add_argument("--fast_dev_run", help="turns fast_dev_run for trainer on, makes it just do one training epoch and one val epoch", action="store_true", default=False)
	parser.add_argument("--overfit_batches", help="if nonzero will overfit to specified num/percent of batches", type=float, default=0.0)
	parser.add_argument("--profiler", choices=["simple", "advanced"], type=str, default="")
	parser.add_argument("--no_shuffle", help="stops shuffling - helpful for debugging", action="store_true", default=False)
	parser.add_argument("--detect_anomaly", help="turns on anomaly detection mode", action="store_true", default=False)
	parser.add_argument("--find_unused_parameters", help="turns on pl find unused params mode - DO NOT KEEP ON for actual training, helpful if want to debug. this uses DDPStrategy and ignores distributed_strategy", action="store_true", default=False)
	parser.add_argument("--debug_unused_parameters", help="makes it so it tracks which params are used to find the params that are causing the unused params issue", action="store_true", default=False)
	parser.add_argument("--manual_gc_collect_every_n_steps", help="manually call gc collect every n steps, can be done to prevent CPU RAM memory 'leak'", type = int, default=-1)

	# VJEPA SPECIFIC #####################################################################################

	parser.add_argument("--num_segments", help="num_segments as specified in the VJEPA repo config.yaml for the dataset you are trying to fit", type=int, default=2)
	parser.add_argument("--dataset_train_csv_path", help="path to the training dataset csv file that contains video file path and the classification label pairs", type=str, default="")
	parser.add_argument("--dataset_val_csv_path", help="path to the validation dataset csv file that contains video file path and the classification label pairs", type=str, default="")

	# TDV SPECIFIC #######################################################################################

	# -- architecture
	parser.add_argument("--use_ema_for_frame_encoder", help="setting this would enable using a teacher and student for frame encoder where teacher is updated only using ema while the student is actively trained", action="store_true", default=False)
	parser.add_argument("--ema_momentum", help="momentum for teacher ema updates", type=float, default=0.9999)
	parser.add_argument("--remove_motion_encoder", help="setting this would remove the motion encoder and just use the frame encoder with ema", action="store_true", default=False)

	# -- masking
	parser.add_argument("--frame_encoder_mask_ratio", help="mask ratio for masking input to frame encoder", type=float, default=0.0)
	parser.add_argument("--use_ibot_masking", help="setting this would use ibot style masking for the frame encoder", action="store_true", default=False)
	parser.add_argument("--ibot_mask_sample_probability", help="probability of how many samples to mask in a batch while using ibot masking", type=float, default=0.5)
	parser.add_argument("--ibot_mask_ratio_min_max", help="min and max mask ratio for ibot style masking for the frame encoder", nargs='+', type=float, default=[0.4, 0.8])

	parser.add_argument("--use_rope", help="use rope for positional encoding in frame encoder and motion encoder", action="store_true", default=False)
	parser.add_argument("--use_ape", help="use absolute positional encoding (APE) in frame encoder and motion encoder; if both --use_ape and --use_rope are set, both encodings are applied", action="store_true", default=False)

	parser.add_argument("--use_different_init_teacher", help="uses different seed for initializing the teacher", action="store_true", default=False)
	parser.add_argument("--load_teacher_from_checkpoint", help="load teacher frame encoder from this checkpoint", type=str, default="")
	parser.add_argument("--use_fixed_dino_teacher", help="init teacher to be a fixed pretrained dinov2 model without ema updates", action="store_true", default=False)

	# -- frame encoder
	parser.add_argument("--unfreeze_frame_encoder", help="setting this would prevent freezing the frame encoder while training", action="store_true", default=False)
	parser.add_argument("--use_only_cls_token", help="set this to use only the class tokens outputted from the difference encoder to calculate the loss", action="store_true", default=False)
	parser.add_argument("--load_without_weights", help="choose to load the image encoder without weights", action="store_true", default=False)

	# -- augmentation
	parser.add_argument("--use_dino_augmentation", help="apply separate DINO-style augmentations to student and teacher input frames", action="store_true", default=False)
	parser.add_argument("--rgb_diff_from_unaugmented", help="when use_dino_augmentation is on, compute rgb_diff from unaugmented frames instead of student frames", action="store_true", default=False)

	# -- motion encoder
	parser.add_argument("--rgb_diff_threshold", help="if the rgb diff is lower than this value, loss will not be calculated", type=float, default=-1)
	parser.add_argument("--use_full_frame", help="pass in the full frame to the difference encoder instead of the rgb differrence", action="store_true", default=False)
	parser.add_argument("--reinit_difference_encoder_every_epoch", help="setting this would reinitialize the weights for the difference/motion encoder after every epoch", action="store_true", default=False)
	parser.add_argument("--freeze_difference_encoder", help="setting this would freeze the difference/motion encoder while training", action="store_true", default=False)
	parser.add_argument("--difference_encoder_lr_multiplier", help="scale up or down the learning rate for the difference encoder", type=float, default=1.0)
	parser.add_argument("--use_spatial_conditioning", help="setting this will use spatial conditioning in motion encoder instead of cross attention", action="store_true", default=False)
	parser.add_argument("--use_gating", help="setting this will use gated spatial conditioning", action="store_true", default=False)
	parser.add_argument("--ignore_prefix_tokens_in_condition", help="setting this will use ignore the cls tokens while using spatial conditioning", action="store_true", default=False)

	# -- loss
	parser.add_argument("--use_mse_loss", help="setting this would enable mse loss on the tdv model", action="store_true", default=False)
	parser.add_argument("--mse_loss_weight", help="coefficient for the mse loss", type=float, default=1.0)
	parser.add_argument("--rollout_n_frames", help="the number of timesteps to make next frame latent prediction with mse loss", type=int, default=1)
	parser.add_argument("--recon_predicted_temp", help="temperature for softening student output with reconstruction loss", type=float, default=1.0)
	parser.add_argument("--recon_target_temp", help="temperature for sharpening teacher output with reconstruction loss", type=float, default=1.0)
	parser.add_argument("--recon_loss_type", help="type of reconstruction loss to use, mse or smoothl1", type=str, default="mse")
	parser.add_argument("--recon_use_sharpening", help="use sharpening on teacher and softening on student in reconstruction loss", action="store_true", default=False)
	parser.add_argument("--recon_use_centering", help="use centering on teacher in reconstruction loss", action="store_true", default=False)
	parser.add_argument("--use_motion_loss", help="setting this would enable motion loss on the tdv model", action="store_true", default=False)
	parser.add_argument("--motion_loss_weight", help="coefficient for the motion loss", type=float, default=1.0)
	parser.add_argument("--min_embed_diff_per_pixel_diff", help="the motion loss constant", type=float, default=0.44607)
	parser.add_argument("--use_dino_head", help="use dino head after the frame encoder", action="store_true", default=False)
	parser.add_argument("--dino_head_prototype_dim", help="dimension of the dino head output", type=int, default=768)
	parser.add_argument("--use_sharpening", help="use sharpening on teacher and softening on student in dino loss", action="store_true", default=False)
	parser.add_argument("--use_centering", help="use centering on teacher in dino loss", action="store_true", default=False)
	parser.add_argument("--use_seperate_ibot_head", help="use different head for ibot prototypes", action="store_true", default=False)
	parser.add_argument("--use_seperate_ibot_center", help="use seperate center while doing centering for ibot prototypes", action="store_true", default=False)
	parser.add_argument("--use_dino_loss", help="use dino loss on class tokens from the frame encoder + dino head", action="store_true", default=False)
	parser.add_argument("--dino_loss_weight", help="coefficient for the dino loss", type=float, default=1.0)
	parser.add_argument("--dino_student_temp", help="temperature for softening student output with dino loss", type=float, default=1)
	parser.add_argument("--dino_teacher_temp", help="temperature for sharpening teacher output with dino loss", type=float, default=1)
	parser.add_argument("--use_ibot_loss", help="use ibot loss on patch tokens from the frame encoder + dino head", action="store_true", default=False)
	parser.add_argument("--ibot_loss_weight", help="coefficient for the ibot loss", type=float, default=1.0)
	parser.add_argument("--ibot_student_temp", help="temperature for softening student output with ibot loss", type=float, default=1)
	parser.add_argument("--ibot_teacher_temp", help="temperature for sharpening teacher output with ibot loss", type=float, default=1)
	parser.add_argument("--dino_center_update_momentum", help="coefficient for momentum updates of the running center when using dino loss", type=float, default=0.9)
	parser.add_argument("--ibot_center_update_momentum", help="coefficient for momentum updates of the running center when using ibot loss", type=float, default=0.9)

	# -- logging and debugging
	parser.add_argument("--log_baseline_losses", help="enable to log baseline log metrics for comparison", action="store_true", default=False)
	parser.add_argument("--log_var_covar", help="log variance and covariance of the calculated embeddings", action="store_true", default=False)

	# ONLINE EVAL (KNN, PROBES, MOT) #####################################################################

	parser.add_argument("--run_online_evaluations", help="set up knn/probe/object tracking online evaluation during training", type=str, default="probe")

	# -- KNN
	parser.add_argument("--knn_eval_data_dir", help="directory where the knn evaluation data is stored", type=str, default="")
	parser.add_argument("--knn_k_values", help="list of k values to use for knn evaluation", nargs='+', type=int, default=(10, 20))
	parser.add_argument("--knn_temperature", help="temperature to use for knn evaluation", type=float, default=0.07)
	parser.add_argument("--knn_pooling", help="pooling method for features before knn (cls/avg)", type=str, default='cls')
	parser.add_argument("--knn_batch_size", help="batch size to use for knn evaluation", type=int, default=32)
	parser.add_argument("--eval_train_num_samples_per_class", help="number of samples per class to use for the training set the knn model", type=int, default=50)
	parser.add_argument("--eval_val_num_samples_per_class", help="number of samples per class to use for the validation set the knn model", type=int, default=-1)

	# -- PROBE
	parser.add_argument("--probe_eval_dataset", help="the dataset to be used for online evaluation (imagenet1k/ssv2)", type=str, default="imagenet1k")
	parser.add_argument("--probe_eval_data_dir", help="directory where the probe evaluation data is stored", type=str, default="")
	parser.add_argument("--probe_eval_type", help="the type of probe to be evaluated", type=str, default="linear")
	parser.add_argument("--probe_eval_bs", help="batch size for online probe evaluations", type=int, default=128)
	parser.add_argument("--probe_eval_max_epochs", help="epochs for training the linear/attentive probe classifiers", type=int, default=1)

	# -- MOT
	parser.add_argument("--mot_eval_data_dir", help="directory where the mot evaluation data is stored", type=str, default="")
	parser.add_argument("--mot_eval_max_epochs", help="epochs for training the linear/attentive probe classifiers", type=int, default=1)

	# -- attention visualization
	parser.add_argument("--max_images", type=int, default=-1, help="optional max number of images to run through (for speed/debug)")
	parser.add_argument("--max_batches", type=int, default=-1, help="max number of batches to iterate through")
	parser.add_argument("--log_every_n_batches", type=int, default=1, help="log attention maps every N batches")
	parser.add_argument("--max_images_per_batch", type=int, default=-1, help="how many images per logged batch to visualize")
	parser.add_argument("--save_dir", type=str, default="", help="directory to save attention visualizations as PNG files")
	parser.add_argument("--viz_seed", type=int, default=42, help="fixed seed for reproducible random image selection across checkpoints")
	parser.add_argument("--viz_mode", type=str, default="attention", choices=["attention", "pca", "both"], help="which visualization to run")

	# -- others
	parser.add_argument("--eval_once_before_training_start", help="run evaluation once before starting the training", action="store_true", default=False)
	parser.add_argument("--run_eval_on_student", help="run evaluation on student encoder as well if using ema", action="store_true", default=False)

	########################################################################################################

	return parser.parse_args()
