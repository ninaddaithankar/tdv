import os, sys, time, random, json
import pytorch_lightning as L

from datetime import datetime

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from eval.data_utils.data_module import ProbeDataModule
from eval.data_utils.mot_data_module import MOT17DataModule
from eval.knn.callback import KNNEvalCallback
from eval.probes.callback import ProbeEvalCallback
from eval.tracking.deepsort.callback import DeepSORTEvalCallback
from hparams.args import get_args
from model.model_utils import load_trained_pl_model, init_wandb_watch
from utils import text_logger
from base_model_trainer import *


# -- ensure only one wandb run is created across all GPUs
@rank_zero_only
def setup_wandb(args):
	import wandb
	if wandb.run is None:
		run = wandb.init(dir="logs/", name=f'{args.run_name}', project=f'{args.wandb_project}', mode = "offline" if args.wandb_offline else "online")
		wandb.define_metric("__init", hidden=True)
		return run
	return None


def main(args):

	# -- set seed for everything
	if(args.is_random_seed):
		seed_everything(random.randint(0,1000000), workers=True)
	else:
		seed_everything(786, workers=True)


	# -- set debug mode configs
	if args.debug_mode:
		args.no_wandb = True
		args.detect_anomaly = True
		args.limit_train_batches = 1

	os.makedirs("./logs", exist_ok=True)

	wandb_logger = None
	# -- init early so wandb captures stdout logs from the start
	if not args.no_wandb:
		# -- need both lines since setup_wandb is @rank_zero_only
		run = None
		run = setup_wandb(args)

		wandb_logger = WandbLogger(save_dir="logs/", name=f'{args.run_name}', project=f'{args.wandb_project}', offline = args.wandb_offline, experiment=run)
		if args.wandb_tags != None:
			wandb_logger.experiment.tags = args.wandb_tags
	else:
		console_log_file_path = os.path.join("./logs", args.console_log_filename)
		sys.stdout = text_logger.Tee(sys.stdout, console_log_file_path)
		sys.stderr = text_logger.Tee(sys.stderr, console_log_file_path)
		print("$$$$$$$$$ NOTE THAT NOT ALL STDOUT LOGS (i.e. pytorch lighting logs) ARE CAPTURED THROUGH CONSOLE LOGGER, THIS IS ONLY RECOMMENDED FOR DEBUGGING $$$$$$$$$$")

	if args.is_slurm_run:
		assert args.debug_mode == False and args.detect_anomaly == False and args.find_unused_parameters == False and args.debug_unused_parameters == False, "for slurm run cannot have certain params set to values since am assuming are not debugging, please check values here"

		print("Current Slurm job ID:", os.environ.get('SLURM_JOBID'))
		print("Current Slurm node list:", os.environ.get('SLURM_NODELIST'))
		print("SLURM_NTASKS:", os.environ.get("SLURM_NTASKS"))
		print("SLURM_GPUS_PER_NODE:", os.environ.get("SLURM_GPUS_PER_NODE"))
		print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


	# -- hparam assertion sanity checks
	if args.modality == "CV":
		if args.backbone_type in ["dinov2", "mae"]:
			assert args.embedding_dim == 0, "embedding dim defined implicitly by encoder dimensionality"
			# -- embed dims from dinov2 repo
			vit_backbone_embed_dim_map = {"xsmall": 192, "small": 384, "base": 768, "large": 1024, "huge": 1280, "giant": 1536}
			args.vit_backbone_dim = vit_backbone_embed_dim_map[args.vit_backbone_size]
			args.embedding_dim = args.vit_backbone_dim
		elif args.backbone_type == "vae":
			assert args.embedding_dim != 0, "must define embedding dim for vae"
		else:
			raise NotImplementedError(f"Unspported backbone type: {args.backbone_type}")
	else:
		raise ValueError(f"please add support for modality {args.modality}")


	# -- find and set number of GPUs
	# -- defaults to 1 if not on slurm; multi-node only supported via slurm
	args.num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', 1))
	print(f"SLURM_JOB_NUM_NODES: {args.num_nodes}")
	print("torch.cuda.device_count()", torch.cuda.device_count())
	if args.gpus == "-1":
		# -- TODO: verify this works for multi-node scaling
		num_gpus = args.num_nodes * torch.cuda.device_count()
	elif '[' in args.gpus:
		num_gpus = len(args.gpus.split(","))
		args.gpus = json.loads(args.gpus)
	else:
		num_gpus = int(args.gpus)


	# -- set total number of workers
	args.total_num_workers = args.num_workers * num_gpus
	print("num_nodes", args.num_nodes, "total num_workers across all GPUs", args.total_num_workers, "num workers per GPU", args.num_workers, "num_GPUs", num_gpus)


	# -- assert GPU availability
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")
	assert (device == torch.device('cuda') and num_gpus > 0), "using cpu instead of cuda. if you would like to proceed please remove this line and change code below to not use GPUs, otherwise check packages to ensure torch/others have cuda support"
	print(f'GPU Availability: {device}, gpus: {num_gpus}\n')


	# -- calculate effective batch size and scale the learning rate accordingly
	args.num_gpus = num_gpus
	effective_batch_size = args.num_gpus * args.batch_size_per_device * args.accumulate_grad_batches
	print(f"effective_batch_size: {effective_batch_size}", "batch_size_per_device", args.batch_size_per_device)
	if args.lr_scaling_rule:
		scaled_lr = args.peak_learning_rate * effective_batch_size / 256
		args.peak_learning_rate = scaled_lr
		print(f"Learning Rate rescaled to: {scaled_lr} based off lr_scaling_rule")
	if args.max_scheduling_steps == -1:
		args.max_scheduling_steps = args.max_steps


	# -- init the model trainer
	model_trainer = ModelTrainer(args)


	# -- if finetuning, load trained pl model
	if args.execution_mode == "finetune":
		# -- TODO: test this path; verify lr, lr schedule, dataset are all correct; may need to reset optimizer and lr scheduler
		assert args.finetuning_model_ckpt != None and args.resume_training_ckpt == "", "Must provide a checkpoint when finetuning and cannot provide a resume_training_ckpt."
		model_trainer.model = load_trained_pl_model(args.finetuning_model_ckpt, args)


	# -- timestamp configs
	timestamp = int(time.time())
	dt_object = datetime.fromtimestamp(timestamp)
	dt_string = dt_object.strftime("%Y-%m-%d_%H-%M-%S")


	# -- set wandb to watch the model_trainer
	if not args.no_wandb and args.wandb_watch:
		init_wandb_watch(wandb_logger, model_trainer, args.wandb_watch_log_freq)


	# -- log model architecture
	if args.log_model_archi:
		print(str(model_trainer.model))
		print(str(args))


	print(f'pytorch version: {torch.__version__}\n')


	# -- set matmul precision
	if args.set_matmul_precision is not None:
		torch.set_float32_matmul_precision(args.set_matmul_precision)


	# -- set up all callbacks
	callbacks = []


	# -- instantiate pl checkpoint callback
	checkpoint_filename = "epoch={epoch}-step={step}-" + args.checkpoint_monitor_string + "={"+args.checkpoint_monitor_string+":.4f}"
	checkpoint_callback = ModelCheckpoint(monitor=args.checkpoint_monitor_string, mode = args.checkpoint_monitor_mode, save_top_k=20, save_last = True, dirpath=f"./logs/checkpoints/{args.run_name}_{dt_string}_", filename=checkpoint_filename, verbose=True)
	callbacks.append(checkpoint_callback)


	# -- instantiate online eval callbacks
	evaluations = args.run_online_evaluations.split(",")
	for eval in evaluations:
		eval = eval.strip()

		if "knn" == eval:
			callback = KNNEvalCallback(args)
			print("KNN Online Eval Callback is set up")

		elif "probe" == eval:
			datamodule = ProbeDataModule(args)
			callback = ProbeEvalCallback(args, datamodule, probe_type=args.probe_eval_type)
			print(f"{args.probe_eval_type} - Probe Eval Callback is set up")

		elif "mot" == eval:
			datamodule = MOT17DataModule(args)
			callback = DeepSORTEvalCallback(args, datamodule)
			print(f"MOT (DeepSORT) Eval Callback is set up")

		callbacks.append(callback)


	# -- print non-trainable params
	for name, param in model_trainer.model.named_parameters():
		if not param.requires_grad:
			print(f"Non-trainable parameters: {name} with shape {param.shape}")


	# -- training
	if not args.only_test:
		print("$$$$$$$$$$  STARTED TRAINING  $$$$$$$$$$")

		trainer = set_trainer(args, wandb_logger, callbacks)

		resume_training_ckpt = None if args.resume_training_ckpt == "" else args.resume_training_ckpt
		trainer.fit(model_trainer, ckpt_path=resume_training_ckpt)

		clear_cache()

		print("$$$$$$$$$$  FINISHED TRAINING  $$$$$$$$$$")

	# -- only testing
	else:
		print("$$$$$$$$$$  ONLY TESTING MODEL ([NO]) TRAINING  $$$$$$$$$$")
		assert args.only_test_model_ckpt != None, "Must supply pretrained model when only testing"
		checkpoint = torch.load(args.only_test_model_ckpt, weights_only=False)
		pretrained_hparams = checkpoint['hyper_parameters']

		# -- merge in any args added after this checkpoint was trained, to support testing older checkpoints
		default_args = vars(args).copy()
		for key, value in default_args.items():
			if key not in pretrained_hparams and key != "execution_mode":
				pretrained_hparams[key] = value
				print(f"MISSING PARAMETER IN PRETRAINED CHECKPOINT: Using args set value for missing parameter in pretrained checkpoint '{key}': {value}")

		model = ModelTrainer(pretrained_hparams)
		model.load_state_dict(checkpoint['state_dict'])
		model.eval()
		# -- use current args so the most recently passed-in model is used
		model_trainer = ModelTrainer(args, trained_model=model.model)

		trainer = set_trainer(args, wandb_logger, callbacks, stage = "test")
		model_trainer.model.eval()
		trainer.test(model_trainer)


def set_trainer(args, wandb_logger, callbacks, stage = "train"):
	# -- pl detect_anomaly is unreliable so set it via autograd directly
	torch.autograd.set_detect_anomaly(args.detect_anomaly)

	if args.find_unused_parameters:
		# -- if having issues with strategy try 'ddp_spawn' instead of 'ddp'
		args.distributed_strategy = DDPStrategy(find_unused_parameters = True)

	args.overfit_batches = int(args.overfit_batches) if int(args.overfit_batches) == args.overfit_batches else args.overfit_batches

	# -- TODO: try devices="auto" and verify testing still works
	devices = [0] if stage == "test" else args.gpus
	print("devices: ", devices)

	profiler = None if args.profiler == "" else args.profiler
	gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None
	limit_val_batches = 0 if args.overfit_batches > 0 else args.limit_val_batches
	# -- multiply val_check_interval by accumulate_grad_batches due to lightning bug:
	# -- https://github.com/Lightning-AI/pytorch-lightning/issues/12205
	val_check_interval = args.val_check_interval if args.val_check_interval == 1.0 else args.val_check_interval * args.accumulate_grad_batches

	trainer = L.Trainer(
		accelerator="auto",
		devices = devices,
		num_nodes=args.num_nodes,
		precision=args.float_precision,
		max_epochs=args.epochs,
		max_steps=args.max_steps,
		logger=wandb_logger,
		enable_model_summary=args.log_model_archi,
		callbacks = [*callbacks, ModelSummary(max_depth=-1)],
		strategy = args.distributed_strategy,
		enable_checkpointing=True,
		fast_dev_run = args.fast_dev_run,
		num_sanity_val_steps = args.val_sanity,
		limit_train_batches = args.limit_train_batches,
		limit_val_batches = limit_val_batches,
		limit_test_batches = args.limit_test_batches,
		detect_anomaly=args.detect_anomaly,
		gradient_clip_val=gradient_clip_val,
		overfit_batches=args.overfit_batches,
		profiler=profiler,
		val_check_interval=val_check_interval,
		check_val_every_n_epoch=args.check_val_every_n_epoch,
		deterministic=args.deterministic,
		log_every_n_steps=args.log_every_n_steps,
		accumulate_grad_batches=args.accumulate_grad_batches,
		# -- disable inference_mode to retain gradients during testing
		inference_mode=False,
	)

	return trainer


def clear_cache():
	torch.cuda.empty_cache()


if __name__ == '__main__':
	args = get_args()
	main(args)
