import argparse

def Parse_Arguments():
	# Argument Parser
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


	# Dataset Paths
	parser.add_argument('data', default='imagenet', help='Path to dataset. (default: imagenet)')
	parser.add_argument('--num-workers', default=64, type=int, help='No.of data loading workers. (default: 64)')
	parser.add_argument('--image-shape', default=(240,240), type=tuple, help='Dimensions of image during training. (default: (224,224))')


	# Mode
	parser.add_argument('--train', action='store_true', help='Training model on training dataset and simultaneouly validating on validation dataset.')
	parser.add_argument('--evaluate', action='store_true', help='Evaluating model on validation dataset.')


	# Path
	parser.add_argument('--main-path', default='', type=str, help='Path to main.py or train.py.')
	parser.add_argument('--resume-ckpt-path', default=None, type=str, help='Path to checkpoints to resume training. Note: The model will be save according to the main-path, dataset-name and model-name.')
	parser.add_argument('--load-weights-ckpt-path', default=None, type=str, help='Path to checkpoints to load weights before start of training.')


	# Training Parameters
	parser.add_argument('--epochs', default=100, type=int, help='No.of total epochs for training. (default: 300)')
	parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size during training. This is the total batch size of all GPUs on the current node when using  Distributed Data Parallel Strategy. (default: 16)')

	
	# Optimizer and it's Parameters
	parser.add_argument('--optimizer', default="AdamW", type=str, help='Optimizer for training. (options: "Adam", "AdamW", "SGD", "Adagrad") (default: "AdamW")')
	parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate. (default: 0.001)')
	parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD Optimizer. (default: 0.9)')
	parser.add_argument('--weight-decay', default=0.05, type=float, help='Weight Decay for all optimizers. (default: 0.05)')


	# Schedulers and it's Parameters
	parser.add_argument('--scheduler', default="CosineAnnealingLR", type=str, help='Scheduler for learning-rate during training. (options: "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "ReduceLROnPlateau") (default: "CosineAnnealingWarmRestarts")')
	parser.add_argument('--warmup-epochs', default=5, type=int, help='No.of warmup epochs for cosine scheduler. (default: 5)')
	parser.add_argument('--multiplier', default=19, type=int, help='Multiplier of no.of warmup epochs for next cycle. (default: 19)')

	
	# Distributed Training Parameters
	parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes for distributed training (default: 1).')
	parser.add_argument('--gpu', default=1, type=int, help='Number of GPUs per nodes for distributed training (default: 1).')
	
	
	# Model and Dataset Names
	parser.add_argument('--model-name', default="", type=str, help="Name of model (Used during saving).")
	parser.add_argument('--dataset-name', default="", type=str, help="Name of dataset (Used during saving).")

	return parser.parse_args()