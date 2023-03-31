import torch
import torch.nn as nn

# Optimizers
def optimizers(
	parameters,
	args
):
	"""
	Getting a optimizer.
	"""
	if args.optimizer == "Adam":
		# print ("Optimizer: Adam")
		return torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

	elif args.optimizer == "AdamW":
		# print ("Optimizer: AdamW")
		return torch.optim.AdamW(parameters, args.lr, weight_decay=args.weight_decay)

	elif args.optimizer == "SGD":
		# print ("Optimizer: SGD")
		return torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

	elif args.optimizer == "Adagrad":
		# print ("Optimizer: Adagrad")
		return torch.optim.Adagrad(parameters, args.lr, weight_decay=args.weight_decay)

	else:
		raise 'Consider a optimizer among ("Adam", "AdamW", "SGD", "Adagrad").'
		return None


# Schedulers
def schedulers(
	optimizer,
	args
):
	"""
	Getting a scheduler.
	"""
	if args.scheduler == "CosineAnnealingLR":
		# print ("Scheduler: CosineAnnealingLR")
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-5)

	elif args.scheduler == "CosineAnnealingWarmRestarts":
		# print ("Scheduler: CosineAnnealingWarmRestarts")
		if args.warmup_epochs == 0:
			raise "No.of warmup epochs should be greater than 0."
		return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.warmup_epochs, T_mult=args.multiplier, eta_min=1e-5)

	elif args.scheduler == "ReduceLROnPlateau":
		# print ("Scheduler: ReduceLROnPlateau")
		return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

	elif args.scheduler is None or args.scheduler == "None":
		# print ("Scheduler: None")
		return None

	else:
		raise 'Consider a scheduler among ("CosineAnnealingLR", "CosineAnnealingWarmRestarts", "ReduceLROnPlateau").'
		return None