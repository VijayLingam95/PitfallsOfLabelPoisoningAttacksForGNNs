import numpy as np

import torch
import torch.nn.functional as F

import higher

from models import *
from top_k import *
from pgd import *
from SST import *

from typeguard import typechecked
from torchtyping import TensorType

import pdb

torch.autograd.set_detect_anomaly(True)



@typechecked
def margin(
		scores: TensorType["batch": ..., torch.float32, "nodes", "classes"],
		y_true: TensorType["nodes", torch.int64]
) -> TensorType["batch": ..., "nodes": ...]:
	all_nodes = torch.arange(y_true.shape[0])
	# Get the scores of the true classes.
	scores_true = scores[..., all_nodes, y_true]

	# Get the highest scores when not considering the true classes.
	scores_mod = scores.clone()
	scores_mod[..., all_nodes, y_true] = -np.inf
	scores_pred_excl_true = scores_mod.amax(dim=-1)
	return scores_true - scores_pred_excl_true


def margin_loss_soft(scores, y_true):

	scores = scores.cuda()
	y_true = y_true.cuda()

	all_nodes = torch.arange(y_true.shape[0])

	y_true_ids = torch.topk(y_true, k=1, dim=1).indices.squeeze() #[0]

	true_mask  = torch.zeros(y_true.shape).cuda()
	true_mask[all_nodes, y_true_ids] = 1.

	# Get the scores of the true classes.
	scores_true = torch.topk(torch.mul(scores, y_true), k=1, dim=1).values.squeeze()

	# Get the highest scores when not considering the true classes.
	y_neg = 1  - true_mask.clone()

	scores_pred_excl_true = torch.topk(torch.mul(scores, y_neg), k=2, dim=1).values.squeeze() #[0] torch.mul(scores, y_neg)

	return torch.sum(scores_true.view(-1, 1) - scores_pred_excl_true, dim=1)   




def k_subset_selection(soft_top_k, log_lab, BUDGET):
	"""
	k-hot vector selection from Stochastic Softmax Tricks
	"""
	_, topk_indices = torch.topk(log_lab, BUDGET)
	X = torch.zeros_like(log_lab).scatter(-1, topk_indices, 1.0)
	hard_X = X
	hard_topk = (hard_X - soft_top_k).detach() + soft_top_k
	budget_top_k = hard_topk.T
	
	return budget_top_k

def _project(budget, values, eps = 1e-7):
	r"""Project :obj:`values`:
	:math:`budget \ge \sum \Pi_{[0, 1]}(\text{values})`."""

	values1 = values.clone().detach()
	if torch.clamp(values1, 0, 1).sum() > budget:
		left = (values1 - 1).min()
		right = values1.max()
		miu = _bisection(values1, left, right, budget)
		values = values - miu
	return torch.clamp(values, min=eps, max=1 - eps)


def _bisection(edge_weights, a, b, n_pert,
				eps=1e-5, max_iter=1e3):
	"""Bisection search for projection."""
	def shift(offset: float):
		return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)

	miu = a
	for _ in range(int(max_iter)):
		miu = (a + b) / 2
		# Check if middle point is root
		if (shift(miu) == 0.0):
			break
		# Decide the side to repeat the steps
		if (shift(miu) * shift(a) < 0):
			b = miu
		else:
			a = miu
		if ((b - a) <= eps):
			break
	return miu

#seeding
def reset_seed(SEED=0):
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

@torch.no_grad()
def weight_reset(m: torch.nn.Module):
	# - check if the current module has reset_parameters & if it's callabed called it on m
	reset_parameters = getattr(m, "reset_parameters", None)
	if callable(reset_parameters):
		m.reset_parameters()

def train(model, data, opt, y_train):
	model.train()
	opt.zero_grad()
	out = model(data)[data.train_mask]
	loss = F.cross_entropy(out, y_train)
	loss.backward()

	opt.step()

def test(model, data):
	model.eval()
	out = model(data)
	val_acc = (out.argmax(1)[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
	test_acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

	return val_acc, test_acc


def meta_attack_labels(model, dataset, args, BUDGET, device, verbose=False):

	reset_seed()

	data = dataset.data

	num_classes = data.y.max() + 1

	poison_model = MLP(dataset, args).to(device) #poison_network

	eye = torch.eye(num_classes).to(device)

	inner_opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) #args.lr #args.weight_decay

	# top-k based log-lab
	top_k = SoftTopK() # SST based soft-top-k

	if args.naive_log_lab:
		# Naive log-lab
		log_lab = torch.nn.Parameter(torch.log(eye[data.y[data.train_mask]]+0.01))
	else:
		log_lab = torch.nn.Parameter(torch.zeros(1, data.false_onehot.shape[0]).to(device)) # for SIMPLE top-k
		torch.nn.init.uniform_(log_lab)
	
	# pre-train meta model
	# pre_opt = torch.optim.Adam(poison_model.parameters(), lr=0.001, weight_decay=5e-4)
	# for e in range(100):
	# 	train(poison_model, data, pre_opt, data.y[data.train_mask]) 
	# 	# val_acc, test_acc = test(poison_model, data)
	# 	# print("epoch: {:4d} \t Val Acc: {:.2f} \t Test Acc: {:.2f}".format(e, val_acc*100, test_acc*100))
	# log_lab = torch.nn.Parameter(poison_model(data)[data.train_mask].flatten().reshape(1, -1))

	# log_lab = torch.nn.Parameter((poison_model(data)[data.train_mask]).detach())
	meta_opt = torch.optim.Adam([log_lab], lr=0.1, weight_decay=1e-6)
	# scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_opt, gamma=0.8)

	# meta_opt = torch.optim.Adam(poison_model.parameters(), lr=0.1, weight_decay=5e-4)


	# for tracking best poisoned labels w.r.t meta test accuracy
	best_test_acc = 100
	best_poisoned_labels = None
	PATIENCE = 20
	patience_ctr = 0

	for ep in range(200):
		poison_model.train()
		# meta_opt.zero_grad()

		# log_lab = poison_model(data)[data.train_mask].flatten().reshape(1, -1) #F.softmax(, dim=1)

		if args.topk_type == 'soft' and args.sampling and not args.naive_log_lab:
			soft_top_k = top_k.apply(log_lab, 2*BUDGET, 1e-2) # SST based soft-top-k
			budget_top_k = soft_top_k.T

		elif args.topk_type == 'hard' and not args.naive_log_lab:
			# hard_top_k from SST
			soft_top_k = top_k.apply(log_lab, BUDGET, 1e-2) # SST based soft-top-k
			budget_top_k = k_subset_selection(soft_top_k, log_lab, BUDGET)

		elif args.topk_type == 'pgd' and not args.naive_log_lab:
			budget_top_k = _project(BUDGET, log_lab).T



		if args.naive_log_lab:
			poisoned_labels = log_lab 	 	

		else:

			H = torch.mul(data.false_onehot.cuda(), budget_top_k)
			H_reshaped = torch.reshape(H, (data.y_train_onehot.shape[0], data.y_train_onehot.shape[1], data.y_train_onehot.shape[1]))
			H_final = torch.sum(H_reshaped, dim=1)
			
			P_reshaped = torch.reshape(budget_top_k, (data.y_train_onehot.shape[0], data.y_train_onehot.shape[1], 1)).sum(dim=1)

			poisoned_labels = ((1-P_reshaped) * data.y_train_onehot.cuda() + H_final).cuda()

		with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
			fmodel.train()


			#Todo: add early stopping based on val acc

			for epoch in range(10):
				out = F.softmax(fmodel(data), dim=1) #fmodel(data)
				# loss = F.cross_entropy(out[data.train_mask], F.softmax(log_lab, dim=1)) 
				# loss = F.cross_entropy(out[data.train_mask], poisoned_labels) 

				if args.naive_log_lab:
					#gumbel_softmax margin loss
					flips =  F.gumbel_softmax(poisoned_labels, tau=1000, hard=True, dim=1) #F.softmax(poisoned_labels, dim=1)
					# loss = -margin_loss_soft(out[data.train_mask], flips).tanh().mean()

					loss = F.cross_entropy(out[data.train_mask], flips)
				else:
					# soft margin loss
					# loss = -margin_loss_soft(out[data.train_mask], poisoned_labels).tanh().mean()

					loss = F.cross_entropy(out[data.train_mask], poisoned_labels) #F.softmax(poisoned_labels, dim=1)

				diffopt.step(loss)

				# #project
				# poisoned_labels = _project(budget=BUDGET, values=poisoned_labels.flatten()).reshape(temp_shape)


			fmodel.eval()
			meta_out = fmodel(data)

			# train_acc = (meta_out.argmax(1)[data.train_mask] == log_lab.argmax(1)).sum() / data.train_mask.sum() 
			train_acc = (meta_out.argmax(1)[data.train_mask] == poisoned_labels.argmax(1)).sum() / data.train_mask.sum()
			acc = (meta_out.argmax(1)[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

			# n_poisoned = (log_lab.argmax(1) != data.y[data.train_mask]).sum() 
			n_poisoned = (poisoned_labels.argmax(1) != data.y[data.train_mask]).sum()

			if verbose:
				# print('acc', acc.cpu().numpy(), 'n_poisoned', n_poisoned.cpu().numpy(), 'diff', diff.cpu().numpy())

				print("epoch: {:4d} \t Meta-Train Acc: {:.2f} \t Meta-Test Acc: {:.2f} \t N-poisoned: {:4d} \t patience ctr: {:2d}".format(ep, \
																						train_acc.cpu().numpy(), acc.cpu().numpy()*100, \
																						n_poisoned.cpu().numpy(), patience_ctr)) #diff.cpu().numpy()

			if args.naive_log_lab or args.topk_type == 'pgd':
				if n_poisoned > BUDGET: 
					break

			# early stopping based on meta-test acc
			if (acc < best_test_acc):
				best_test_acc = acc
				best_poisoned_labels = poisoned_labels.type(torch.float32).detach() #log_lab.detach()
				patience_ctr = 0 
			else:
				patience_ctr += 1

			if patience_ctr >= PATIENCE:
				return best_poisoned_labels

			meta_loss =  margin(meta_out[data.test_mask], data.y[data.test_mask]).tanh().mean()  
			# meta_loss = -F.cross_entropy(meta_out[data.test_mask], data.y[data.test_mask])
			# meta_loss = (meta_out[data.test_mask] * eye[data.y[data.test_mask]]).mean()

			meta_opt.zero_grad()
			meta_loss.backward()
			meta_opt.step()
			# scheduler.step()

	poison_model.apply(weight_reset)
	torch.cuda.empty_cache()

	# return log_lab.detach()
	return best_poisoned_labels



