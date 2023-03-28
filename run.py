import os
import torch
from tqdm import tqdm
import argparse
import copy

import torch
import torch.nn.functional as F

import optuna
from optuna.samplers import TPESampler, RandomSampler

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedShuffleSplit

from models import *
from other_attacks import *
from utils import *


optuna.logging.set_verbosity(optuna.logging.WARNING)


def reset_seed(SEED=0):
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)


@torch.no_grad()
def weight_reset(m: torch.nn.Module):
	reset_parameters = getattr(m, "reset_parameters", None)
	if callable(reset_parameters):
		m.reset_parameters()


def get_best_trial(study):
	# Picks the trial with Best Val Accuracy
	# On conflict, picks the trial with best Test Accuracy
	trials = study.trials.copy()
	val_accs = np.array([t.user_attrs['Val Accuracy'] for t in study.trials])
	test_accs = np.array([t.user_attrs['Test Accuracy'] for t in study.trials])
	best_val = np.max(val_accs)
	best_val_indices = np.array(np.argwhere(val_accs == best_val).reshape(-1))
	best_index = best_val_indices[np.argmax(test_accs[best_val_indices]).reshape((-1))[0]]
	return trials[best_index]


def train_eval(model, y_train, args, verbose=False):

	def train(model, data, opt, y_train):
		model.train()
		opt.zero_grad()
		out = model(data)[data.train_mask]
		loss = F.cross_entropy(out, y_train)
		loss.backward()

		opt.step()
		del out

	def test(model, data):
		model.eval()
		out = model(data)

		# using poisoned labels for computing validation accuracy
		val_acc = (out.argmax(1)[data.val_mask] == y_poisoned[data.val_mask]).sum() / data.val_mask.sum() 
		
		test_acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

		return val_acc, test_acc


	opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model.apply(weight_reset)

	best_val_acc = test_acc = 0
	val_acc_history = []
	predictions = []

	for epoch in range(args.epochs):
		train(model, data, opt, y_train)

		val_acc, tmp_test_acc = test(model, data)

		if verbose:
			print("epoch: {:4d} \t Val Acc: {:.2f} \t Test Acc: {:.2f}".format(epoch, val_acc*100, tmp_test_acc*100))

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			test_acc = tmp_test_acc

		if epoch >= 0:
			val_acc_history.append(val_acc)
			if args.early_stopping > 0 and epoch > args.early_stopping:
				tmp = torch.tensor(
					val_acc_history[-(args.early_stopping + 1):-1])
				if val_acc < tmp.mean().item():
					# print('EARLY STOPPING!')
					break

	return best_val_acc, test_acc


def objective(trial):

	if args.hyp_param_tuning:
		args.lr = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.05, 0.08]) 
		args.weight_decay = trial.suggest_categorical('weight_decay', [0.0, 0.005, 0.0005, 0.00005]) 
		args.dropout = trial.suggest_categorical('dropout', [0.3, 0.5, 0.7]) 

		if args.model == 'APPNP':
			args.alpha = trial.suggest_categorical('alpha', [0.1, 0.3, 0.5, 0.8])

	OPTUNA_model = network(dataset, args).to(device) 

	#evaluate on n CV folds
	val_acc_log, test_acc_log = [], []
	for idx in range(len(CV_train_val_set)):

		if args.setting == 'cv':

			train_index, val_index = CV_train_val_set[idx][0], CV_train_val_set[idx][1]
			
			# reset train_mask and val_mask
			data.train_mask = index_to_mask(dataset.train_ids[train_index], size=dataset.dataset_data['X'].shape[0])	
			data.val_mask = index_to_mask(dataset.train_ids[val_index], size=dataset.dataset_data['X'].shape[0])

		y_train_poisoned = y_poisoned[data.train_mask] 

		v_acc, t_acc = train_eval(OPTUNA_model, y_train_poisoned, args, verbose=False)
		
		val_acc_log.append(v_acc.detach().cpu().numpy())
		test_acc_log.append(t_acc.detach().cpu().numpy())
		
		OPTUNA_model.apply(weight_reset) 
		reset_seed()

	avg_val_acc = np.mean(val_acc_log)
	avg_test_acc = np.mean(test_acc_log)

	trial.set_user_attr('Val Accuracy', avg_val_acc*100)
	trial.set_user_attr('Test Accuracy', avg_test_acc*100)
	trial.set_user_attr('Test Acc Log', test_acc_log)
	trial.set_user_attr('Val Acc Log', val_acc_log)

	return avg_val_acc


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--early_stopping', type=int, default=200)
	parser.add_argument('--hidden', type=int, default=64)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--dataset', type=str, default='cora_ml')
	parser.add_argument('--epochs', type=int, default=2000)
	parser.add_argument('--model', type=str, default='GCN', \
											choices=['MLP', 'GCN', 'GAT', 'APPNP', 'GCN_JKNet'])
	
	parser.add_argument('--attack', type=str, default='random', choices=['MG', 'lafak', 'random', 'degree', 'lp'])
	parser.add_argument('--setting', type=str, default='cv', choices=['small_val', 'large_val', 'cv'])
	parser.add_argument('--hyp_param_tuning', action='store_true')
	parser.add_argument('--random_train_val', action='store_true')

	#APPNP hyper-params
	parser.add_argument('--K', type=int, default=10)
	parser.add_argument('--alpha', type=float, default=0.3)

	#GAT hyper-params
	parser.add_argument('--heads', default=8, type=int)
	parser.add_argument('--output_heads', default=1, type=int)

	args = parser.parse_args()
	args_temp = copy.deepcopy(args)

	if args.model == 'GCN':
		network = GCN
	elif args.model == 'MLP':
		network = MLP
	elif args.model == 'GAT':
		network = GAT
	elif args.model == 'APPNP':
		network = APPNP_net
	elif args.model == 'GCN_JKNet':
		network = GCN_JKNet

	for poison_ratio in [0, 5, 10, 15, 20, 30]:
		
		test_accs = []
		val_accs = []

		for split in tqdm(range(10)): 
			
			args = copy.deepcopy(args_temp)
			reset_seed()

			dataset = DataLoader(args.dataset, split=split, setting=args.setting, random_train_val=args.random_train_val)

			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

			BUDGET = np.ceil(len(dataset.train_ids) * (poison_ratio/100)).astype(int)

			atk_model = network(dataset, args)
			data = dataset.data.to(device)
			model = atk_model.to(device)

			# original labels
			if poison_ratio == 0:
				y = data.y[data.train_mask].cpu()


			elif args.attack == 'MG':

				MG_out = gradient_max(adj=sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.csr_matrix(dataset.adj))), \
								 features=dataset.features, \
								 idx_train=dataset.train_ids, \
								 labels=data.y.cpu().detach().numpy(), \
								 noise_num=BUDGET, \
								 args=args)
				y_poisoned = torch.tensor(MG_out, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()


			elif args.attack == 'lafak':
				if args.setting == 'cv':
					# poisoning train + val
					pkl_data = pkl.load(open("./lafak/lafak_cv/{}_lafak_cv_poisoned_labels.pkl".format(args.dataset), 'rb'))
				
				elif args.setting == 'small_val':
					# poisoning train
					if args.random_train_val:
						pkl_data = pkl.load(open("./lafak/lafak_small_val_random/{}_lafak_smallValRandom.pkl".format(args.dataset), 'rb'))
					else:
						pkl_data = pkl.load(open("./lafak/lafak_small_val/{}_lafak_smallval_poisoned_labels.pkl".format(args.dataset), 'rb'))

				elif args.setting == 'large_val':
					pkl_data = pkl.load(open("./lafak/lafak_large_val/{}_lafak_largeVal_poisoned_labels.pkl".format(args.dataset), 'rb'))
				
				y_poisoned = pkl_data['split_{}'.format(split)]['{}_percent_poison'.format(poison_ratio)].argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()


			elif args.attack == 'random':

				y_poisoned = poison_labels_random(labels=dataset.dataset_data['labels'], \
												  train_idx=dataset.train_ids, \
												  BUDGET=BUDGET).argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()

			
			elif args.attack == 'degree':

				y_poisoned = poison_labels_degree(labels=dataset.dataset_data['labels'], \
												  train_idx=dataset.train_ids, \
												  adj=dataset.adj.todense(), \
												  BUDGET=BUDGET).argmax(1)
				y_poisoned = torch.tensor(y_poisoned, dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()

			
			elif args.attack == 'lp':
				y_poisoned = torch.tensor(lp_attack(dataset, BUDGET), dtype=torch.long).cuda()
				y = y_poisoned[data.train_mask].cpu()

			# Stratified cross validation
			skf = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

			CV_train_val_set = []

			if args.setting == 'cv':
				for train_index, val_index in skf.split(np.zeros(y.shape[0]), y):
					CV_train_val_set.append((train_index, val_index))
			else:
				CV_train_val_set.append([None, None])


			sampler = TPESampler(seed=0) 
			model_study = optuna.create_study(direction='maximize', sampler=sampler) 
			if args.hyp_param_tuning:
				model_study.optimize(objective, n_trials=30, n_jobs=1) 
			else:
				model_study.optimize(objective, n_trials=1, n_jobs=1)

			trial = get_best_trial(model_study)
			best_test_acc = trial.user_attrs['Test Accuracy']
			best_val_acc = trial.user_attrs['Val Accuracy']
			test_accs_log = trial.user_attrs['Test Acc Log']
			val_accs_log = trial.user_attrs['Val Acc Log']

			print("Split: {:2d} \t Val Acc: {:.2f} Test Acc: {:.2f}".format(split, best_val_acc, best_test_acc))

			torch.cuda.empty_cache()
			model.apply(weight_reset)

			test_accs.append(best_test_acc)
			val_accs.append(best_val_acc)
			
			del model

		print("Test Acc for Budget-{}%: {:.2f} ({:.2f})".format(poison_ratio, np.mean(test_accs), np.std(test_accs)))
	
