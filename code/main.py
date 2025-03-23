import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
from model import VBPR
import evaluate
import data_utils

import random
import numpy as np

if __name__ == "__main__":
	seed = 4242
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	cudnn.benchmark = True

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default="../data/", help="path for dataset")
	parser.add_argument("--model", type=str, default="VBPR", help="model name")
	parser.add_argument("--emb_size", type=int, default=64, help="embedding size (F)")
	parser.add_argument("--visual_size", type=int, default=512, help="input visual feature size (4096 in paper, 512 in our data)")
	parser.add_argument("--visual_projection_size", type=int, default=64, help="projected visual factor size (D)")
	parser.add_argument("--category_size", type=int, default=16, help="category embedding size")

	parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
	parser.add_argument("--l2_reg", type=float, default=0.0001, help="L2 regularization weight")
	parser.add_argument("--dropout", type=float,default=0.0,  help="dropout rate")
	parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
	parser.add_argument("--epochs", type=int, default=100, help="training epoches")
	parser.add_argument("--device", type=str, default="cpu")

	parser.add_argument("--alpha", type=float, default=0.5, 
                        help="weight for balancing accuracy and diversity (0: only accuracy, 1: only diversity)")
	parser.add_argument("--top_k", default='[10, 20, 50, 100]', help="compute metrics@top_k")
	parser.add_argument("--log_name", type=str, default='log', help="log_name")
	parser.add_argument("--model_path", type=str, default="./models/", help="main path for model")
	parser.add_argument("--use_diversity_reranking", action="store_true", help="use diversity-aware reranking during evaluation")

	args = parser.parse_args()

	############################ PREPARE DATASET ##########################
	train_path = args.data_path + '/training_dict.npy'
	valid_path = args.data_path + '/validation_dict.npy'
	test_path = args.data_path + '/testing_dict.npy'
	visual_path = args.data_path + '/visual_feature.npy'
	category_path = args.data_path + '/category_feature.npy'

	# load interaction data
	user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(train_path, valid_path, test_path)

	# load visual features
	visual_features = np.load(visual_path, allow_pickle=True).item()

	# load category features
	category_features = np.load(category_path, allow_pickle=True).item()

	# Print information about the data
	print(f"Number of users: {user_num}")
	print(f"Number of items: {item_num}")
	print(f"Number of visual features: {len(visual_features)}")
	print(f"Number of category features: {len(category_features)}")
    
	# Find max category ID for debugging
	if category_features:
		max_category_id = max(category_features.values())
		print(f"Maximum category ID: {max_category_id}")

	# construct the train datasets & dataloader
	train_dataset = data_utils.MFData(train_data, item_num, train_dict, True)
	train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

	########################### CREATE MODEL ##############################
	print("\nCreating model...")
	if args.model == 'VBPR':
		model = VBPR(user_num, item_num, args.emb_size, args.visual_size, args.category_size, 
                     args.visual_projection_size, args.dropout)
		print("Setting visual features...")
		model.set_visual_features(visual_features)
		print("Setting category features...")
		model.set_category_features(category_features)
		if hasattr(model, 'max_category_id'):
			print(f"Category embedding size: {model.max_category_id} x {args.category_size}")
	else:
		raise ValueError(f"Model {args.model} not supported")
	
	model.to(args.device)
	loss_function = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

	########################### TRAINING ##################################
	best_f1 = 0
	for epoch in range(args.epochs):
		# train
		model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample()

		total_loss = 0.0
		num_batches = 0

		for user, item, label in train_loader:
			user = user.to(args.device)
			item = item.to(args.device)
			label = label.float().to(args.device)

			model.zero_grad()

			prediction = model(user, item)
			loss = loss_function(prediction, label)

			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			num_batches += 1
		
		avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
		
		if (epoch+1) % 1 == 0:
			# evaluation
			model.eval()
			valid_result = evaluate.metrics(args, model, eval(args.top_k), train_dict, valid_dict, valid_dict, item_num, 0)
			test_result = evaluate.metrics(args, model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 1)
			elapsed_time = time.time() - start_time

			print('---'*18)
			print("The time elapse of epoch {:03d}".format(epoch) + " is: " +  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
			evaluate.print_results(avg_loss, valid_result, test_result)
			print('---'*18)

			current_ndcg = valid_result[1][0] 
			
			if current_ndcg > best_f1:
				best_epoch = epoch
				best_f1 = current_ndcg
				best_results = valid_result
				best_test_results = test_result
				# save model
				if not os.path.exists(args.model_path):
					os.mkdir(args.model_path)
				torch.save(model, '{}{}_{}lr_{}F_{}D_{}cat_alpha{:.1f}_{}.pth'.format(
					args.model_path, args.model, args.lr, args.emb_size, args.visual_projection_size,
					args.category_size, args.alpha, args.log_name))
				
	print('==='*18)
	print(f"End. Best Epoch is {best_epoch}")
	print(f"Best F1@10: {best_f1}")
	evaluate.print_results(None,best_results,best_test_results)