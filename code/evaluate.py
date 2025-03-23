import numpy as np
import torch

def evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag):
	recommends = []
	for i in range(len(top_k)):
		recommends.append([])

	with torch.no_grad():
		pred_list_all = []
		for i in gt_dict.keys(): # for each user
			if len(gt_dict[i]) != 0:
				user = torch.full((item_num,), i, dtype=torch.int64).to(args.device) # create n_item users for prediction
				item = torch.arange(0, item_num, dtype=torch.int64).to(args.device) 
				prediction = model(user, item)
				prediction = prediction.detach().cpu().numpy().tolist()
				for j in train_dict[i]: # mask train
					prediction[j] -= float('inf')
				if flag == 1: # mask validation
					if i in valid_dict:
						for j in valid_dict[i]:
							prediction[j] -= float('inf')
				pred_list_all.append(prediction)

		predictions = torch.Tensor(pred_list_all).to(args.device) # shape: (n_user,n_item)
		for idx in range(len(top_k)):
			_, indices = torch.topk(predictions, int(top_k[idx]))
			recommends[idx].extend(indices.tolist())
	return recommends

def calculate_diversity(recommends, category_features, k_idx):

    all_categories = set(category_features.values())
    recommended_categories = set()
    
    total_ild = 0.0
    user_count = 0
    
    user_ilds = []
    
    for user_idx, rec_list in enumerate(recommends[k_idx]):
        # Calculate intra-list diversity
        different_category_count = 0
        k_value = len(rec_list)
        
        # Skip empty recommendation lists
        if k_value <= 1:
            continue
            
        user_categories = set()
        
        # Count pairs with different categories
        for i_idx in range(k_value):
            item_i = rec_list[i_idx]
            if item_i in category_features:
                user_categories.add(category_features[item_i])
                recommended_categories.add(category_features[item_i])
                
            for j_idx in range(i_idx + 1, k_value):
                item_j = rec_list[j_idx]
                if item_i in category_features and item_j in category_features:
                    if category_features[item_i] != category_features[item_j]:
                        different_category_count += 1
        
        # Calculate raw ILD score according to the formula
        ild_score = (2.0 * different_category_count) / (k_value * (k_value - 1))
        user_ilds.append(ild_score)
        
        total_ild += ild_score
        user_count += 1
    
    # Calculate averages
    avg_ild = total_ild / user_count if user_count > 0 else 0
    coverage = len(recommended_categories) / len(all_categories) if len(all_categories) > 0 else 0
    
    return avg_ild, coverage, user_ilds

def calculate_f1(ndcg_scores, ild_scores):

    f1_scores = []
    
    for ndcg, ild in zip(ndcg_scores, ild_scores):
        if ndcg + ild > 0:
            f1 = (2 * ndcg * ild) / (ndcg + ild)
            f1_scores.append(f1)
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return avg_f1

def metrics(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag):

    RECALL, NDCG, ILD, F1 = [], [], [], []
    
    # Get recommendations
    recommends = evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag)
    
    # Try to get category features from the model for diversity calculation
    category_features = None
    if hasattr(model, 'category_features'):
        category_features = model.category_features
    
    for idx in range(len(top_k)):
        sumForRecall, sumForNDCG, user_length = 0, 0, 0
        k = -1
        gt_keys = list(gt_dict.keys())
        
        # To store per-user NDCG scores for F1 calculation
        user_ndcg_scores = []
        
        for i in gt_keys:  # for each user
            k += 1
            if len(gt_dict[i]) != 0:
                # Calculate Recall and NDCG
                userhit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(gt_dict[i])
                ndcg = 0

                for index, item_id in enumerate(recommends[idx][k]):
                    if item_id in gt_dict[i]:
                        userhit += 1
                        dcg += 1.0 / (np.log2(index+2))
                    if idcgCount > 0:
                        idcg += 1.0 / (np.log2(index+2))
                        idcgCount -= 1
                if idcg != 0:
                    ndcg += (dcg / idcg)

                sumForRecall += userhit / len(gt_dict[i])
                sumForNDCG += ndcg
                user_ndcg_scores.append(ndcg)
                user_length += 1

        # Calculate average metrics
        avg_recall = round(sumForRecall/user_length, 4) if user_length > 0 else 0
        avg_ndcg = round(sumForNDCG/user_length, 4) if user_length > 0 else 0

        RECALL.append(avg_recall)
        NDCG.append(avg_ndcg)
        
        # Calculate diversity metrics if category features are available
        if category_features and len(category_features) > 0:
            try:
                avg_ild, coverage, user_ilds = calculate_diversity(recommends, category_features, idx)
                
                # Calculate F1 score combining NDCG and ILD
                avg_f1 = calculate_f1(user_ndcg_scores, user_ilds)
                
                ILD.append(round(avg_ild, 4))
                F1.append(round(avg_f1, 4))
                
                # Print diversity metrics for the first k value
                if idx == 0:
                    print(f"Diversity Metrics for top-{top_k[idx]}:")
                    print(f"  ILD: {avg_ild:.4f}")
                    print(f"  Category Coverage: {coverage:.4f}")
                    print(f"  F1 Score (NDCG-ILD): {avg_f1:.4f}")
            except Exception as e:
                print(f"Error calculating diversity metrics: {e}")
                ILD.append(0.0)
                F1.append(0.0)
        else:
            ILD.append(0.0)
            F1.append(0.0)
    
    if len(ILD) > 0 and len(F1) > 0:
        return RECALL, NDCG, ILD, F1
    else:
        return RECALL, NDCG

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    
    if valid_result is not None: 
        if len(valid_result) >= 4:  # If we have ILD and F1 metrics
            print("[Valid]: Recall: {} NDCG: {} ILD: {} F1: {}".format(
                        '-'.join([str(x) for x in valid_result[0]]), 
                        '-'.join([str(x) for x in valid_result[1]]),
                        '-'.join([str(x) for x in valid_result[2]]),
                        '-'.join([str(x) for x in valid_result[3]])))
        else:  # Traditional metrics only
            print("[Valid]: Recall: {} NDCG: {}".format(
                        '-'.join([str(x) for x in valid_result[0]]), 
                        '-'.join([str(x) for x in valid_result[1]])))
    
    if test_result is not None: 
        if len(test_result) >= 4:  # If we have ILD and F1 metrics
            print("[Test]: Recall: {} NDCG: {} ILD: {} F1: {}".format(
                        '-'.join([str(x) for x in test_result[0]]), 
                        '-'.join([str(x) for x in test_result[1]]),
                        '-'.join([str(x) for x in test_result[2]]),
                        '-'.join([str(x) for x in test_result[3]])))
        else:  # Traditional metrics only
            print("[Test]: Recall: {} NDCG: {}".format(
                        '-'.join([str(x) for x in test_result[0]]), 
                        '-'.join([str(x) for x in test_result[1]])))