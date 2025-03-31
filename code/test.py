import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

import model
import evaluate
import data_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", default='[10, 20, 50, 100]', help="compute metrics@top_k")
    parser.add_argument("--data_path", type=str, default="../data/", help="main path for dataset")
    parser.add_argument("--model", type=str, default="MF", help="model name")
    parser.add_argument("--ckpt", type=str, default="MF_0.001lr_64emb_log.pth")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    cudnn.benchmark = True

    ############################## PREPARE DATASET ##########################
    train_path = args.data_path + '/training_dict.npy'
    valid_path = args.data_path + '/validation_dict.npy'
    test_path = args.data_path + '/testing_dict.npy'
    category_path = args.data_path + '/category_feature.npy'
    visual_path = args.data_path + '/visual_feature.npy'
    # test_path = args.data_path + '/heldout_dict.npy' # for live evaluation
    user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(train_path, valid_path, test_path)

    ########################### LOAD MODEL #################################
    # Load checkpoint
    checkpoint = torch.load(f"./models/{args.ckpt}", map_location=args.device)
    
    # Initialize model
    model = model.VBPR(
        user_num, item_num,
        embedding_size=64,
        visual_size=512,
        category_size=16,
        visual_projection_size=64,
        dropout=0.0,
        alpha=checkpoint.get('alpha', 0.5)  # Load saved alpha, will be overwritten by checkpoint value
    )
    
    # Load weights and alpha
    model.load_state_dict(checkpoint['model_state_dict'])
    model.alpha.data = torch.tensor(checkpoint['alpha'])  # Explicitly set alpha
    
    model.to(args.device)

    try:
        category_features = np.load(category_path, allow_pickle=True).item()
        model.set_category_features(category_features)
        print(f"Succecssfully loaded category features: {len(category_features)} items")
    except Exception as e:
        print(f"Error loading category features: {e}")

    try:
        visual_features = np.load(visual_path, allow_pickle=True).item()
        model.set_visual_features(visual_features)
        print(f"Succecssfully loaded visual features: {len(visual_features)} items")
    except Exception as e:
        print(f"Error loading visual features: {e}")

    ########################### EVALUATION #####################################
    model.eval()
    
    # test_result = evaluate.metrics(args, model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 1)
    
    valid_result = evaluate.metrics(args, model, eval(args.top_k), train_dict, valid_dict, valid_dict, item_num, 0)

    print('---'*18)
    # print("Test Results:")
    # evaluate.print_results(None, None, test_result) 
    print("Validation Results:")
    evaluate.print_results(None, None, valid_result) 
    print('---'*18)