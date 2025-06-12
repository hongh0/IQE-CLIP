import os
import argparse
import numpy as np
import torch
import torchmetrics
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.iqeclip import IQE_CLIP
from utils import  _load_stages, Logger,normalize,setup_seed
import warnings
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Brain')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--log_path', type=str, default='./log/temp.log')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/Brain.pth')
    
    parser.add_argument("--iqm_config", type=str, default='./config/config_iqm.json')
    parser.add_argument("--prompt_len", type=int, default=2, help="the length of the learnable category vectors r")
    parser.add_argument("--use_global", default=True, action="store_false")
    parser.add_argument("--deep_prompt_len", type=int, default=1, help="the length of the learnable text embeddings n ")
    parser.add_argument("--total_d_layer_len", type=int, default= 11, help="number of layers for the text encoder with learnable text embeddings")
    args = parser.parse_args()
    setup_seed(args.seed)
    
    logger = Logger(args.log_path)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    with open(f'./CLIP/model_configs/{args.model_name}.json', 'r') as f:
        model_configs = json.load(f)
    model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, 
                            require_pretrained=True, deep_prompt_len = args.deep_prompt_len, total_d_layer_len = args.total_d_layer_len).to(device)
    model.eval()
    iqe_clip = IQE_CLIP(model, features_list=args.features_list, model_configs=model_configs, prompt_len=args.prompt_len, iqm_config=args.iqm_config, query_vison=True).to(device)
    iqe_clip.eval()
    
    checkpoint = torch.load(args.ckpt_path, map_location= device)
    iqe_clip.trainable_layer.load_state_dict(checkpoint["trainable_linearlayer"], strict=False)
    iqe_clip.New_Lan_Embed.load_state_dict(checkpoint["New_Lan_Embed"])
    iqe_clip.iqm.load_state_dict(checkpoint["iqm"], strict=False)
    iqe_clip.query_tokens.data.copy_(checkpoint['query_tokens'])
    iqe_clip.query_linear.load_state_dict(checkpoint['query_linear'])
    _load_stages(model, checkpoint, "prompt")

    kwargs = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*4, shuffle=False, **kwargs)
    
    score = test(args, iqe_clip, test_loader, args.features_list, logger)


@torch.no_grad()
def test(args, iqe_clip, test_loader, features_list, logger):
    iqe_clip.eval()
    gt_list = []
    gt_mask_list = []
    anomaly_map_raw_list = []
    anomaly_map_new_list = []
    image_scores_raw = []
    image_scores_new = []
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens, x_ = iqe_clip.clip.encode_image(image, features_list, return_x=True)

            query_tokens = iqe_clip.get_query(image_features, use_global = args.use_global)
            class_token = iqe_clip.New_Lan_Embed.before_extract_feat(patch_tokens,image_features.clone(), use_global = args.use_global)
            text_embeddings = iqe_clip.prompt_pre.forward_ensemble(iqe_clip.clip, class_token, device)

            text_embeddings = text_embeddings.permute(0,2,1)
            anomaly_maps_new = []
            anomaly_score_new = 0
            patch_tokens_linear = iqe_clip.trainable_layer(patch_tokens)
            for layer in range(len(patch_tokens_linear)):
                dense_feature = patch_tokens_linear[layer].clone()
                query_feature = iqe_clip.iqm(query_embeds=query_tokens, encoder_hidden_states=dense_feature,text_encoder_hidden_states=text_embeddings.permute(0,2,1))[0]
                dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                query_feature = query_feature / query_feature.norm(dim=-1, keepdim = True)
                anomaly_map_new = (dense_feature @ query_feature.permute(0,2,1))
                anomaly_score_new += torch.softmax(anomaly_map_new, dim=-1)[:, :, 1].mean(dim=-1)
                B, L, C = anomaly_map_new.shape 
                H = int(np.sqrt(L))
                anomaly_map_new = F.interpolate(anomaly_map_new.permute(0, 2, 1).view(B,2,H,H),
                                            size = args.img_size, mode = 'bilinear', align_corners=True)

                anomaly_map_new = torch.softmax(anomaly_map_new, dim =1)[:, 1, :, :]

                anomaly_maps_new.append(anomaly_map_new.cpu().numpy())
            image_scores_new.append(anomaly_score_new.cpu().numpy())

            anomaly_maps_raw = []
            anomaly_score_raw = 0
            for layer in range(len(patch_tokens_linear)):
                dense_feature = patch_tokens_linear[layer].clone()
                dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                anomaly_map_raw = (iqe_clip.New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings)
                anomaly_score_raw += torch.softmax(anomaly_map_raw, dim=-1)[:, :, 1].mean(dim=-1)
                B, L, C = anomaly_map_raw.shape 
                H = int(np.sqrt(L))
                anomaly_map_raw = F.interpolate(anomaly_map_raw.permute(0, 2, 1).view(B,2,H,H),
                                            size = args.img_size, mode = 'bilinear', align_corners=True)

                anomaly_map_raw = torch.softmax(anomaly_map_raw, dim =1)[:, 1, :, :]
                anomaly_maps_raw.append(anomaly_map_raw.cpu().numpy())
            image_scores_raw.append(anomaly_score_raw.cpu().numpy())

            anomaly_map_raw = np.mean(anomaly_maps_raw, axis=0)
            anomaly_map_new = np.mean(anomaly_maps_new, axis=0)

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            anomaly_map_raw_list.append(anomaly_map_raw)
            anomaly_map_new_list.append(anomaly_map_new)
        
        

    gt_list = np.array(gt_list)
    gt_mask_list = np.concatenate(gt_mask_list, axis=0)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    anomaly_map_raw_list = np.concatenate(anomaly_map_raw_list, axis=0)
    anomaly_map_new_list = np.concatenate(anomaly_map_new_list, axis=0)

    segment_scores = normalize(gaussian_filter(0.2 * anomaly_map_raw_list + (1 - 0.2) * anomaly_map_new_list, sigma=8,axes = (1,2))) 
   
    image_scores = np.mean(anomaly_map_raw_list.reshape(anomaly_map_raw_list.shape[0], -1), axis=1) + np.mean(anomaly_map_new_list.reshape(anomaly_map_new_list.shape[0], -1), axis=1)
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())
    
    gt_list_tensor = torch.from_numpy(gt_list).long().to(device)
    image_scores_tensor = torch.from_numpy(image_scores).float().to(device).squeeze()
    img_roc_auc_det = torchmetrics.functional.auroc(image_scores_tensor, gt_list_tensor, task='binary').item()
    logger.info(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

   
    if CLASS_INDEX[args.obj] > 0:
        gt_mask_list_tensor = torch.from_numpy(gt_mask_list).float().to(device)
        segment_scores_tensor = torch.from_numpy(segment_scores).float().to(device).squeeze()
        seg_roc_auc = torchmetrics.functional.auroc(segment_scores_tensor, gt_mask_list_tensor.long(), task='binary').item()
        logger.info(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')

        return seg_roc_auc + img_roc_auc_det
    else:
        return img_roc_auc_det




if __name__ == '__main__':
    main()


