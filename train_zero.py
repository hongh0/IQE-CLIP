import argparse
import numpy as np
import torch
from torch.nn import functional as F
import torchmetrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_zero import MedTestDataset, MedTrainDataset
from CLIP.clip import create_model
from CLIP.iqeclip import IQE_CLIP
from loss import FocalLoss, BinaryDiceLoss
from utils import  _freeze_stages, Logger,normalize,setup_seed
import json
import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Brain')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_path', type=str, default='./log/temp.log')
    parser.add_argument('--save_dir', type=str, default='ckpt')
    # hyper-parameter
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
    model_optim = _freeze_stages(model, "prompt")
    iqe_clip = IQE_CLIP(model, features_list=args.features_list, model_configs=model_configs, prompt_len=args.prompt_len, iqm_config=args.iqm_config, query_vison=True).to(device)
    
    

    parameter_prompt_list = []
    for n, m in iqe_clip.New_Lan_Embed.named_parameters():
        if n != "prompt_temp": #prompt_temp is learnable temperature coefficient $\tau_1$
            parameter_prompt_list.append(m)
        else:
            print(n)

    parameter_query_list = []        
    
    parameter_query_list.append(iqe_clip.query_tokens)


    parameter_model_prompt_list = [value for key,value in model_optim.items()]

    lr_group1 = list(iqe_clip.trainable_layer.parameters()) + list(iqe_clip.query_linear.parameters()) +  parameter_prompt_list +  parameter_model_prompt_list + list(iqe_clip.iqm.parameters()) + parameter_query_list
    lr_group2 = [iqe_clip.New_Lan_Embed.prompt_temp] 
    
    optimizer = torch.optim.Adam([{'params': lr_group1, 'lr': args.learning_rate}, {'params': lr_group2, 'lr':0.01}], lr = args.learning_rate, betas = (0.5 , 0.999))  
    
    
    features_list = args.features_list
    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedTrainDataset(args.data_path, args.obj, args.img_size, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2*args.batch_size, shuffle=False, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    save_score = 0.0

    
    for epoch in range(args.epoch):
        logger.info(f'epoch: {epoch}')

        loss_list = []
        idx = 0
        for (image, image_label, mask, seg_idx) in tqdm(train_loader):
            if idx % (len(train_loader) // 5) == 0:
                score = test(args, iqe_clip, test_loader, features_list, logger)
                if score >= save_score:
                    save_score = score
                    ckp_path = f'./{args.save_dir}/{args.obj}.pth'
                    save_dict = {'trainable_linearlayer': iqe_clip.trainable_layer.state_dict(), 
                                'New_Lan_Embed': iqe_clip.New_Lan_Embed.state_dict(), 
                                "iqm":iqe_clip.iqm.state_dict(),
                                "query_tokens": iqe_clip.query_tokens,
                                "query_linear":iqe_clip.query_linear.state_dict()}
                    save_dict.update(model_optim)
                    torch.save(save_dict, ckp_path)
                    logger.info(f'best epoch found: epoch {epoch} batch {idx}')
            idx += 1

            image = image.squeeze(0).to(device)
            seg_idx = seg_idx.item()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    image_features, patch_tokens, x_ = iqe_clip.clip.encode_image(image, features_list, return_x=True)
                class_token = iqe_clip.New_Lan_Embed.before_extract_feat(patch_tokens, image_features, use_global = args.use_global)
                text_embeddings = iqe_clip.prompt_pre.forward_ensemble(iqe_clip.clip, class_token, device)
                query_tokens = iqe_clip.get_query(image_features, use_global = args.use_global)
                text_embeddings = text_embeddings.permute(0,2,1)

                anomaly_maps_new = []
                det_loss = 0
                image_label = image_label.squeeze(0).to(device)
                patch_tokens_linear = iqe_clip.trainable_layer(patch_tokens)
                for layer in range(len(patch_tokens_linear)):
                    dense_feature = patch_tokens_linear[layer].clone()
                    
                    query_feature = iqe_clip.iqm(query_embeds=query_tokens, encoder_hidden_states=dense_feature,text_encoder_hidden_states=text_embeddings.permute(0,2,1))[0]
                    dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                    query_feature = query_feature / query_feature.norm(dim=-1, keepdim = True)
                    anomaly_map_new = (dense_feature @ query_feature.permute(0,2,1))
                    anomaly_scores_new = torch.mean(torch.softmax(anomaly_map_new, dim=-1)[:, :, 1], dim=-1).squeeze()
                    B, L, C = anomaly_map_new.shape
                    H = int(np.sqrt(L))
                    anomaly_map_new = F.interpolate(anomaly_map_new.permute(0, 2, 1).view(B,2,H,H),
                                            size = args.img_size, mode = 'bilinear', align_corners=True)

                    anomaly_map_new = torch.softmax(anomaly_map_new, dim =1) 
                    anomaly_maps_new.append(anomaly_map_new)
                    det_loss += loss_bce(anomaly_scores_new, image_label)

                anomaly_maps_raw = []
                for layer in range(len(patch_tokens_linear)):
                    dense_feature = patch_tokens_linear[layer].clone()
                    dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                    anomaly_map_raw = (iqe_clip.New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings)
                    anomaly_scores_raw = torch.mean(torch.softmax(anomaly_map_raw, dim=-1)[:, :, 1], dim=-1).squeeze()
                    B, L, C = anomaly_map_raw.shape 
                    H = int(np.sqrt(L))
                    anomaly_map_raw = F.interpolate(anomaly_map_raw.permute(0, 2, 1).view(B,2,H,H),
                                                size = args.img_size, mode = 'bilinear', align_corners=True)

                    anomaly_map_raw = torch.softmax(anomaly_map_raw, dim =1)
                    anomaly_maps_raw.append(anomaly_map_raw)
                    det_loss += loss_bce(anomaly_scores_raw, image_label)
                
                if seg_idx > 0:
                    mask = mask.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    loss_new = 0
                    for num in range(len(anomaly_maps_new)):
                        loss_new += loss_focal(anomaly_maps_new[num], mask)
                        loss_new += loss_dice(anomaly_maps_new[num][:, 1, :, :], mask)
                    loss_base = 0
                    for num in range(len(anomaly_maps_raw)):
                        loss_base += loss_focal(anomaly_maps_raw[num], mask)
                        loss_base += loss_dice(anomaly_maps_raw[num][:, 1, :, :], mask)
                    loss = loss_new + loss_base + det_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    loss = det_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_list.append(loss.item())
                torch.cuda.empty_cache()
        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

        # logs
        logger.info(f"Loss: {np.mean(loss_list)}" )
        


@torch.no_grad()
def test(args, iqe_clip, test_loader, features_list, logger):
    iqe_clip.eval()
    gt_list = []
    gt_mask_list = []
    anomaly_map_raw_list = []
    anomaly_map_new_list = []
    image_scores_raw = []
    image_scores_new = []
    img_list = []
    query_feature_list=[]
    text_embeddings_list=[]
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens, x_ = iqe_clip.clip.encode_image(image, features_list, return_x=True)

            query_tokens = iqe_clip.get_query(image_features, use_global = args.use_global)
            class_token = iqe_clip.New_Lan_Embed.before_extract_feat(patch_tokens,image_features.clone(), use_global = args.use_global)
            text_embeddings = iqe_clip.prompt_pre.forward_ensemble(iqe_clip.clip, class_token, device)

            text_embeddings = text_embeddings.permute(0,2,1)
            text_embeddings_list.append(text_embeddings.cpu().numpy())
            anomaly_maps_new = []
            anomaly_score_new = 0
            q_list=[]
            patch_tokens_linear = iqe_clip.trainable_layer(patch_tokens)
            for layer in range(len(patch_tokens_linear)):
                dense_feature = patch_tokens_linear[layer].clone()
                query_feature = iqe_clip.iqm(query_embeds=query_tokens, encoder_hidden_states=dense_feature,text_encoder_hidden_states=text_embeddings.permute(0,2,1))[0]
                dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                query_feature = query_feature / query_feature.norm(dim=-1, keepdim = True)
                q_list.append(query_feature.cpu().numpy())
                anomaly_map_new = (dense_feature @ query_feature.permute(0,2,1))
                anomaly_score_new += torch.softmax(anomaly_map_new, dim=-1)[:, :, 1].mean(dim=-1)
                B, L, C = anomaly_map_new.shape 
                H = int(np.sqrt(L))
                anomaly_map_new = F.interpolate(anomaly_map_new.permute(0, 2, 1).view(B,2,H,H),
                                            size = args.img_size, mode = 'bilinear', align_corners=True)

                anomaly_map_new = torch.softmax(anomaly_map_new, dim =1)[:, 1, :, :]

                anomaly_maps_new.append(anomaly_map_new.cpu().numpy())
            image_scores_new.append(anomaly_score_new.cpu().numpy())
            query_feature_list.append(np.mean(q_list, axis=0))
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
            img_list.append(image.squeeze().detach())
            anomaly_map_raw_list.append(anomaly_map_raw)
            anomaly_map_new_list.append(anomaly_map_new)
        
        
    
    gt_list = np.array(gt_list)
    gt_mask_list = np.concatenate(gt_mask_list, axis=0)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)
    anomaly_map_raw_list = np.concatenate(anomaly_map_raw_list, axis=0)
    anomaly_map_new_list = np.concatenate(anomaly_map_new_list, axis=0)

    segment_scores = normalize(gaussian_filter(0.2 * anomaly_map_raw_list + (1 - 0.2) * anomaly_map_new_list, sigma=8,axes = (1,2))) 


    image_scores = np.concatenate(image_scores_new, axis=0) + np.concatenate(image_scores_raw, axis=0)
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


