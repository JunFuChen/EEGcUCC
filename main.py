import math
import numpy as np
import torch
from modules import network, contrastive_loss, BrainMamba, losses
from EEGloader import  loadata
from utils_metrics import clusteringMetricsv2
import os


def select_dataset(x, y, seed=32, ratio=0.1):
    labfea = []
    lab_groundtruth = []
    unlabfea = []
    unlab_groundtruth = []
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    class_indices = {label: np.where(y == label)[0] for label in unique_labels}
    np.random.seed(int(seed))
    for label, indices in class_indices.items():
        num_samples = len(indices)
        num_10_percent = max(1, int(np.ceil(num_samples * ratio)))  
        np.random.shuffle(indices)
        selected_indices_10_percent = indices[:num_10_percent]  
        remaining_indices_90_percent = indices[num_10_percent:]  

        labfea.append(x[selected_indices_10_percent])
        lab_groundtruth.append(y[selected_indices_10_percent])
        unlabfea.append(x[remaining_indices_90_percent])
        unlab_groundtruth.append(y[remaining_indices_90_percent])
    labfea = np.concatenate(labfea, axis=0)
    lab_groundtruth = np.concatenate(lab_groundtruth, axis=0)
    unlabfea = np.concatenate(unlabfea, axis=0)
    unlab_groundtruth = np.concatenate(unlab_groundtruth, axis=0)
    return labfea, lab_groundtruth, unlabfea, unlab_groundtruth

def normalized_entropy(y: torch.Tensor):
    return -1 * y.mul(y.log()).sum(dim=1) / torch.log(torch.tensor(y.shape[1]))

def calculate_padding(input_size, kernel_size, stride, dilation=1):
    output_size = (input_size + stride - 1) // stride
    padding = max(0, (output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size)
    return padding


def compute_prototypes(features, labels):
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    feature_dim = features.shape[1]
    prototypes = torch.zeros((num_classes, feature_dim), device=features.device)

    for i, label in enumerate(unique_labels):
        class_features = features[labels == label]  
        prototype = class_features.mean(dim=0)  
        prototypes[i] = prototype
    return prototypes, unique_labels


from utils import sliece_len_of_dataset, IJS2Simi, PairEnum

class augumentation(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.03 
        self.max_seg = 8


def logof_dataset(data_path):
    if data_path == 'IV_3_s1_data' or data_path == 'IV_3_s2_data':
        channel, lenth = 10, 4000
        K_cluster = 4
        layerunit = [2048, 2048, 512, 512, 2048, 2048]
        layerpar = [4000, 2048, 512, 20, 512, 2048, 4000]
        z_size = 64
    if data_path == 'IV_2b_s1_data' or data_path == 'IV_2b_s2_data' or data_path == 'IV_2b_s3_data':
        channel, lenth = 3, 939
        K_cluster = 2
        layerunit = [1024, 512, 256, 256, 2048, 2048]
        layerpar = [939, 600, 300, 20, 300, 600, 939]
        z_size = 16
    if data_path == 'IV_2a_s1_data' or data_path == 'IV_2a_s2_data' or data_path == 'IV_2a_s3_data':
        channel, lenth = 22, 6886
        K_cluster = 4
        layerunit = [2048, 2048, 512, 512, 2048, 2048]
        layerpar = [6886, 2048, 512, 20, 512, 2048, 6886]
        z_size = 64
    if data_path == 'III_V_s1_data' or data_path == 'III_V_s2_data' or data_path == 'III_V_s3_data':
        channel, lenth = 8, 96
        K_cluster = 3
        layerunit = [128, 64, 64, 64, 64, 128]
        layerpar = [96, 128, 64, 10, 64, 128, 96]
        z_size = 16
    if data_path == 'II_Ia_data':
        channel, lenth = 6, 5376
        K_cluster = 2
        layerunit = [2048, 2048, 512, 512, 2048, 2048]
        layerpar = [5376, 2048, 512, 20, 512, 2048, 5376]
        z_size = 64
    if data_path == 'II_Ib_data':
        channel, lenth = 7, 8064
        K_cluster = 2
        layerunit = [2048, 2048, 512, 512, 2048, 2048]
        layerpar = [8064, 2048, 512, 20, 512, 2048, 8064]
        z_size = 64
    if data_path == 'II_Ia_Ib_data':
        channel, lenth = 6, 5376
        K_cluster = 4
        layerunit = [2048, 2048, 512, 512, 2048, 2048]
        layerpar = [5376, 2048, 512, 20, 512, 2048, 5376]
        z_size = 64

    return channel, lenth, K_cluster, layerunit, z_size, layerpar


from EEGloader import DataTransform_TD


class aug_config(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.03  
        self.max_seg = 8


def train_epoch(align_lab, labfea, labfea_augview, lab_groundtruth, unlabfea, unlabfea_augview, batch_size, lab_batch_size, unlab_batch_size):
    model.train()
    loss_epoch = 0
    num_batch = int(math.ceil(1.0 * x_.shape[0] / batch_size))
    for batch_idx in range(num_batch):
        alignlab_batch = align_lab[batch_idx * lab_batch_size: min((batch_idx + 1) * lab_batch_size, labfea.shape[0])]
        lab_feabatch = labfea[batch_idx * lab_batch_size: min((batch_idx + 1) * lab_batch_size, labfea.shape[0])]
        labfea_augview_batch = labfea_augview[batch_idx * lab_batch_size: min((batch_idx + 1) * lab_batch_size, labfea.shape[0])]

        lab_batch = lab_groundtruth[batch_idx * lab_batch_size: min((batch_idx + 1) * lab_batch_size, labfea.shape[0])]

        unlabfea_batch = unlabfea[batch_idx * unlab_batch_size: min((batch_idx + 1) * unlab_batch_size, unlabfea.shape[0])]
        unlabfea_augview_batch = unlabfea_augview[batch_idx * unlab_batch_size: min((batch_idx + 1) * unlab_batch_size, unlabfea.shape[0])]
        all_feabatch_aug = torch.cat([labfea_augview_batch,unlabfea_augview_batch])
        all_feabatch = torch.cat([lab_feabatch, unlabfea_batch])
        _, _, c_i, c_j = model(all_feabatch, all_feabatch_aug)
        loss_cluster = criterion_cluster(c_i, c_j) # L_q


        lab_predcit_logit = model.forward_pseudoass(lab_feabatch)  
        lab_predcit_psl = torch.argmax(lab_predcit_logit, dim=1)
        lab_predict_fea,_,_,_ = model.forward(lab_feabatch,labfea_augview_batch) 


        unlab_predict_logit = model.forward_pseudoass(unlabfea_batch)  
        unlab_predict_psl = torch.argmax(unlab_predict_logit, dim=1)


        unlab_augpredict_logit = model.forward_pseudoass(unlabfea_augview_batch)
        unlab_augpredict_psl = torch.argmax(unlab_augpredict_logit, dim=1)
        _, unlab_augpredict_fea,_,_ = model.forward(unlabfea_batch,unlabfea_augview_batch)
        proto, prolab = compute_prototypes(lab_predict_fea, lab_batch)


        prob1, prob2 = PairEnum(lab_predcit_logit)  
        lab_batch_ = lab_batch.clone().detach()
        hard_constraint = (lab_batch_.unsqueeze(1) == lab_batch_).int()
        hard_constraint = hard_constraint.view(-1)
        sup_constraint = criterion_cons(prob1, prob2, hard_constraint)


        disc_prelogit = unlab_predict_logit.clone().detach()  
        nentropy = normalized_entropy(y=disc_prelogit)
        high_informative_ind = nentropy < threshold
        low_informative_ind = nentropy >= threshold

        disc_prepsl = unlab_predict_psl.clone().detach() 


        unlab_inforpredictfea = unlab_augpredict_fea[high_informative_ind]  
        unlab_infor_psl = disc_prepsl[high_informative_ind] 

        unlab_uninforpredictfea = unlab_augpredict_fea[low_informative_ind]  
        unlab_uninfor_psl = disc_prepsl[low_informative_ind] + torch.tensor(K_cluster).to(device) 


        Z_union = torch.cat((lab_predict_fea, unlab_inforpredictfea, unlab_uninforpredictfea, proto), dim=0)
        Z_union =  Z_union.unsqueeze(1) 
        y_union = torch.cat((alignlab_batch, unlab_infor_psl, unlab_uninfor_psl, prolab))

        unioncl_loss = criterion_unioncl(Z_union, y_union)
        unlab_infor_passignment = unlab_predict_logit[high_informative_ind]
        unlab_infor_augpassignment = unlab_augpredict_logit[high_informative_ind] 

        prob1_saug, prob2_saug = PairEnum(unlab_infor_augpassignment)




        ijs_targets = IJS2Simi(y=unlab_infor_passignment.detach())
        unsup_constraint = criterion_cons(prob1=prob1_saug, prob2=prob2_saug, simi=ijs_targets, pseudo_targets=True)
        constraint_loss = sup_constraint

        loss = loss_cluster + 0.1 * unioncl_loss + 0.1 * constraint_loss
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch

import pandas as pd

def eval_whole(labfea, lab_groundtruth, unlabfea, unlab_groundtruth):
    model.eval()
    with torch.no_grad():
        unlab_predcit_logit = model.forward_pseudoass(unlabfea)
        unlab_predcit = torch.argmax(unlab_predcit_logit, dim=1)
        lab_predcit_logit = model.forward_pseudoass(labfea)  
        lab_predcit = torch.argmax(lab_predcit_logit, dim=1)

    unlab_predcit = unlab_predcit.cpu().detach().numpy()
    lab_predcit = lab_predcit.cpu().detach().numpy()
    all_predict = np.concatenate((unlab_predcit, lab_predcit), dim=0)
    all_ground = np.concatenate((unlab_groundtruth, lab_groundtruth), dim=0)
    RI, ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetricsv2(K_cluster, all_ground, all_predict)
    print(f"Epoch [{epo}/{epochs}]\t :RI, NMI, Fscore, ACC, ARI {RI}, {NMI}, {Fscore}, {ACC}, {ARI}")
    stacked_ = np.stack((all_ground, all_predict), dim =1)
    df = pd.DataFrame(stacked_, columns=["ground_t","predict"])
    df.to_excel("clustering_results/"+dataset + "/epoch"+str(epo)+ "ratio"+ str(int(100* ratio))   +"evalunlab.xlsx", index=False)



def eval_unlabel(unlabfea, unlab_groundtruth):
    model.eval()
    with torch.no_grad():
        unlab_predcit_logit = model.forward_pseudoass(unlabfea) 
        unlab_predcit = torch.argmax(unlab_predcit_logit, dim=1)
    unlab_predcit = unlab_predcit.cpu().detach().numpy()
    unlab_gt =  unlab_groundtruth.cpu().detach().numpy()
    RI, ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetricsv2(K_cluster, unlab_gt,unlab_predcit)
    print(f"Epoch [{epo}/{epochs}]\t RI, NMI, Fscore, ACC, ARI {RI}, {NMI}, {Fscore}, {ACC}, {ARI}")
    stacked_ = np.stack((unlab_gt, unlab_predcit), axis=1)
    df = pd.DataFrame(stacked_, columns=["ground_t","predict"])
    df.to_excel(dataset + "epoch"+str(epo)+"evalunlab.xlsx", index=False)

dai_bciIII_V = ['III_V_s1_data', 'III_V_s2_data', 'III_V_s3_data']  
dai_bciIV_2a = ['IV_2a_s1_data', 'IV_2a_s2_data', 'IV_2a_s3_data']  #
dai_bciIV_2b = ['IV_2b_s1_data', 'IV_2b_s2_data']  
dai_bciIV_3 = ['IV_3_s1_data', 'IV_3_s2_data'] 
dai_bciII = ['II_Ia_data', 'II_Ib_data', 'II_Ia_Ib_data']  
data_all = dai_bciII + [subset for subset in dai_bciIII_V[:2]] + dai_bciIV_2a + dai_bciIV_2b + dai_bciIV_3

if __name__ == "__main__":
    ratio = 0.1 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 32
    embed_size = 2
    epochs = 200
    batch_size = 100  
    threshold = 0.3

    for dataset in ['III_V_s1_data']:
        folder_path = "clustering_results/ " + dataset
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"filefolder '{folder_path}' will be established")
        else:
            print(f"filefolder '{folder_path}' exists")


        pretrain_path = 'cc_pretrain/model_params' + dataset + 'best_RI.pth'
        print('pre-train model path: ',pretrain_path)


        _, _, _, slice_num_list = sliece_len_of_dataset(dataset)
        slice_num = slice_num_list[0]  
        augudefault = aug_config()
        channel, lenth, K_cluster, _, _, _ = logof_dataset(dataset)
        data_lab, data_fea, nor_data = loadata(dataset)
        nor_data = nor_data.astype(np.float32)
        reshape_data = nor_data.reshape(-1, channel, lenth // channel)

        channel_len = int(lenth / channel)
        kernel_size = math.ceil(channel_len / slice_num)

        stride = kernel_size
        padding = calculate_padding(channel_len, kernel_size, stride,
                                    dilation=1) 
        padding = int(padding)
        embed_dims = 32
        feature_dim, class_num = 8, K_cluster
        embednet1 = BrainMamba.embednet(patch_channel_timestep=[slice_num, channel, kernel_size],
                                        arch=[embed_dims, 2],
                                        padding=padding,
                                        pe_type='learnable',
                                        path_type='forward_reverse_shuffle_gate',
                                        cls_position='none',  
                                        out_indices=-1,
                                        drop_rate=0.,
                                        norm_cfg=dict(type='LN', eps=1e-6),
                                        final_norm=True,
                                        out_type='raw',
                                        frozen_stages=-1,
                                        interpolate_mode='bicubic',
                                        layer_cfgs=dict(),
                                        init_cfg=None)
        model = network.Network(embednet1, feature_dim, class_num)
        model.load_state_dict(torch.load(pretrain_path, map_location=device))
        model = model.to(device)


        cluster_temperature = 1.0
        instance_temperature = 0.5
        criterion_unioncl = contrastive_loss.SupConLoss(device)
        criterion_cons = losses.MBCE()
        criterion_cluster = contrastive_loss.ClusterLoss(class_num, cluster_temperature, device).to(device)
        criterion_instance = contrastive_loss.InstanceLoss(batch_size, instance_temperature, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00003, weight_decay=0.0003)

        reshape_data = nor_data.reshape(-1, channel, lenth // channel)
        x_ = reshape_data.astype(np.float32)
        y_ = data_lab.astype(np.int32)
        labfea, lab_groundtruth, unlabfea, unlab_groundtruth = select_dataset(x_, y_, seed =32, ratio = ratio)
        random_index = np.arange(labfea.shape[0])
        random_unlab = np.arange(unlabfea.shape[0])
        np.random.seed(seed)
        np.random.shuffle(random_index)
        np.random.shuffle(random_unlab)
        labfea = labfea[random_index]

        lab_groundtruth = lab_groundtruth[random_index]
        unlabfea, unlab_groundtruth = unlabfea[random_unlab], unlab_groundtruth[random_unlab]

        augumentation1 = augumentation()
        unlabfea_augview = DataTransform_TD(unlabfea, augumentation1)
        unlabfea_augview = unlabfea_augview.astype(np.float32)
        labfea_augview = DataTransform_TD(unlabfea, augumentation1)
        labfea_augview = labfea_augview.astype(np.float32)


        labfea, lab_groundtruth, unlabfea, unlab_groundtruth, unlabfea_augview, labfea_augview  = torch.tensor(labfea).to(
            device), torch.tensor(lab_groundtruth).to(device), \
            torch.tensor(unlabfea).to(device), torch.tensor(unlab_groundtruth).to(device), \
            torch.tensor(unlabfea_augview).to(device), torch.tensor(labfea_augview).to(device)


        with torch.no_grad():
            lab_predcit_logit1 = model.forward_pseudoass(labfea)  
            lab_predcit1 = torch.argmax(lab_predcit_logit1, dim=1)
        lab_pre1 = lab_predcit1.clone().cpu().detach().numpy()
        lab_gdtruth1 =lab_groundtruth.clone().cpu().detach().numpy()
        from utils import match_true_to_predicted
        align_lab = match_true_to_predicted(K_cluster,lab_gdtruth1,lab_pre1)
        align_lab = torch.tensor(align_lab).to(device)

        lab_batch_size = int(ratio * batch_size)
        unlab_batch_size = batch_size - lab_batch_size

        epo = -1
        print('pretrain_reuslt')
        eval_unlabel(unlabfea, unlab_groundtruth)

        if x_.shape[0] < 200:
            batch_size = x_.shape[0]
            lab_batch_size = int(ratio * batch_size)
            unlab_batch_size = batch_size - lab_batch_size
            criterion_instance = contrastive_loss.InstanceLoss(batch_size, instance_temperature, device).to(device)
            for epo in range(epochs):
                train_epoch(align_lab, labfea, labfea_augview, lab_groundtruth, unlabfea, unlabfea_augview, batch_size, lab_batch_size, unlab_batch_size)  

        else:
            for epo in range(epochs):
                train_epoch(align_lab, labfea, labfea_augview, lab_groundtruth, unlabfea, unlabfea_augview, batch_size, lab_batch_size, unlab_batch_size)

