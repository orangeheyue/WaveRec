# coding: utf-8
# rongqing001@e.ntu.edu.sg
r"""
SMORE - Multi-modal Recommender System
Reference:
    ACM WSDM 2025: Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation

Reference Code:
    https://github.com/kennethorq/SMORE
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import pywt
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class SMORE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMORE, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda() 

        self.fusion_adj = self.max_pool_fusion()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        

    def pre_epoch_processing(self):
        pass

    def max_pool_fusion(self):
        image_adj = self.image_original_adj.coalesce()
        text_adj = self.text_original_adj.coalesce()

        image_indices = image_adj.indices().to(self.device)
        image_values = image_adj.values().to(self.device)
        text_indices = text_adj.indices().to(self.device)
        text_values = text_adj.values().to(self.device)

        combined_indices = torch.cat((image_indices, text_indices), dim=1)
        combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

        combined_values_image = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
        combined_values_text = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)

        combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
        combined_values_text[unique_idx[image_indices.size(1):]] = text_values
        combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

        fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()

        return fusion_adj

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # def spectrum_convolution(self, image_embeds, text_embeds):
    #     """
    #     Modality Denoising & Cross-Modality Fusion
    #     """
    #     image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')           
    #     text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

    #     image_complex_weight = torch.view_as_complex(self.image_complex_weight)   
    #     text_complex_weight = torch.view_as_complex(self.text_complex_weight)
    #     fusion_complex_weight = torch.view_as_complex(self.fusion_complex_weight)

    #     #   Uni-modal Denoising
    #     image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')    
    #     text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

    #     #   Cross-modality fusion
    #     fusion_conv = torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho') 
    #     # print("image_conv.shape:", image_conv.shape, "text_conv.shape:", text_conv.shape, "fusion_conv.shape:", fusion_conv.shape)
    #     return image_conv, text_conv, fusion_conv

    def spectrum_convolution(self, image_embedding, text_embedding):
        '''
        Desc: 多模态多尺度小波变换对齐
        Args:
            image_embedding: 图像embedding, image_embedding.shape: torch.Size([7050, 64]) 
            text_embedding: 文本embedding, text_embedding.shape: torch.Size([7050, 64]), 7050个物品,每个物品编码为64维向量
        Function:
            对图像和文本嵌入进行多模态多尺度小波变换对齐：
            Steps1: 选取模态领域适应的合适小波基['harr', 'db1-20', 'bior1.3'], 选择高通滤波器的下采样分级尺度level,本文默认选择level3, 注意输入的特征向量是一维度，因此暂时不考虑水平、垂直、对角。
            Steps2: 获取图像模态特征的image_coeffs（一个低通分量，三个高通分量）；文本模态特征的text_coeffs（一个低通分量，三个高通分量）
            Steps3: 多模态多尺度频域空间对齐：
                    3.1低频信号对齐
                    1.对于低频信号（第 3 级水平低频分量），首先对图像模态和文本模态下的低频信号分别进行LoRA(奇异值分解（SVD）),去除冗余信息，保留主要特征。 
                    2.然后,再分别对进行归一化，对图像低频信号，可将其像素值归一化到特定范围，如 [0, 1] 或 [-1, 1]，以消除不同图像之间的亮度差异等影响。 对于文本低频信号，可对词向量或特征向量进行归一化，使不同文本的特征具有可比性，例如采用 L2 归一化。
                    3.然后再将LoRa和归一化后的两个模态的特征向量进行对齐， 生成一个新的融合模态第 3 级水平低频分量
                    3.高频信号多尺度对齐
                        1.对图像和文本的第 3 级水平高频分量进行对齐，生成一个新的融合模态第 3 级水平高频分量
                        2.对图像和文本的第 2 级水平高频分量进行对齐， 生成一个新的融合模态第 2 级水平高频分量
                        3.对图像和文本的第 1 级水平高频分量分别先进行去噪，再进行对齐， 生成一个新的融合模态第 1 级水平高频分量

        Returns:
            image_embedding_wave.shape: torch.Size([7050, 64]) text_embedding_wave.shape: torch.Size([7050, 64]) fusion_wave.shape: torch.Size([7050, 64])
        '''
        # 转换为numpy数组进行小波变换
        image_np = image_embedding.detach().cpu().numpy()
        text_np = text_embedding.detach().cpu().numpy()

        # 定义小波类型
        wavelet = 'db4'
        level = 3 

        # 对图像和文本嵌入进行小波变换
        '''
        原始信号 如果输入是 (256, 256) 的图像，axis=1（水平方向分解），level=3：
│
        ├── cA3（最低频，最模糊的近似）      主体信号，轮廓 （保留）  (256, 32)  第 3 级水平低频分量
        │
        ├── cD3（第3级细节，较大尺度的高频） 大尺度边缘 最精细的细节信息 (256, 32) 第 3 级水平高频分量
        │
        ├── cD2（第2级细节，中等尺度的高频） 中尺度细节，更细微的边缘和纹理变化 (256, 64) 第 2 级水平高频分量
        │
        └── cD1（第1级细节，最细尺度的高频） 高频噪声->去噪 (256, 128) 第 1 级水平高频分量
        '''
        image_coeffs = pywt.wavedec(image_np, wavelet, level=level, axis=1)
        text_coeffs = pywt.wavedec(text_np, wavelet, level=level, axis=1)

        # len(image_coeffs): 4 len(text_coeffs): 4
        # print("len(image_coeffs):", len(image_coeffs), "len(text_coeffs):", len(text_coeffs))
        # 对每一级系数进行对齐和融合
        fused_coeffs = []
        for i in range(len(image_coeffs)):
            # 简单的平均融合
            fused_coeff = (image_coeffs[i] + text_coeffs[i]) / 2
            fused_coeffs.append(fused_coeff)

        # 进行小波逆变换
        image_embedding_wave_np = pywt.waverec(image_coeffs, wavelet, axis=1)
        text_embedding_wave_np = pywt.waverec(text_coeffs, wavelet, axis=1)
        fusion_wave_np = pywt.waverec(fused_coeffs, wavelet, axis=1)

        # 转换回torch张量
        image_embedding_wave = torch.tensor(image_embedding_wave_np, dtype=torch.float32, device=image_embedding.device)
        text_embedding_wave = torch.tensor(text_embedding_wave_np, dtype=torch.float32, device=text_embedding.device)
        fusion_wave = torch.tensor(fusion_wave_np, dtype=torch.float32, device=image_embedding.device)

        return image_embedding_wave, text_embedding_wave, fusion_wave

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        
        #   Spectrum Modality Fusion
        image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
        fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

        #   User-Item (Behavioral) View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        #   Item-Item Modality Specific and Fusion views
        #   Image-view
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        #   Text-view
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #   Fusion-view
        if self.sparse:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        #   Modality-aware Preference Module
        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        fusion_soft_v = self.softmax(fusion_att_v)
        agg_image_embeds = fusion_soft_v * image_embeds

        fusion_soft_t = self.softmax(fusion_att_t)
        agg_text_embeds = fusion_soft_t * text_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        fusion_prefer = self.gate_fusion_prefer(content_embeds)
        image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)

        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
        fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    

    