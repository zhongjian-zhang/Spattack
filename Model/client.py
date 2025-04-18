import numpy as np
import torch
import torch.nn as nn
from eval import evaluate_recall, evaluate_ndcg


class FedRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, m_item, args):
        super().__init__()
        self.args = args
        self._train_ = train_ind
        self._test_ = test_ind
        self.m_item = m_item
        self._train = torch.Tensor(train_ind).long()
        self._test = torch.Tensor(test_ind).long()

        self.train_neg_ind = []
        for _ in train_ind:
            neg_item = np.random.randint(m_item)
            while neg_item in train_ind:
                neg_item = np.random.randint(m_item)
            self.train_neg_ind.append(neg_item)

        train_all = train_ind + self.train_neg_ind
        train_all.sort()
        self.train_all = torch.Tensor(train_all).long()

        d = {idx: train_all_idx for train_all_idx, idx in
             enumerate(train_all)}
        self._train_pos = torch.Tensor([d[i] for i in train_ind]).long()
        self._train_neg = torch.Tensor([d[i] for i in self.train_neg_ind]).long()
        self.train_ind = torch.Tensor(train_ind).long()
        self.train_neg_ind = torch.Tensor(self.train_neg_ind).long()

        self.dim = args.dim
        self.items_emb_grad = None
        self._user_emb = nn.Embedding(1, self.dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)

    def forward(self, items_emb):
        scores = torch.sum(self._user_emb.weight * items_emb, dim=-1)
        return scores

    def train_(self, items_emb, reg=0.1):
        items_emb = items_emb[self.train_all].clone().detach().requires_grad_(True)
        self._user_emb.zero_grad()
        pos_items_emb = items_emb[self._train_pos]
        neg_items_emb = items_emb[self._train_neg]
        pos_scores = self.forward(pos_items_emb)
        neg_scores = self.forward(neg_items_emb)
        if self.args.l2_reg:
            loss = -(pos_scores - neg_scores).sigmoid().log().sum() + 0.5 * (
                    self._user_emb.weight.norm(2).pow(2) + items_emb.norm(2).pow(2)) * reg
        else:
            loss = -(pos_scores - neg_scores).sigmoid().log().sum()
        loss.backward()

        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-self.args.lr)
        self.items_emb_grad = items_emb.grad

        return self.train_all, self.items_emb_grad, loss.cpu().item()

    def eval_(self, items_emb):
        rating = self.forward(items_emb)
        items = []
        for _ in range(99):
            neg_item = np.random.randint(self.m_item)
            while neg_item in self._train_ or neg_item in items:
                neg_item = np.random.randint(self.m_item)
            items.append(neg_item)
        items.append(self._test_[0])
        items = torch.Tensor(items).long().to(self.args.device)

        rating[self._train] = - (1 << 10)
        hit_at_5 = evaluate_recall(rating[items], [99], 5)
        hit_at_10 = evaluate_recall(rating[items], [99], 10)
        ndcg_at_5 = evaluate_ndcg(rating[items], [99], 5)
        ndcg_at_10 = evaluate_ndcg(rating[items], [99], 10)
        return hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10
