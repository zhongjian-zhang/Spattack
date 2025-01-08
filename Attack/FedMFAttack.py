import numpy as np
import torch
import torch.nn as nn


class signAtkClient(nn.Module):
    def __init__(self, train_ind, test_ind, m_item, args):
        super().__init__()
        self.args = args
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        self.m_item = m_item
        self.last_items_emb = None
        self.train_all = torch.Tensor(np.arange(self.m_item)).long()

        self.dim = args.dim
        self.items_emb_grad = None
        self.cnt = 0
        self.atk_start_epoch = args.atk_start_epoch if args.atk_start_epoch >= 0 else 0
        self.random_attack_flag = False

    def train_(self, benign_grads, scale):
        items_emb_grad = None
        if self.cnt >= self.atk_start_epoch:
            items_emb_grad = -scale[self.train_all] * benign_grads[self.train_all]
        self.cnt += 1
        loss = 0.0
        return self.train_all, items_emb_grad, loss

    def sample_items(self, max_item_count, items_popularity):
        items = np.random.choice(np.arange(self.m_item), max_item_count, replace=False, p=items_popularity)
        self.train_all = torch.Tensor(items).long()
        return items

    def eval_(self, items_emb):
        return None, None, None, None


class sameValueAtkClient(nn.Module):
    def __init__(self, train_ind, test_ind, m_item, args):
        super().__init__()
        self.args = args
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        self.m_item = m_item
        self.last_items_emb = None
        self.train_all = torch.Tensor(np.arange(self.m_item)).long()

        self.dim = args.dim
        self.items_emb_grad = None
        self.cnt = 0
        self.atk_start_epoch = args.atk_start_epoch if args.atk_start_epoch >= 0 else 0
        torch.manual_seed(0)
        self.items_emb_grad = torch.normal(0, 1, size=(self.m_item, self.dim), device=self.args.device)

    def train_(self, benign_grads, scale):
        items_emb_grad = None
        if self.cnt >= self.atk_start_epoch:
            items_emb_grad = self.items_emb_grad[self.train_all]
        self.cnt += 1
        loss = 0.0
        return self.train_all, items_emb_grad, loss

    def eval_(self, items_emb):
        return None, None, None, None

    def sample_items(self, max_item_count, items_popularity):
        items = np.random.choice(np.arange(self.m_item), max_item_count, replace=False, p=items_popularity)
        self.train_all = torch.Tensor(items).long()
        return items
