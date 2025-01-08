import torch
import torch.nn as nn
from pyspark import SparkConf, SparkContext
from Defense.denfense import *


class FedRecServer(nn.Module):
    def __init__(self, m_item, items_popularity, args):
        super().__init__()
        self.args = args
        self.m_item = m_item
        self.dim = args.dim
        self.items_emb = nn.Embedding(self.m_item, self.dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)
        self.atk_scale = None

        import os
        import sys
        os.environ['PYSPARK_PYTHON'] = sys.executable
        conf = SparkConf().setAppName("wordcount").setMaster("local[2]").set("spark.driver.memory", "100g")
        self.sc = SparkContext(conf=conf)
        self.sc.setLogLevel("ERROR")
        self.createCombiner = (lambda el: el)
        self.mergeVal = (lambda aggregated, el: aggregated + el)
        self.mergeComb = (lambda agg1, agg2: agg1 + agg2)
        self.items_popularity = items_popularity

    def train_(self, clients, malicious_clients_limit):
        batch_loss = []
        tupo_list = []
        benign_grads = torch.zeros_like(self.items_emb.weight)

        max_item_count = 0
        for idx in range(0, len(clients) - malicious_clients_limit):
            client = clients[idx]
            items, items_emb_grad, loss = client.train_(self.items_emb.weight)
            max_item_count = max(max_item_count, len(set(items.cpu().tolist())))
            if items_emb_grad is None:
                continue
            with torch.no_grad():
                if self.args.defense == 'Norm':
                    assert self.args.grad_limit > 0, "Positive grad_limit is needed!"
                    norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
                    too_large = norm[:, 0] > self.args.grad_limit
                    items_emb_grad[too_large] /= (norm[too_large] / self.args.grad_limit)
                zipped = zip(items.cpu().tolist(), items_emb_grad.cpu().tolist())
                tupo_list.extend(list(zipped))
                benign_grads[items] += items_emb_grad.clone().detach()
            if loss is not None:
                batch_loss.append(loss)

        if malicious_clients_limit > 0:
            if self.atk_scale is None:
                if self.args.sample_items is True:
                    self.atk_scale = self.collect_sample_items(clients, malicious_clients_limit, max_item_count)
                else:
                    self.atk_scale = torch.full((self.m_item, 1), 1.0 / malicious_clients_limit,
                                                dtype=torch.float32).to(self.args.device)
            for idx in range(len(clients) - malicious_clients_limit, len(clients)):
                client = clients[idx]
                items, items_emb_grad, loss = client.train_(benign_grads, self.atk_scale)
                if items_emb_grad is None:
                    continue
                with torch.no_grad():
                    if self.args.defense == 'Norm':
                        assert self.args.grad_limit > 0, "Positive gradient limit is needed!"
                        norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
                        too_large = norm[:, 0] > self.args.grad_limit
                        items_emb_grad[too_large] /= (norm[too_large] / self.args.grad_limit)
                    zipped = zip(items.tolist(), items_emb_grad.cpu().tolist())
                    tupo_list.extend(list(zipped))

        final_grads = self.aggregate_grad(tupo_list)
        with torch.no_grad():
            self.items_emb.weight.data.add_(torch.as_tensor(final_grads).to(self.args.device), alpha=-self.args.lr)
        return batch_loss

    def collect_sample_items(self, clients, malicious_clients_limit, max_item_count):
        malicious_item = np.zeros(self.m_item)
        for idx in range(len(clients) - malicious_clients_limit, len(clients)):
            client = clients[idx]
            items = client.sample_items(max_item_count, self.args.items_popularity)
            malicious_item[items] += 1
        scale = torch.tensor(1 / malicious_item, dtype=torch.float32).to(self.args.device).unsqueeze(1)
        scale_ = torch.where(torch.isinf(scale), torch.full_like(scale, 0), scale)
        return scale_

    def aggregate_grad(self, tupo_list):
        args_broad = self.sc.broadcast(self.args)
        inputdata = self.sc.parallelize(tupo_list)
        y = inputdata.combineByKey(self.createCombiner, self.mergeVal,
                                   self.mergeComb)

        if self.args.sample_items is True:
            assert self.args.defense in ["Mean", "Median", "Norm"]
        else:
            assert self.args.defense in ["Mean", "Median", "Norm", "Trimmean", "Krum"]

        if self.args.defense == 'Median':
            y = y.mapValues(lambda x: median_sc(x, args_broad.value)).collect()
        elif self.args.defense == 'Trimmean':
            y = y.mapValues(lambda x: trimmedmean_sc(x, args_broad.value)).collect()
        elif self.args.defense == 'Krum':
            y = y.mapValues(lambda x: krum_sc(x, args_broad.value)).collect()
        elif self.args.defense == 'Mean' or self.args.defense == 'Norm':
            y = y.mapValues(lambda x: mean_sc(x, args_broad.value)).collect()

        final_grads = np.zeros((self.m_item, self.dim))
        final_grads[list(dict(y).keys())] = list(dict(y).values())
        return final_grads

    def eval_(self, clients):
        hit_at_5_list = []
        hit_at_10_list = []
        ndcg_at_5_list = []
        ndcg_at_10_list = []
        for client in clients:
            hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10 = client.eval_(self.items_emb.weight)
            if hit_at_5 is not None:
                hit_at_5_list.append(hit_at_5)
                hit_at_10_list.append(hit_at_10)
                ndcg_at_5_list.append(ndcg_at_5)
                ndcg_at_10_list.append(ndcg_at_10)
        return 1.0 * sum(hit_at_5_list) / len(hit_at_5_list), 1.0 * sum(hit_at_10_list) / len(hit_at_10_list), \
               1.0 * sum(ndcg_at_5_list) / len(ndcg_at_5_list), 1.0 * sum(ndcg_at_10_list) / len(ndcg_at_10_list)
