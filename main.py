import os
import torch
import random
import numpy as np
from time import time
from data import load_dataset
from Model.server import FedRecServer
from Model.client import FedRecClient
from parse import parse_args, update_parser_data


def main():
    args = parse_args()
    setup_seed(args.times)

    experiment = f"{args.attack}_{args.sample_items}_{args.clients_limit}_{args.atk_start_epoch}_{args.defense}_{args.dataset.strip('/')}_{args.times}"
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)
    m_item, all_train_ind, all_test_ind, items_popularity = load_dataset(args.path + args.dataset)
    args = update_parser_data(args, m_item, all_train_ind, all_test_ind, items_popularity)

    t0 = time()
    server = FedRecServer(m_item, items_popularity, args).to(args.device)
    clients = [FedRecClient(train_ind, test_ind, m_item, args).to(args.device) for train_ind, test_ind in
               zip(all_train_ind, all_test_ind)]
    malicious_clients_limit = int(len(clients) * args.clients_limit)
    assert args.attack in ["Spattack_O", "Spattack_L"]
    attack_client = getattr(__import__("Attack"), args.attack)
    print(f"Benign clients number: {len(clients)}")
    clients.extend(attack_client(None, [], m_item, args).to(args.device) for _ in range(malicious_clients_limit))
    print(f"All clients number: {len(clients)}")

    log_dir = f"./Log/{args.attack}/{args.sample_items}/"
    os.makedirs(log_dir, exist_ok=True)
    f = open(f'{log_dir}{experiment}.txt', 'w')
    f.write("Arguments: %s " % args_str + "\n")
    f.write("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (
        time() - t0, len(clients), m_item, sum([len(i) for i in all_train_ind]),
        sum([len(i) for i in all_test_ind])) + "\n")

    with torch.no_grad():
        hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10 = server.eval_(clients)
        print(
            "Iteration:0(init),hit_5:%.4f" % hit_at_5 + ", hit_10:%.4f" % hit_at_10 + ", ndcg_5:%.4f" % ndcg_at_5 + ", ndcg_10:%.4f" % ndcg_at_10)

    try:
        server.scores = np.array([0.0] * len(clients))
        for epoch in range(1, args.epochs + 1):
            losses = server.train_(clients, malicious_clients_limit)
            loss = np.mean(np.array(losses)).item()

            with torch.no_grad():
                hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10 = server.eval_(clients)
                record_log(epoch, hit_at_10, hit_at_5, loss, ndcg_at_10, ndcg_at_5, file=f)

    except KeyboardInterrupt:
        pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def record_log(epoch, hit_at_10, hit_at_5, loss, ndcg_at_10, ndcg_at_5, file=None):
    if file is not None:
        file.write(
            "Iteration:%d, loss:%.5f" % (epoch,
                                         loss) + ",hit_5:%.4f" % hit_at_5 + ", hit_10:%.4f" % hit_at_10 + ", ndcg_5:%.4f" % ndcg_at_5 + ",ndcg_10:%.4f" % ndcg_at_10 + "\n")
    print("Iteration:%d, loss:%.5f" % (epoch,
                                       loss) + ",hit_5:%.4f" % hit_at_5 + ", hit_10:%.4f" % hit_at_10 + ", ndcg_5:%.4f" % ndcg_at_5 + ",ndcg_10:%.4f" % ndcg_at_10)


if __name__ == "__main__":
    main()
