import argparse
import numpy as np
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--dim', type=int, default=32, help='Dim of latent vectors.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k/', help='Choose a dataset.',
                        choices=["ml-100k/", "ml-1m/", "steam/"])
    parser.add_argument('--device', type=int, default=3 if cuda.is_available() else 'cpu',
                        help='Which device to run the Model.')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--l2_reg', type=bool, default=True, help='L2 norm regularization in loss.')
    parser.add_argument('--sample_items', type=bool, default=True, help='Whether sample attacked items in malicious client.')

    parser.add_argument('--grad_limit', type=float, default=1.0, help='Limit of l2-norm of item gradients.')
    parser.add_argument('--clients_limit', type=float, default=0.010101,
                        help='Limit of proportion of malicious clients.',
                        choices=[0.010101, 0.052631, 0.111111, 0.176470])
    parser.add_argument('--atk_start_epoch', type=int, default=0, help='Epoch starting attack.')
    parser.add_argument('--defense', nargs='?', default='Mean', help='Defense baselines.',
                        choices=["Mean", "Median", "Norm", "Trimmean", "Krum"])
    parser.add_argument('--times', type=int, default=0, help='random seed')
    parser.add_argument('--attack', type=str, default="signAtkClient", help='Attack method',
                        choices=['signAtkClient', 'sameValueAtkClient'])

    return parser.parse_args()


def update_parser_data(args, m_item, all_train_ind, all_test_ind, items_popularity):
    data = vars(args)
    data['m_item'] = m_item
    data['all_train_ind'] = all_train_ind
    data['all_test_ind'] = all_test_ind
    data['items_popularity'] = items_popularity / np.sum(items_popularity)
    data['m_cln_client'] = len(all_test_ind)
    data['clients_popularity'] = np.array([len(x) for x in all_train_ind])
    data['max_degree'] = int(items_popularity.max())
    parser = argparse.Namespace(**data)
    return parser
