import random

from app.core import main
from app.utils.dataset import DatasetManager

if __name__ == '__main__':
    kind = DatasetManager.KIND_MOVIELENS_100K
    # kind = DatasetManager.KIND_MOVIELENS_1M
    # kind = DatasetManager.KIND_MOVIELENS_10M

    hyper_params = {
        'K': 7,
        'lambda_value': 10,
    }
    valid_rmse, test_rmse = main(kind, **hyper_params)

    print('\t'.join(sorted(hyper_params.keys())))
    msg = '{}\t{}\t{}'.format('\t'.join(
        str(hyper_params[key])
        for key in sorted(hyper_params.keys())), valid_rmse, test_rmse)
    print(msg)

    # with open('results/ml-100k.txt', 'a') as f:
    #     f.write(msg + '\n')
