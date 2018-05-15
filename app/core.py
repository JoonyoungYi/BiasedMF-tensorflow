import time
from collections import deque

import tensorflow as tf
import numpy as np
import pandas as pd

from .utils import dataset
from .configs import *
from .models import init_models

EPOCH_NUMBER = 10


def _train(session, models, train_data, valid_data):

    for epoch in range(EPOCH_NUMBER):
        _ = session.run(
            [models['train_op']],
            feed_dict={
                models['u']: train_data['user_id'],
                models['i']: train_data['item_id'],
                models['r']: train_data['rating'],
            })
        # pred_batch = np.clip(pred_batch, 1.0, 5.0)

        _ = session.run(
            [models['r_ui_hat']],
            feed_dict={
                models['u']: valid_data['user_id'],
                models['i']: valid_data['item_id'],
                models['r']: valid_data['rating'],
            })

        print('hi!')
        break

    # print("Computing Final Test Loss...")
    #
    # bloss = 0
    # for xx in range(num_batch_loop):
    #     pred_batch = prediction.eval({
    #         user_batch:
    #         test_data[xx * TS_BATCH_SIZE:(xx + 1) * TS_BATCH_SIZE, 0],
    #         movie_batch:
    #         test_data[xx * TS_BATCH_SIZE:(xx + 1) * TS_BATCH_SIZE, 1]
    #     })
    #     pred_batch = np.clip(pred_batch, 1.0, 5.0)
    #     bloss += np.mean(
    #         np.power(pred_batch - test_data[xx * TS_BATCH_SIZE:
    #                                         (xx + 1) * TS_BATCH_SIZE, 2], 2))
    #     if (xx + 1) % 50 == 0:
    #         per = float(xx + 1) / (num_batch_loop) * 100
    #         print(str(per) + "% Completed")
    # test_loss = np.sqrt(bloss / num_batch_loop)
    # print("Test Loss:" + str(round(test_loss, 3)))
    #
    # RMSEtr[0] = RMSEts[0]
    # saver.save(sess, 'gen-model')
    # print("Awesome !!")


def main():
    kind = dataset.ML_100K

    K = 5
    lambda_value = 10

    data = dataset.load_data(kind)
    N, M = dataset.get_N_and_M(kind)
    models = init_models(N, M, K, lambda_value)

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        _train(session, models, data['train'], data['valid'])
