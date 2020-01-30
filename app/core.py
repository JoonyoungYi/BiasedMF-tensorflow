import os
import time
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .utils.batch import BatchManager
from .configs import *
from .models import init_models

EPOCH_NUMBER = 10000
EARLY_STOP = True
EARLY_STOP_MAX_ITER = 100


def _train(session, saver, kind, models, batch_manager):
    model_file_path = _init_model_file_path(kind)
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0

    for epoch in range(EPOCH_NUMBER):
        _, train_rmse = session.run(
            [models['train_op'], models['rmse']],
            feed_dict={
                models['u']: batch_manager.train_data[:, 0],
                models['i']: batch_manager.train_data[:, 1],
                models['r']: batch_manager.train_data[:, 2],
            })

        _, valid_rmse = session.run(
            [models['loss'], models['rmse']],
            feed_dict={
                models['u']: batch_manager.valid_data[:, 0],
                models['i']: batch_manager.valid_data[:, 1],
                models['r']: batch_manager.valid_data[:, 2],
            })
        if epoch % 10 == 0:
            print('>> EPOCH:', "{:3d}".format(epoch), "{:3f}, {:3f}".format(
                train_rmse, valid_rmse))

        if EARLY_STOP:
            early_stop_iters += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_iters = 0
                saver.save(session, model_file_path)
            elif early_stop_iters >= EARLY_STOP_MAX_ITER:
                print("Early stopping ({} vs. {})...".format(
                    prev_valid_rmse, valid_rmse))
                break
        else:
            saver.save(session, model_file_path)

    return model_file_path


def _test(session, models, batch_manager):
    valid_rmse = session.run(
        [models['rmse']],
        feed_dict={
            models['u']: batch_manager.valid_data[:, 0],
            models['i']: batch_manager.valid_data[:, 1],
            models['r']: batch_manager.valid_data[:, 2],
        })[0]

    test_rmse = session.run(
        [models['rmse']],
        feed_dict={
            models['u']: batch_manager.test_data[:, 0],
            models['i']: batch_manager.test_data[:, 1],
            models['r']: batch_manager.test_data[:, 2],
        })[0]
    print("Final valid RMSE: {}, test RMSE: {}".format(valid_rmse, test_rmse))

    return valid_rmse, test_rmse


def _init_model_file_path(kind):
    folder_path = 'logs/{}'.format(int(time.time() * 1000))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return os.path.join(folder_path, 'model.ckpt')


# def __add_count_into_histogram(histogram, rmse):
#     key = int(rmse * 10)
#     count = histogram.get(key, 0)
#     histogram[key] = count + 1
#     return histogram
#
#
# def __save_plot(histogram):
#     for key, item in sorted(histogram.items(), key=lambda x: x[0]):
#         print('--', key, item)
#     plt.plot([i / 10
#               for i in range(40)], [histogram.get(i, 0) for i in range(40)])
#
#     plt.savefig('histogram.png')
#     plt.clf()


def main(kind, K=7, lambda_value=10):
    batch_manager = BatchManager(kind)
    models = init_models(K, lambda_value, batch_manager)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.02)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        session.run(tf.global_variables_initializer())
        model_file_path = _train(session, saver, kind, models, batch_manager)

        print('Loading best checkpointed model:', model_file_path)
        saver.restore(session, model_file_path)
        valid_rmse, test_rmse = _test(session, models, batch_manager)

        # save learning results
        p, q, b_u, b_i = session.run((models['p'], models['q'], models['b_u'],
                                      models['b_i']))
        np.save('data/{}/result-p.npy'.format(kind), p)
        np.save('data/{}/result-q.npy'.format(kind), q)
        np.save('data/{}/result-b-u.npy'.format(kind), b_u)
        np.save('data/{}/result-b-i.npy'.format(kind), b_i)

        # plot histogram
        # histogram_manager = HistogramManager()
        # histogram_manager.set_data(batch_manager.)
        return valid_rmse, test_rmse


# train_data = np.load('ml-100k-train.npy')
# valid_data = np.load('ml-100k-valid.npy')
# test_data = np.load('ml-100k-test.npy')
#
# N, M = 943, 1682
# mu = np.mean(train_data[:, 2])
#
# p = np.load('p.npy')
# q = np.load('q.npy')
# b_u = np.load('b_u.npy')
# b_i = np.load('b_i.npy')
# m = np.matmul(p, np.transpose(q)) + np.matmul(
#     np.reshape(b_u, (-1, 1)), np.ones(
#         (1, M))) + np.matmul(np.ones((N, 1)), np.reshape(b_i,
#                                                          (1, -1))) + mu
# print(m)
#
# histogram = {}
# # for _data in [train_data, valid_data, test_data]:
# for _data in [test_data]:
#     error, count = 0, 0
#     error_dict = {}
#     for i in range(_data.shape[0]):
#         user_id = int(_data[i][0])
#         item_id = int(_data[i][1])
#         rating = _data[i][2]
#
#         delta = rating - m[user_id][item_id]
#         # print(rating, m[user_id][item_id])
#         # print(delta)
#         # print(delta)
#         error += (delta * delta)
#         count += 1
#
#         errors = error_dict.get(user_id, [])
#         errors.append(delta * delta)
#         error_dict[user_id] = errors
#
#     for user_id in range(N):
#         rmse = np.mean(error_dict[user_id])
#         # print(rmse)
#         __add_count_into_histogram(histogram, rmse)
#     # print(error_dict[user_id])
#     print(math.sqrt(error / count))
#
# __save_plot(histogram)
#
# for _data in [train_data, valid_data, test_data]:
#     for i in range(_data.shape[0]):
#         user_id = int(_data[i][0])
#         item_id = int(_data[i][1])
#         rating = _data[i][2]
#         m[user_id][item_id] = rating
#
# np.save('ml-100k-biased-mf-full-real.npy', m)
# #
# assert False

# print(mu)
