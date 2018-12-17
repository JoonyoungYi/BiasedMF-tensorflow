import math

import numpy as np

from app.utils.dataset import DatasetManager


def main(kind):
    full = np.load('data/{}/result-q.npy'.format(kind))
    user_num = full.shape[0]

    user_sim = np.ones((user_num, user_num))
    for i in range(user_num):
        for j in range(i + 1, user_num):
            i_vector = full[i]
            j_vector = full[j]
            s = np.dot(i_vector, j_vector) / (
                np.linalg.norm(i_vector) * np.linalg.norm(j_vector))
            if not math.isnan(s):
                user_sim[j][i] = s
                user_sim[i][j] = s
                continue

            i_vector = full[i] * 1e20
            j_vector = full[j] * 1e20
            s = np.dot(i_vector, j_vector) / (
                np.linalg.norm(i_vector) * np.linalg.norm(j_vector))
            if not math.isnan(s):
                user_sim[j][i] = s
                user_sim[i][j] = s
            else:
                user_sim[j][i] = -1
                user_sim[i][j] = -1

    # print(user_sim)
    sim_i = np.ones((user_num, user_num))
    sim_v = np.ones((user_num, user_num))
    for i in range(user_num):
        s_vector = user_sim[i]
        index = np.argsort(s_vector)
        sim_i[i] = index
        sim_v[i] = s_vector[index]

    np.save('data/{}/sim-item-i.npy'.format(kind), sim_i)
    np.save('data/{}/sim-item-v.npy'.format(kind), sim_v)

    # sim_i = np.load('sim-item-i.npy')
    # sim_v = np.load('sim-item-v.npy')

    for j in range(user_num):
        print('>>', j, np.min(sim_v[:, j]))


if __name__ == '__main__':
    kind = DatasetManager.KIND_MOVIELENS_100K
    main(kind)
