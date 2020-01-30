import math

import numpy as np

from app.utils.dataset import DatasetManager


def main(kind):
    full = np.load('data/{}/result-q.npy'.format(kind))
    print('>> np.std(full):', np.std(full))
    print('>> np.mean(full):', np.mean(full))

    user_num = full.shape[0]

    user_sim = np.ones((user_num, user_num))
    for i in range(user_num):
        for j in range(i + 1, user_num):
            i_vector = full[i] / np.mean(full[i])
            j_vector = full[j]  / np.mean(full[j])

            # print()
            # print(np.mean(j_vector))

            s = np.dot(i_vector, j_vector) / (
                np.linalg.norm(i_vector) * np.linalg.norm(j_vector))
            if not math.isnan(s) and math.isfinite(s):
                user_sim[j][i] = s
                user_sim[i][j] = s
            else:
                print(s)
                print(i_vector)
                print(j_vector)
                input('!!')

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
    # kind = DatasetManager.KIND_MOVIELENS_100K
    kind = DatasetManager.KIND_MOVIELENS_1M
    main(kind)
