import random

import numpy as np



f = open('data/ml-100k/u.data', 'r')
full = np.zeros((943, 1682))
max_row = 0
for row, line in enumerate(f):
    cols = line.strip().split('\t')
    u = int(cols[0]) - 1
    i = int(cols[1]) - 1
    r = float(cols[2])
    full[u][i] = r
    max_row = row
f.close()
# np.save('ml-100k-full.npy', full)

f = open('data/ml-100k/u.data', 'r')
sparse = np.zeros((max_row + 1, 3))
for row, line in enumerate(f):
    cols = line.strip().split('\t')
    u = int(cols[0]) - 1
    i = int(cols[1]) - 1
    r = float(cols[2])
    sparse[row][0] = u
    sparse[row][1] = i
    sparse[row][2] = r
f.close()
# np.save('ml-100k-sparse.npy', sparse)

count = np.count_nonzero(full, axis=1)
print(np.mean(count))
print(np.std(count))
print(np.min(count))
max_count = np.max(count)
print(max_count)
# np.save('ml-100k-count.npy', count)

rating_dict = {}
for row_idx in range(sparse.shape[0]):
    user_id = sparse[row_idx][0]
    item_id = sparse[row_idx][1]
    rating = sparse[row_idx][2]

    ratings = rating_dict.get(user_id, [])
    ratings.append((item_id, rating))
    rating_dict[user_id] = ratings

train_rows = []
valid_rows = []
test_rows = []
for user_id, ratings in rating_dict.items():
    random.shuffle(ratings)
    # print(len(ratings))
    # print(ratings)
    for j, (item_id, rating) in enumerate(ratings):
        if j < 5:
            test_rows.append((user_id, item_id, rating))
        elif j < 7:
            valid_rows.append((user_id, item_id, rating))
        else:
            train_rows.append((user_id, item_id, rating))

np.save('ml-100k-train.npy', train_rows)
np.save('ml-100k-valid.npy', valid_rows)
np.save('ml-100k-test.npy', test_rows)

print(np.load('ml-100k-train.npy'))
