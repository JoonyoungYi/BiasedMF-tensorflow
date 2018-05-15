import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv('data/training.csv')
data = np.array(df)

data[:, 0] -= 1
data = np.asfarray(data)

index = pd.read_csv('data/train_index.csv')
userId = data[:, 2]
movId = data[:, 0]
data[:, 2] = index['user_index']
data[:, 0] = index['mov_index']

NUM_ROW = data.shape[0]
np.random.shuffle(data)
split = int(0.8 * NUM_ROW)
train_data = data[:split]
test_data = data[split:-13]

NUM_TR_ROW = train_data.shape[0]
NUM_TS_ROW = test_data.shape[0]
print("Data preprocessing completed.")

print("Start here..")

num = 15

sess = tf.Session()

saver = tf.train.import_meta_graph('user/user-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./user'))

graph = tf.get_default_graph()
user_batch = graph.get_tensor_by_name("user_batch:0")
movie_batch = graph.get_tensor_by_name("movie_batch:0")
rating_batch = graph.get_tensor_by_name("rating_batch:0")
genres = pd.read_csv('data/genres.csv')
W_MOVIE = np.array(genres).astype(np.float32)

feed_dict = {
    user_batch: train_data[0:num, 2],
    movie_batch: train_data[0:num, 0],
    rating_batch: train_data[0:num, 1]
}

op_to_restore = graph.get_tensor_by_name("output:0")

predict = sess.run(op_to_restore, feed_dict)

print "Act ::", train_data[0:num, 1]
print "Pred::", np.around(predict * 2) / 2
