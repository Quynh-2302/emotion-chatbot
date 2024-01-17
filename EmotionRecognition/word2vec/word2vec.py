#! /usr/bin/env python3


import collections
import logging
import os
import tarfile
import tensorflow as tf
import tensorlayer as tl

tf.compat.v1.disable_eager_execution()

# đọc dữ liệu từ tệp ../data/ijcnlp_dailydialog/dialogues_text.txt
# và trả về một danh sách các từ


def load_dataset():
    files = ['../data/ijcnlp_dailydialog/dialogues_text.txt']
    words = []
    for file in files:
        f = open(file, encoding='utf-8')
        for dialog in f:
            lines = dialog.split('__eou__')
            for line in lines:
                for word in line.strip().split(' '):
                    if word != '':
                        words.append(word)
        f.close()
    return words

# min_freq = 3: 3 là tần suất tối thiểu mà một từ phải xuất hiện để được tính vào kích thước của từ vựng


def get_vocabulary_size(words, min_freq=3):
    size = 1
    # đếm số lần xuất hiện của mỗi từ trong danh sách words và
    # sắp xếp các cặp (từ, số lần xuất hiện) theo thứ tự giảm dần của số lần xuất hiện
    counts = collections.Counter(words).most_common()
    for word, c in counts:
        if c >= min_freq:
            size += 1
    return size

# đảm bảo rằng thư mục chứa checkpoint file đã tồn tại trước khi lưu trạng thái của mô hình vào file đó


def save_checkpoint(ckpt_file_path):
    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tf.compat.v1.train.Saver().save(sess, ckpt_file_path + '.ckpt')

# trước khi tiếp tục huấn luyện hoặc sử dụng mô hình đã được lưu trạng thái,
# kiểm tra checkpoint file đã tồn tại và có đầy đủ thông tin cần thiết không


def load_checkpoint(ckpt_file_path):
    ckpt = ckpt_file_path + '.ckpt'
    index = ckpt + ".index"
    meta = ckpt + ".meta"
    # Nếu cả hai file tồn tại (index, meta),
    # khôi phục trạng thái của mô hình từ checkpoint file (ckpt) vào phiên TensorFlow sess
    if os.path.isfile(index) and os.path.isfile(meta):
        tf.compat.v1.train.Saver().restore(sess, ckpt)


def save_embedding(dictionary, network, embedding_file_path):
    words, ids = zip(*dictionary.items())
    params = network.normalized_embeddings
    embeddings = tf.nn.embedding_lookup(
        params, tf.constant(ids, dtype=tf.int32)).eval()
    wv = dict(zip(words, embeddings))
    path = os.path.dirname(os.path.abspath(embedding_file_path))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tl.files.save_any_to_npy(save_dict=wv, name=embedding_file_path + '.npy')


def train(model_name):
    # Nạp dữ liệu huấn luyện từ tập tin dialogues_text.txt
    words = load_dataset()
    data_size = len(words)  # 140161
    vocabulary_size = get_vocabulary_size(words, min_freq=3)
    batch_size = 500
    embedding_size = 200
    skip_window = 5
    num_skips = 10
    num_sampled = 64
    learning_rate = 0.1
    n_epoch = 50
    # hàm train thực hiện num_steps thì dừng
    num_steps = int((data_size / batch_size) * n_epoch)  # 1400

    data, count, dictionary, reverse_dictionary = \
        tl.nlp.build_words_dataset(words, vocabulary_size)
    train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])

    with tf.device('/cpu:0'):
        emb_net = tl.layers.Word2vecEmbeddingInputlayer(
            inputs=train_inputs,
            train_labels=train_labels,
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            num_sampled=num_sampled)
        # Định nghĩa hàm loss và thuật toán tối ưu hoá (Gradient Descent)
        loss = emb_net.nce_cost
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate).minimize(loss)

    sess.run(tf.compat.v1.global_variables_initializer())
    # model_name = model_word2vec_200
    ckpt_file_path = "checkpoint/" + model_name
    load_checkpoint(ckpt_file_path)

    step = data_index = 0
    loss_vals = []
    while step < num_steps:
        batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(
            data=data, batch_size=batch_size, num_skips=num_skips,
            skip_window=skip_window, data_index=data_index)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        loss_vals.append(loss_val)
        if (step != 0) and (step % 200) == 0:
            logging.info("(%d/%d) latest average loss: %f.", step,
                         num_steps, sum(loss_vals) / len(loss_vals))
            del loss_vals[:]
            # Lưu checkpoint
            save_checkpoint(ckpt_file_path)
            embedding_file_path = "../data/output/" + model_name
            # Lưu embedding
            save_embedding(dictionary, emb_net, embedding_file_path)
        step += 1


if __name__ == '__main__':
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    sess = tf.compat.v1.InteractiveSession()
    train('model_word2vec_200')
    sess.close()
