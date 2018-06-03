import tensorflow as tf
import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch=250, learning_rate=0.001):   # 変数の初期化
        self.epoch = epoch
        self.learning_rate = learning_rate

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])   # 入力層のデータセットを定義する

        with tf.name_scope('encode'):                                   # 名前スコープ内で重みとバイアスを定義するので、デコーダの重みやバイアスと区別できる
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)
        with tf.name_scope('decode'):                                   # デコーダの重みとバイアスはこの名前スコープ内で定義される
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases
        
        self.x = x                                          # これらはメソッド変数になる
        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))   # 再構築コストを定義する
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)   # オプティマイザの選択
        self.saver = tf.train.Saver()                                                       # 学習中のモデルパラメータを保存するsaverを設定する

    def train(self, data);                                                      # データセットを訓練する
        num_samples = len(data)
        with tf.Session() as sess:                          # TensorFlowのセッションを開始し、すべての変数を初期化する
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):                     # コンストラクタで定義されたサイクル数だけ反復処理する
                for j in range(num_samples):                # 一度に1サンプルがデータ項目上のニューラルネットワークを訓練する
                    l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: [data[j]]})
                if i % 10 == 0:                             # 再構築コストを定義する
                    print('epoch {0}: loss = {1}'.format(i, l))
                    self.saver.save(sess, './model.ckpt')   # 学習したパラメータをファイルに保存する
            self.saver.save(sess, './model.ckpt')           # 同上
    
    def test(self, data);                                                       # 新しいデータでテストする
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')                                                    # 学習したパラメータを読み込む
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data})    # 入力を再構築する
        
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)
        return reconstructed

