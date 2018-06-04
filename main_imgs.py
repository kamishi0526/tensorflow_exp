from scipy.misc import imread, imresize
import pickle
import numpy as np

names = unpickle('./cifar-10-batches-py/batches.meta')['label_names']
data, labels = [], []
for i in range(1, 6):                                           # 6つのファイル分ループする
    filename = './cifar-10-batches-py/data_batch_' + str(i)
    batch_data = unpickle(filename)                             # ファイルを読み込みPythonの辞書を取得する
    if len(data) > 0:
        data = np.vstack((data, batch_data['data']))            # データサンプルの行は各サンプルを表しているので垂直に積み重ねる
        labels = np.hstack((labels, batch_data['labels']))      # ラベルは一次元なので横に積み重ねる
    else:
        data = batch_data['data']
        labels = batch_data['labels']

data = grayscale(data)                                          # CIFAR-10の画像をグレースケール化

def unpickle(file):                             # CIFAR-10を読み込み、読み込まれた辞書を返す
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')]
    fo.close()
    return dict

def grayscale(a):                               # グレースケール化
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)

def tmp():
    gray_image = imread(filepath, True)                 # 画像をグレースケールで読み込む
    small_gray_image = imresize(gray_image, 1. / 8.)    # 小さなものにリサイズ
    x = small_gray_image.flatten()                      # 一次元構造に変換する
