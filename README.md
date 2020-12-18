# meetup SageMaker で画像分類

目次

1. Notebookインスタンスの作成
1. 学習～推論結果の確認
1. Notebookインスタンスの削除


## Notebookの作成

学習を実行するためのノートブックインスタンスを作成します。

1. AWSマネジメントコンソールにログイン

   ![img001](https://user-images.githubusercontent.com/66664167/102601915-24828280-4164-11eb-8573-ed0f855d32d3.PNG)

1. SageMakerを検索

   ![img002](https://user-images.githubusercontent.com/66664167/102601963-33693500-4164-11eb-9ab9-e96bf5de79ba.PNG)

1. 左メニュー > ノートブック > ノートブックインスタンス

   ![img003](https://user-images.githubusercontent.com/66664167/102601994-3d8b3380-4164-11eb-886b-67317f857cb6.PNG)

1. ノートブックインスタンスの作成

   ![img004](https://user-images.githubusercontent.com/66664167/102602022-44b24180-4164-11eb-996a-1a67d6947304.PNG)

| key                              | val                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------- |
| ノートブックインスタンス名       | meetup-sagemaker-sample                                                         |
| ノートブックインスタンスのタイプ | ml.t2.medium                                                                    |
| Elastic Inference                | なし                                                                            |
| IAM ロール                       | 新しいロールの作成 > ロールの作成                                               |
| ルートアクセス                   | 有効化                                                                          |

1. インスタンスのステータスが InService になるまで待つ

1. Jupyterを開く から JupyterNotebook にアクセスする

   ![img005](https://user-images.githubusercontent.com/66664167/102602039-4f6cd680-4164-11eb-86b6-ab84ea53d4c1.PNG)

1. New▼ > conda_python3

   ![img006](https://user-images.githubusercontent.com/66664167/102602063-57c51180-4164-11eb-9f73-26100e20fa59.PNG)

これで実行環境の用意が完了しました。


## 学習～推論

![ミートアップ_本日のスコープ](https://user-images.githubusercontent.com/66664167/102602098-66abc400-4164-11eb-8d4a-613912794cb1.jpg)

今回は `Fashion_MNIST` という衣服の画像集からどの形状の衣服であるかを予測するモデルを作成します。

まずはじめに機械学習のライブラリである `TensorFlow` をノートブックインスタンスにインストールします。
また、学習データや結果を描画するために `Matplotlib` も入れておきます。
```
!pip install tensorflow
!pip install matplotlib
```

---

インストールした `TensorFlow` に含まれる `Keras` をインポートします。
`Keras` は `TensorFlow` を簡単に扱えるようにしたライブラリです。
あわせてデータの加工を行う `NumPy` と先程インストールした `Matplotlib` から `pyplog` をインポートします。
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
```

---

次にデータの読み込みを行います。
簡単なテストや学習用にしばしば利用される `MNIST` のデータのうち、衣服画像をあつめた `Fashion_MNIST` を用意します。
訓練用の画像と正解ラベル、テスト用の画像と正解ラベルに分割します。
```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

---

正解ラベルは 0～9 までの数字で表現されていて何を意味するかがわかりにくいので正解ラベルの名称を定義します。
| Label | Name        |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |
```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

---

訓練用データセットの構造を確認します。
```python
train_images.shape
```

`60,000` は画像の枚数で、 `28` はこの画像の縦横それぞれのピクセル数を表します。
つまり、トレーニング用に 28 x 28 の画像が 60,000 枚用意されているということになります。

---

正解ラベルの内容を確認します。
```python
train_labels
```
1つ目の画像は `9` つまり `Ankle boot` であることがわかります。
続いて `T-shirt`, `T-shirt`, ... 最後は `Sandal` の画像ですね。

---

訓練データの1件目を確認してみましょう。
```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```
`Ankle boot` の画像ですが、1ピクセルあたり 0～255 の色情報を持っています。
学習させるデータは数値の大きさの影響を受けたくないため、スケールさせてあげることが多いです。
今回は色情報を 0～1 の間にスケールさせて白黒画像として学習させてあげましょう。

---

各ピクセルの色情報を 255 で割ることで 0～1 の間にスケールします。
```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

---

いくつかスケールさせた画像を確認してみましょう。
```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

---

テスト用データはこちら。
```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
plt.show()
```

---

ここからは訓練データを使用して推論を行うモデルを作成しましょう。
まずは学習の進め方の設定を行います。
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

`keras.layers.Flatten` では 28x28 の2次元画像データを1次元の配列に変換します。

`keras.layers.Dense` ではニューラルネットワークの最初の層と最後の層を定義しています。
最初の層では 128個のニューロンを使って `relu`(ReLU関数) を計算して学習をします。
最後の層では `softmax` 関数を使って 10種類の分類を行っています。
この `softmax` 関数でそれぞれの衣服である確率を求めて一番可能性が高いものを推論結果として選びます。

---

先程設定した内容を踏まえてモデルのコンパイルを行います。
```python
model.compile( loss='sparse_categorical_crossentropy',
               optimizer='adam', 
               metrics=['accuracy'])
```

`loss='sparse_categorical_crossentropy'` では訓練データと推論内容の差を計算する損失関数を設定します。
この損失関数を小さくするように学習が進められます。

`optimizer='adam'` では損失関数を小さくするアルゴリズムを指定します。
`adam` というアルゴリズムが広く利用されています。

`metrics=['accuracy']` では学習の進捗具合を正しく分類ができた比率として計測する設定です。

---

`fit` で実際に学習を進めます。
`epochs` は用意した訓練データを何周するかを指定します。
今回の場合は 60,000 枚の画像を 5 周するので 300,000 枚の画像を学習することになります。
```python
model.fit(train_images, train_labels, epochs=5)
```
`accuracy` がおよそ 0.8～0.9 の間になったのではないでしょうか。
訓練データに対しては 8,9 割は正しく推論できるモデルが作成できました。

---

学習が完了しましたので、どの程度正しい結果を予測できるか確認しましょう。
ここではテスト用のデータで計測してみます。

```python
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

ここでも 0.8～0.9 の間になったのではないでしょうか。
でも先程よりも正解率が少し下がっていますね。

---

もう少しテストデータの予測結果を詳しく見てみましょう。
先頭のデータを見てみます。

```python
predictions = model.predict(test_images)
predictions[0]
```

数値が 10 個出力されています。
これが `softmax` 関数で出力されたどの衣服であるかの確率です。
どれも非常に小さい数値ですが、9番目の値が一番大きそうですね。

---

視覚的に確認するために関数を定義します。
```python
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
```

---

実際に結果を見てみましょう。
```python
import random
i = random.randint(0, 1000)
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
```

---

まとめて確認します。
```python
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
```


## Notebookインスタンスの削除

無料枠を超過した場合は課金が発生してしまうので起動した Notebook インスタンスを削除しましょう。

1. SageMakerのメニューに戻ります
1. ノートブックインスタンスを選択
![img101](https://user-images.githubusercontent.com/66664167/102602159-7c20ee00-4164-11eb-8cfd-b55c27b661cd.PNG)
1. アクション▼
1. 停止
1. ノートブックインスタンスのステータスが `Stopped` になるまでしばらく待ちます
![img102](https://user-images.githubusercontent.com/66664167/102602195-8511bf80-4164-11eb-8d9c-76ebca5e6cc0.PNG)
1. アクション▼
1. 削除

これで安心です。


## 用語

**SageMaker**

AWSの機械学習環境を構築するサービス
学習、モデルのデプロイなどをサポートする

**JupyterNotebook**

機械学習向けの開発環境
Pythonでの記述が多いけど他の言語も対応している
セルという単位でコードを記述し実行していく

**MNIST**

データセットの名前
単に MNIST というと手書きの数字の画像データを指す。
扱いやすいので機械学習初学者やテスト向けのデータとしてたびたび目にする。
今回は衣服のデータとカテゴリの正解ラベルがセットになった Fashion MNIST を使う。

**TensorFlow**

Google製の機械学習ライブラリのこと。

**Keras**

TensorFlow を使いやすくラップしたライブラリのこと。

**過学習**

訓練用のデータでは高い精度の推論ができるが、それ以外のデータの場合はよくない制度になってしまう問題。
