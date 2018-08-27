# Seq2seq_Translation
Tensorflow实现 seq2seq，并训练实现英-中翻译

方法一、LSTM

方法二、LSTM+attention

# Data
来源:https://wit3.fbk.eu/mt.php?release=2015-01

需要更多数据可以参考WMT数据集。WMT，全称Workshop on Machine Translation

# Train
第一次训练之前先运行数据处理程序
> python read_utils.py

训练
> python train.py

```
start to training...
samples number: 209941
step: 20/20000...  loss: 3.5204544067382812...  1.6092 sec/batch
step: 40/20000...  loss: 3.428807258605957...  1.6902 sec/batch
step: 60/20000...  loss: 3.4038989543914795...  1.6112 sec/batch
.
.
.
step: 19880/20000...  loss: 1.8291130065917969...  0.1160 sec/batch
step: 19900/20000...  loss: 1.8759247064590454...  0.1166 sec/batch
step: 19920/20000...  loss: 1.5523113012313843...  0.1174 sec/batch
step: 19940/20000...  loss: 1.590277075767517...  0.1164 sec/batch
step: 19960/20000...  loss: 1.7223188877105713...  0.1161 sec/batch
step: 19980/20000...  loss: 1.9042510986328125...  0.1182 sec/batch
step: 20000/20000...  loss: 1.7752913236618042...  0.1157 sec/batch
```

# Result
> python test.py

```
english: what is that ?
chinese: 那么什么呢？
english:just do it
chinese: 就是这样的
english:i love you too
chinese: 我爱你们
english:what's your name ?
chinese: 你的名字是什么？
english:Most of the planet is ocean water .
chinese: 地球上的海洋是水平。
english:We have to have a very special technology to get into that unfamiliar world .
chinese: 你必须有一个非常有意识的技术来创造一个非常有意义的人
english:This is too good to be true .
chinese: 太棒了很多。
```

# Model:

链接：https://pan.baidu.com/s/1xCd3kW6ggtXxW2H6ygbFuA 密码：27i4
