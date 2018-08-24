import tensorflow as tf

class Config(object):
    """RNN配置参数"""
    file_name = 'rnn3'  #保存模型文件

    embedding_dim = 128      # 词向量维度
    seq_length = 26        # 序列长度
    # num_classes = 2        # 类别数
    en_vocab_size = 10000       # 词汇表达小
    zh_vocab_size = 4000

    # num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    # rnn = 'gru'             # lstm 或 gru
    share_emb_and_softmax = True  # 是否共享词向量层和sorfmax层的参数。（共享能减少参数且能提高模型效果）

    train_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 32  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size

        # 待输入的数据
        self.en_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='encode_input')
        self.en_length = tf.placeholder(tf.int32, [None], name='ec_length')

        self.zh_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='decode_input')
        self.zh_length = tf.placeholder(tf.int32, [None], name='zh_length')
        self.zh_seqs_label = tf.placeholder(tf.int32, [None, self.config.seq_length], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(3, dtype=tf.float32, trainable=False, name="global_loss")

        # seq2seq模型
        self.seq2seq()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def seq2seq(self):
        """seq2seq模型"""

        # 词嵌入层

        en_embedding = tf.get_variable('en_emb', [self.config.en_vocab_size, self.config.embedding_dim])
        zh_embedding = tf.get_variable('zh_emb', [self.config.zh_vocab_size, self.config.embedding_dim])
        embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
        self.en_embedding = tf.concat([en_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
        self.zh_embedding = tf.concat([zh_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值

        embed_en_seqs = tf.nn.embedding_lookup(self.en_embedding, self.en_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
        embed_zh_seqs = tf.nn.embedding_lookup(self.zh_embedding, self.zh_length)

        # 在词嵌入上进行dropout
        embed_en_seqs = tf.nn.dropout(embed_en_seqs, keep_prob=self.keep_prob)
        embed_zh_seqs = tf.nn.dropout(embed_zh_seqs, keep_prob=self.keep_prob)

        with tf.name_scope("encoder"):
            # 定义rnn网络
            enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, forget_bias=1.0)
            self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([enc_base_cell]*self.config.num_layers)
            # 通过dynamic_rnn对cell展开时间维度
            enc_output, self.enc_state= tf.nn.dynamic_rnn(self.enc_cell,
                                                            inputs=embed_en_seqs,
                                                            sequence_length=self.en_length,
                                                            # initial_state=self.initial_state1,  # 可有可无，自动为0状态
                                                            time_major=False,
                                                            dtype=tf.float32)
        with tf.name_scope("decoder"):
            # 定义rnn网络
            dec_base_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, forget_bias=1.0)
            self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([dec_base_cell]*self.config.num_layers)
            # 通过dynamic_rnn对cell展开时间维度
            dec_output, self.dec_state= tf.nn.dynamic_rnn(self.dec_cell,
                                                            inputs=embed_zh_seqs,
                                                            sequence_length=self.zh_length,
                                                            initial_state=self.enc_state,  # 编码层的输出来初始化解码层的隐层状态
                                                            time_major=False,
                                                            dtype=tf.float32)

        with tf.name_scope("sorfmax_weights"):
            if self.config.share_emb_and_softmax:
                self.softmax_weight = tf.transpose(self.zh_embedding)
            else:
                self.softmax_weight = tf.get_variable("weight",[self.config.hidden_dim, self.config.zh_vocab_size])
            self.softmax_bias = tf.get_variable("bias",[self.config.zh_vocab_size])


        with tf.name_scope("loss"):
            out_put = tf.reshape(dec_output, [-1, self.config.hidden_dim])
            logits = tf.matmul(out_put, self.softmax_weight) + self.softmax_bias
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.zh_seqs_label,[-1]), logits=logits)

        with tf.name_scope("optimize"):
            # 优化器
            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
            # train_op = tf.train.AdamOptimizer(self.config.learning_rate)
            # self.optim = train_op.apply_gradients(zip(grads, tvars),global_step=self.global_step)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss, global_step=self.global_step)

    def train(self, batch_train_g, model_path, val_g):
        with self.session as sess:
            for batch_en, batch_en_len, batch_zh, batch_zh_len, batch_zh_label in batch_train_g:
                feed = {self.en_seqs: batch_en,
                        self.en_length: batch_en_len,
                        self.zh_seqs: batch_zh,
                        self.zh_length: batch_zh_len,
                        self.zh_seqs_label: batch_zh_label,
                        self.keep_prob: self.config.train_keep_prob}

                _ = sess.run(self.optim , feed_dict=feed)

