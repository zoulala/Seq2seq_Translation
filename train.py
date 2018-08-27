import os
import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import  Model,Config
# from model_attention import Model,Config

def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)


    et = TextConverter(text=None,save_dir='models/en_vocab.pkl', max_vocab=Config.en_vocab_size, seq_length = Config.seq_length)
    zt = TextConverter(text=None,save_dir='models/zh_vocab.pkl', max_vocab=Config.zh_vocab_size, seq_length = Config.seq_length+1)  # +1是因为，decoder层序列拆成input=[:-1]和label=[1:]
    print('english vocab lens:',et.vocab_size)
    print('chinese vocab lens:',zt.vocab_size)

    en_arrs = et.get_en_arrs('data/train.tags.en-zh.en_clear')
    zh_arrs = zt.get_en_arrs('data/train.tags.en-zh.zh_clear')

    train_g = batch_generator( en_arrs, zh_arrs, Config.batch_size)


    # 加载上一次保存的模型
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to training...')
    model.train(train_g, model_path)



if __name__ == '__main__':
    tf.app.run()