import os
import tensorflow as tf
import numpy as np
from read_utils import TextConverter, batch_generator
from model import Model,Config
# from model_attention import Model,Config

def main(_):

    model_path = os.path.join('models', Config.file_name)

    et = TextConverter(text=None,save_dir='models/en_vocab.pkl', max_vocab=Config.en_vocab_size, seq_length = Config.seq_length)
    zt = TextConverter(text=None,save_dir='models/zh_vocab.pkl', max_vocab=Config.zh_vocab_size, seq_length = Config.seq_length+1)  # +1是因为，decoder层序列拆成input=[:-1]和label=[1:]
    print('english vocab lens:',et.vocab_size)
    print('chinese vocab lens:',zt.vocab_size)


    # 加载上一次保存的模型
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    while True:
        # english_speek = 'what can i help you ?'
        # print('english:', english_speek)
        english_speek = input("english:")

        english_speek = english_speek.split()
        en_arr, arr_len = et.text_to_arr(english_speek)

        test_g = [np.array([en_arr,]), np.array([arr_len,])]
        output_ids = model.test(test_g, model_path, zt)
        strs = zt.arr_to_text(output_ids)
        print('chinese:',strs)


if __name__ == '__main__':
    tf.app.run()