import os,re
import numpy as np
import pickle
import random


class Preprocess():
    def __init__(self):
        pass

    def clears(self,):
        # 数据清洗
        data_en = 'data/train.tags.en-zh.en'
        data_zh = 'data/train.tags.en-zh.zh'
        data_en_clear = 'data/train.tags.en-zh.en_clear'
        data_zh_clear = 'data/train.tags.en-zh.zh_clear'
        with open(data_en, 'r',encoding='utf-8') as f_en:
            f_en_w = open(data_en_clear, 'w', encoding='utf-8')
            lineID = 0
            for line in f_en:
                if '<'==line[0] and '>' ==line[-2]:
                    continue
                lineID += 1
                line = re.sub(u"[^0-9a-zA-Z.,?!']+",' ',line)  # 清除不需要的字符
                line = re.sub(u"[.,?!]+", lambda x:' '+x.group(0),line)  # 在标点符号前插入空格
                f_en_w.write(line+'\n')
            print('english lines number:',lineID)
            f_en_w.close()

        with open(data_zh, 'r',encoding='utf-8') as f_zh:
            f_zh_w = open(data_zh_clear, 'w', encoding='utf-8')
            lineID = 0
            for line in f_zh:
                if '<' == line[0] and '>' == line[-2]:
                    continue
                lineID += 1
                line = re.sub(u"[^0-9\u4e00-\u9fa5。，？！']+",'',line)  # 清除不需要的字符
                # line = re.sub(u"[.,?!]+", lambda x:' '+x.group(0),line)  # 在标点符号前插入空格
                f_zh_w.write(line+'\n')
            print('chinese lines number:',lineID)
            f_zh_w.close()

    def get_text(self):
        # 获得所有文字
        data_en_clear = 'data/train.tags.en-zh.en_clear'
        data_zh_clear = 'data/train.tags.en-zh.zh_clear'
        en_list = []  # 英文以单词为粒度
        zh_list = ''  # 中文以字符为粒度
        with open(data_en_clear, 'r', encoding='utf-8') as f_en:
            for line in f_en:
                en_list += line.split()

        with open(data_zh_clear, 'r', encoding='utf-8') as f_zh:
            for line in f_zh:
                line = line.strip()
                zh_list += line

        return en_list,zh_list

    def get_samples(self):
        pass



class TextConverter(object):
    def __init__(self, text=None, save_dir=None, max_vocab=5000 , seq_length = 20):
        if os.path.exists(save_dir):
            with open(save_dir, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print('字符数量：%s ' % len(vocab))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab
            with open(save_dir, 'wb') as f:
                pickle.dump(self.vocab, f)

        self.seq_length = seq_length  # 样本序列最大长度
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)


    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)


    def text_to_arr(self, text):
        arr = []
        last_num = len(self.vocab)
        query_len = len(text)
        for word in text:
            arr.append(self.word_to_int(word))

        # padding
        if query_len < self.seq_length:
            arr += [last_num] * (self.seq_length - query_len)
        else:
            arr = arr[:self.seq_length]
            query_len = self.seq_length

        return np.array(arr), np.array(query_len)



    def QAs_to_arr(self, QAs):

        QA_arrs = []
        for query, response, label in QAs:
            # text to arr
            query_arr,query_len = self.text_to_arr(query)
            response_arr,response_len = self.text_to_arr(response)
            QA_arrs.append([query_arr,query_len,response_arr,response_len, float(label)])
        return QA_arrs

    def libs_to_arrs(self,libs):
        libs_arrs = []
        for response in libs:
            response_arr,response_len = self.text_to_arr(response)
            libs_arrs.append([response_arr,response_len])
        return libs_arrs

    def batch_generator(self,QA_arrs, batchsize):
        '''产生训练batch样本'''
        n_samples = len(QA_arrs)
        n_batches = int(n_samples / batchsize)
        n = n_batches * batchsize
        while True:
            random.shuffle(QA_arrs)  # 打乱顺序
            for i in range(0, n, batchsize):
                batch_samples = QA_arrs[i:i + batchsize]
                batch_q = []
                batch_q_len = []
                batch_r = []
                batch_r_len = []
                batch_y = []
                for sample in batch_samples:
                    batch_q.append(sample[0])
                    batch_q_len.append(sample[1])
                    batch_r.append(sample[2])
                    batch_r_len.append(sample[3])
                    batch_y.append(sample[4])
                yield np.array(batch_q), np.array(batch_q_len), np.array(batch_r), np.array(batch_r_len), np.array(batch_y)


    def val_samples_generator(self,QA_arrs, batchsize=500):
        '''产生验证样本，batchsize分批验证，减少运行内存'''

        val_g = []
        n = len(QA_arrs)
        for i in range(0, n, batchsize):
            batch_samples = QA_arrs[i:i + batchsize]
            batch_q = []
            batch_q_len = []
            batch_r = []
            batch_r_len = []
            batch_y = []
            for sample in batch_samples:
                batch_q.append(sample[0])
                batch_q_len.append(sample[1])
                batch_r.append(sample[2])
                batch_r_len.append(sample[3])
                batch_y.append(sample[4])
            val_g.append((np.array(batch_q), np.array(batch_q_len), np.array(batch_r), np.array(batch_r_len), np.array(batch_y)))
        return val_g


    def index_to_QA_and_save(self,indexs,QAs, path):
        print("start writing to eccel... ")
        outputbook = xlwt.Workbook()
        oh = outputbook.add_sheet('sheet1',cell_overwrite_ok=True)
        for index_q, index_r in indexs:
            que = QAs[index_q][0]
            oh.write(index_q, 0, que)
            k = 0
            for r_i in list(index_r):
                res = QAs[r_i][1]
                oh.write(index_q, 2+k, res)
                k += 1
                if k > 5:
                    break
        outputbook.save(path+'_Q_for_QA.xls')
        print('finished!')

    def index_to_response(self, index_list, libs):
        responses = []
        for index in index_list:
            responses.append(libs[index])
        return responses
    def index_to_response2(self, index_list, QAs):
        responses = []
        for index in index_list:
            responses.append(QAs[index][-1])
        return responses

    def save_to_excel(self, QAY, path):
        '''result to save...'''
        outputbook = xlwt.Workbook()
        oh = outputbook.add_sheet('sheet1', cell_overwrite_ok=True)
        k = 0
        for query, y_response, responses in QAY:
            oh.write(k, 0, query)
            oh.write(k, 1, y_response)
            i = 0
            for response in responses:
                oh.write(k, 2 + i, response)
                i += 1
                if i > 5:
                    break
            k += 1
        outputbook.save(path )
        print('finished!')



if __name__ == '__main__':
    pass
    # loadConversations('data/xiaohuangji50w_fenciA.conv')

    pre =  Preprocess()
    # pre.clears()
    a, b = pre.get_text()
    print(len(a))
    print(len(b))

    et = TextConverter(text=a,save_dir='models/en_vocab.pkl')
    zt = TextConverter(text=b,save_dir='models/zh_vocab.pkl')
    print(et.vocab)
    print(zt.vocab)
