"""
分词策略对中英词向量学习的影响 - 训练模块
===============================================
包含词向量训练器定义
"""
import os
import numpy as np
from gensim.models.word2vec import Word2Vec

np.random.seed(42)


class Word2VecTrainer:
    """词向量训练器"""

    def __init__(self, tokenizer, name="model"):
        self.tokenizer = tokenizer
        self.name = name
        self.model = None
        self.vocab = None
        self.train_history = {'epoch': [], 'loss': []}

    def prepare_corpus(self, en_sentences, zh_sentences):
        """准备语料库"""
        tokenized_corpus = []

        for sent in en_sentences:
            tokens = self.tokenizer.tokenize(sent, lang='en')
            if tokens:
                tokenized_corpus.append(tokens)

        for sent in zh_sentences:
            tokens = self.tokenizer.tokenize(sent, lang='zh')
            if tokens:
                tokenized_corpus.append(tokens)

        return tokenized_corpus

    def train(self, corpus, vector_size=100, window=5, min_count=3, epochs=20, workers=4):
        """训练Word2Vec模型"""
        print(f"\n[{self.name}] 开始训练词向量...")
        print(f"  语料规模: {len(corpus)} 句子")
        print(f"  向量维度: {vector_size}")
        print(f"  窗口大小: {window}")
        print(f"  最小词频: {min_count}")
        print(f"  训练轮数: {epochs}")

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("  (提示: 安装tqdm可显示训练进度条: pip install tqdm)")

        class TrainingCallback:
            def __init__(self, trainer, total_epochs, use_tqdm):
                self.trainer = trainer
                self.total_epochs = total_epochs
                self.use_tqdm = use_tqdm
                self.pbar = None
                if use_tqdm:
                    self.pbar = tqdm(total=total_epochs, desc=f"[{trainer.name}] Epoch", leave=True)

            def on_train_begin(self, model):
                pass

            def on_train_end(self, model):
                if self.use_tqdm and self.pbar:
                    self.pbar.close()

            def on_epoch_begin(self, model):
                pass

            def on_epoch_end(self, model):
                self.trainer.train_history['epoch'].append(len(self.trainer.train_history['epoch']) + 1)
                try:
                    loss = model.get_latest_training_loss()
                    self.trainer.train_history['loss'].append(loss)
                except:
                    loss = 0
                    self.trainer.train_history['loss'].append(loss)

                if self.use_tqdm and self.pbar:
                    self.pbar.update(1)
                    self.pbar.set_postfix({'loss': f'{loss:.2f}'})

            def __call__(self, model, epoch):
                self.on_epoch_end(model)

            def __del__(self):
                if self.use_tqdm and self.pbar:
                    self.pbar.close()

        callback = TrainingCallback(self, epochs, use_tqdm)

        self.model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=1,
            hs=0,
            negative=5,
            seed=42,
            callbacks=[callback]
        )

        self.vocab = {word: self.model.wv[word] for word in self.model.wv.index_to_key}

        unique_tokens = len(self.model.wv)
        print(f"  词表大小: {unique_tokens}")
        print(f"[{self.name}] 训练完成!")
        if self.train_history['loss']:
            final_loss = self.train_history['loss'][-1]
            print(f"  最终损失值: {final_loss:.2f}")

        return self.model

    def get_similar(self, word, topn=10):
        """获取相似词"""
        if self.model is None:
            return []
        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            return []

    def save_model(self, path):
        """保存模型"""
        self.model.save(path)

    def load_model(self, path):
        """加载模型"""
        self.model = Word2Vec.load(path)
        self.vocab = {word: self.model.wv[word] for word in self.model.wv.index_to_key}
