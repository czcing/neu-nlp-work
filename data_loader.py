"""
分词策略对中英词向量学习的影响 - 数据加载模块
===============================================
包含语料库处理器和分词器定义
"""
import os
import re
import xml.etree.ElementTree as ET
import random
from collections import Counter

random.seed(42)


class YiyanCorpusProcessor:
    """燚炎语料库处理器"""

    def __init__(self, tmx_dir):
        self.tmx_dir = tmx_dir
        self.en_sentences = []
        self.zh_sentences = []
        self.stats = {
            'total_files': 0,
            'total_tuples': 0,
            'en_chars': 0,
            'zh_chars': 0,
            'en_words': 0,
            'zh_words': 0
        }

    def parse_tmx_file(self, filepath):
        """解析单个TMX文件"""
        en_sents = []
        zh_sents = []

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            for tu in root.findall('.//tu'):
                tuv_list = tu.findall('tuv')
                en_text = None
                zh_text = None

                for tuv in tuv_list:
                    lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang') or tuv.get('xml:lang', '')
                    seg = tuv.find('seg')
                    if seg is not None and seg.text:
                        if 'en' in lang.lower():
                            en_text = seg.text.strip()
                        elif 'zh' in lang.lower():
                            zh_text = seg.text.strip()

                if en_text and zh_text and len(en_text) > 5 and len(zh_text) > 3:
                    en_sents.append(en_text)
                    zh_sents.append(zh_text)

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return en_sents, zh_sents

    def process_all(self):
        """处理所有TMX文件"""
        tmx_files = [f for f in os.listdir(self.tmx_dir) if f.endswith('.tmx')]
        self.stats['total_files'] = len(tmx_files)

        for i, filename in enumerate(tmx_files):
            filepath = os.path.join(self.tmx_dir, filename)
            en_sents, zh_sents = self.parse_tmx_file(filepath)
            self.en_sentences.extend(en_sents)
            self.zh_sentences.extend(zh_sents)
            self.stats['total_tuples'] += len(en_sents)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(tmx_files)} files, collected {len(self.en_sentences)} parallel sentences")

        self._calculate_stats()
        return self.en_sentences, self.zh_sentences

    def _calculate_stats(self):
        """计算语料统计信息"""
        all_en_text = ' '.join(self.en_sentences)
        all_zh_text = ' '.join(self.zh_sentences)

        self.stats['en_chars'] = len(all_en_text)
        self.stats['zh_chars'] = len(all_zh_text)
        self.stats['en_words'] = len(all_en_text.split())
        self.stats['zh_words'] = len(re.findall(r'[\u4e00-\u9fff]+', all_zh_text))

    def get_clean_data(self):
        """返回清洗后的数据"""
        return self.en_sentences, self.zh_sentences

    def print_stats(self):
        """打印语料库统计信息"""
        print("\n" + "="*60)
        print("燚炎语料库统计信息")
        print("="*60)
        print(f"TMX文件总数: {self.stats['total_files']}")
        print(f"平行句对数: {self.stats['total_tuples']}")
        print(f"英文总字符数: {self.stats['en_chars']:,}")
        print(f"中文总字符数: {self.stats['zh_chars']:,}")
        print(f"英文词数(估): {self.stats['en_words']:,}")
        print(f"中文字数(估): {self.stats['zh_words']:,}")
        print(f"平均英文句长: {self.stats['en_chars'] / max(1, self.stats['total_tuples']):.1f} 字符")
        print(f"平均中文句长: {self.stats['zh_chars'] / max(1, self.stats['total_tuples']):.1f} 字符")
        print("="*60)


class BaseTokenizer:
    """分词器基类"""

    def tokenize(self, text, lang='en'):
        raise NotImplementedError

    def train(self, corpus):
        """训练分词器"""
        pass


class ChineseCharEnglishBPETokenizer(BaseTokenizer):
    """方案一：中文按字符，英文按BPE - 使用HuggingFace Tokenizers"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.en_tokenizer = None

    def train(self, en_corpus):
        """训练英文BPE分词器"""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        
        bpe_model = models.BPE()
        self.en_tokenizer = Tokenizer(bpe_model)
        self.en_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<unk>", "<s>", "</s>"]
        )
        
        self.en_tokenizer.train_from_iterator(
            en_corpus.split('\n'),
            trainer=trainer
        )
        
        print(f"  ✓ BPE分词器训练完成，词表大小: {self.en_tokenizer.get_vocab_size()}")

    def save(self, path):
        """保存BPE分词器到文件"""
        self.en_tokenizer.save(path)
        print(f"  ✓ BPE分词器已保存: {path}")

    def load(self, path):
        """从文件加载BPE分词器"""
        from tokenizers import Tokenizer
        self.en_tokenizer = Tokenizer.from_file(path)
        print(f"  ✓ BPE分词器已加载: {path}")

    def tokenize(self, text, lang='en'):
        if lang == 'en':
            result = self.en_tokenizer.encode(text.lower())
            return [token for token in result.tokens if token not in ["<s>", "</s>", "<unk>"]]
        else:
            return list(text)


class PureBPETokenizer(BaseTokenizer):
    """方案二：纯BPE分词 - 使用HuggingFace Tokenizers"""

    def __init__(self, vocab_size=15000):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, corpus):
        """训练统一BPE分词器"""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        
        bpe_model = models.BPE()
        self.tokenizer = Tokenizer(bpe_model)
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<unk>", "<s>", "</s>"]
        )
        
        self.tokenizer.train_from_iterator(
            corpus.split('\n'),
            trainer=trainer
        )
        
        print(f"  ✓ 统一BPE分词器训练完成，词表大小: {self.tokenizer.get_vocab_size()}")

    def save(self, path):
        """保存BPE分词器到文件"""
        self.tokenizer.save(path)
        print(f"  ✓ BPE分词器已保存: {path}")

    def load(self, path):
        """从文件加载BPE分词器"""
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        print(f"  ✓ BPE分词器已加载: {path}")

    def tokenize(self, text, lang='both'):
        result = self.tokenizer.encode(text)
        return [token for token in result.tokens if token not in ["<s>", "</s>", "<unk>"]]


class WordLevelTokenizer(BaseTokenizer):
    """方案三：词级分词（中文结巴分词，英文按空格）"""

    def __init__(self):
        try:
            import jieba
            self.jieba = jieba
        except ImportError:
            print("Warning: jieba not installed, using character fallback for Chinese")
            self.jieba = None

    def tokenize(self, text, lang='en'):
        if lang == 'en':
            return text.lower().split()
        else:
            if self.jieba:
                return list(self.jieba.cut(text))
            else:
                return list(text)


def create_typo_words(vocab, num=10):
    """创建模拟错拼词"""
    typo_words = []
    vocab_list = list(vocab)[:100]

    for word in random.sample(vocab_list, min(num, len(vocab_list))):
        if len(word) > 3:
            chars = list(word)
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            typo_words.append(''.join(chars))

    return typo_words
