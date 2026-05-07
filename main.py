"""
分词策略对中英词向量学习的影响
=====================================
燚炎语料库中英平行语料分析

数据来源：燚炎TMX平行语料库
分词方案对比：
  1. 方案一：中文Char + 英文BPE
  2. 方案二：纯BPE（统一BPE分词）
  3. 方案三：中文词级 + 英文词级
"""
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import (
    YiyanCorpusProcessor,
    ChineseCharEnglishBPETokenizer,
    PureBPETokenizer,
    WordLevelTokenizer,
    create_typo_words
)
from train import Word2VecTrainer
from evaluate import Evaluator, EnhancedVisualizer


def main():
    """主函数"""
    print("="*70)
    print("分词策略对中英词向量学习的影响")
    print("="*70)

    corpus_dir = r"d:\code\nlp\work\Yiyan_tmx"
    output_dir = r"d:\code\nlp\work\output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[Step 1] 加载并处理燚炎语料库...")
    processor = YiyanCorpusProcessor(corpus_dir)
    en_sents, zh_sents = processor.process_all()
    processor.print_stats()

    all_sentences = en_sents + zh_sents

    en_corpus = ' '.join(en_sents)
    zh_corpus = ' '.join(zh_sents)
    mixed_corpus = ' '.join(all_sentences)

    print("\n[Step 2] 准备三种分词方案...")

    model1_path = os.path.join(output_dir, 'scheme1_word2vec.model')
    model2_path = os.path.join(output_dir, 'scheme2_word2vec.model')
    model3_path = os.path.join(output_dir, 'scheme3_word2vec.model')
    tokenizer1_path = os.path.join(output_dir, 'scheme1_tokenizer.json')
    tokenizer2_path = os.path.join(output_dir, 'scheme2_tokenizer.json')

    training_histories = {}

    print("\n  方案一：中文Char + 英文BPE")
    tokenizer1 = ChineseCharEnglishBPETokenizer(vocab_size=8000)
    if os.path.exists(model1_path) and os.path.exists(tokenizer1_path):
        print(f"  ✓ 检测到已有模型和分词器，直接加载...")
        from gensim.models import Word2Vec
        model1 = Word2Vec.load(model1_path)
        tokenizer1.load(tokenizer1_path)
        trainer1 = Word2VecTrainer(tokenizer1, name="Scheme1_CharBPE")
        corpus1 = trainer1.prepare_corpus(en_sents, zh_sents)
        training_histories['Char+BPE'] = {'epoch': [], 'loss': []}
    else:
        tokenizer1.train(en_corpus)
        if os.path.exists(tokenizer1_path):
            print(f"  ✓ 检测到已有分词器，加载中...")
            tokenizer1.load(tokenizer1_path)
        trainer1 = Word2VecTrainer(tokenizer1, name="Scheme1_CharBPE")
        corpus1 = trainer1.prepare_corpus(en_sents, zh_sents)
        if os.path.exists(model1_path):
            print(f"  ✓ 检测到已有模型，加载中...")
            from gensim.models import Word2Vec
            model1 = Word2Vec.load(model1_path)
        else:
            model1 = trainer1.train(corpus1, vector_size=100, window=5, min_count=2, epochs=15)
            model1.save(model1_path)
            print(f"  ✓ 模型已保存")
        if not os.path.exists(tokenizer1_path):
            tokenizer1.save(tokenizer1_path)
            print(f"  ✓ 分词器已保存")
        training_histories['Char+BPE'] = trainer1.train_history

    print("\n  方案二：纯BPE（统一分词）")
    tokenizer2 = PureBPETokenizer(vocab_size=12000)
    if os.path.exists(model2_path) and os.path.exists(tokenizer2_path):
        print(f"  ✓ 检测到已有模型和分词器，直接加载...")
        from gensim.models import Word2Vec
        model2 = Word2Vec.load(model2_path)
        tokenizer2.load(tokenizer2_path)
        trainer2 = Word2VecTrainer(tokenizer2, name="Scheme2_PureBPE")
        corpus2 = trainer2.prepare_corpus(en_sents, zh_sents)
        training_histories['PureBPE'] = {'epoch': [], 'loss': []}
    else:
        tokenizer2.train(mixed_corpus)
        if os.path.exists(tokenizer2_path):
            print(f"  ✓ 检测到已有分词器，加载中...")
            tokenizer2.load(tokenizer2_path)
        trainer2 = Word2VecTrainer(tokenizer2, name="Scheme2_PureBPE")
        corpus2 = trainer2.prepare_corpus(en_sents, zh_sents)
        if os.path.exists(model2_path):
            print(f"  ✓ 检测到已有模型，加载中...")
            from gensim.models import Word2Vec
            model2 = Word2Vec.load(model2_path)
        else:
            model2 = trainer2.train(corpus2, vector_size=100, window=5, min_count=2, epochs=15)
            model2.save(model2_path)
            print(f"  ✓ 模型已保存")
        if not os.path.exists(tokenizer2_path):
            tokenizer2.save(tokenizer2_path)
            print(f"  ✓ 分词器已保存")
        training_histories['PureBPE'] = trainer2.train_history

    print("\n  方案三：中文词级 + 英文词级")
    tokenizer3 = WordLevelTokenizer()
    if os.path.exists(model3_path):
        print(f"  ✓ 检测到已有模型，直接加载: {model3_path}")
        from gensim.models import Word2Vec
        model3 = Word2Vec.load(model3_path)
        trainer3 = Word2VecTrainer(tokenizer3, name="Scheme3_WordLevel")
        corpus3 = trainer3.prepare_corpus(en_sents, zh_sents)
        training_histories['WordLevel'] = {'epoch': [], 'loss': []}
    else:
        trainer3 = Word2VecTrainer(tokenizer3, name="Scheme3_WordLevel")
        corpus3 = trainer3.prepare_corpus(en_sents, zh_sents)
        model3 = trainer3.train(corpus3, vector_size=100, window=5, min_count=2, epochs=15)
        training_histories['WordLevel'] = trainer3.train_history
    
    print("\n[Step 3] 评估各方案...")
    evaluators = {
        'Char+BPE': Evaluator(model1, 'Char+BPE'),
        'PureBPE': Evaluator(model2, 'PureBPE'),
        'WordLevel': Evaluator(model3, 'WordLevel')
    }

    train_vocab1 = set(model1.wv.index_to_key)
    train_vocab2 = set(model2.wv.index_to_key)
    train_vocab3 = set(model3.wv.index_to_key)

    print("\n--- 高频词近邻分析 ---")
    high_freq_results = {}
    for name, eval in evaluators.items():
        print(f"  正在评估: {name}")
        result = eval.evaluate_high_freq_words(corpus1, top_n=15)
        high_freq_results[name] = result
        print(f"  [{name}] 评估完成，获取 {len(result)} 个高频词")
        print(f"\n[{name}] 高频词词表覆盖情况:")
        sample_words = list(result.keys())[:5]
        for word in sample_words:
            status = "✓ 在词表中" if result[word]['in_vocab'] else "✗ 不在词表中"
            print(f"  {word}: {status}")

    print("\n--- 低频词近邻分析 ---")
    low_freq_results = {}
    for name, eval in evaluators.items():
        result = eval.evaluate_low_freq_words(corpus1, bottom_n=20)
        low_freq_results[name] = result
        print(f"\n[{name}] 低频词词表覆盖情况:")
        if result:
            sample_words = list(result.keys())[:3]
            for word in sample_words:
                status = "✓ 在词表中" if result[word]['in_vocab'] else "✗ 不在词表中"
                print(f"  {word}: {status}")

    print("\n--- OOV词分析 ---")
    typo_words = create_typo_words(train_vocab1, num=15)
    for name, vocab in [('Char+BPE', train_vocab1), ('PureBPE', train_vocab2), ('WordLevel', train_vocab3)]:
        oov_count = sum(1 for w in typo_words if w not in vocab)
        print(f"[{name}] 错拼词OOV数量: {oov_count}/{len(typo_words)}")

    print("\n[Step 4] 定量分析指标...")

    test_pairs = [
        ('america', 'united', 0.8),
        ('china', 'beijing', 0.7),
        ('japan', 'tokyo', 0.7),
        ('korea', 'seoul', 0.7),
        ('government', 'state', 0.6),
        ('president', 'obama', 0.9),
        ('clinton', 'hillary', 0.85),
        ('economic', 'financial', 0.7),
    ]

    quantitative_metrics = {}

    for name, eval in evaluators.items():
        metrics = {}

        coherence = eval.compute_similarity_coherence(test_pairs)
        if coherence:
            metrics['word_pair_coverage'] = coherence['coverage_rate']

        coverage = eval.compute_vocab_coverage(corpus1)
        metrics['coverage'] = coverage['coverage']
        metrics['oov_rate'] = coverage['oov_rate']

        avg_sim = 0
        count = 0
        for word in list(model1.wv.index_to_key)[:100]:
            try:
                if word in eval.model.wv:
                    count += 1
            except KeyError:
                continue
        metrics['avg_similarity'] = avg_sim / max(1, count) if count > 0 else 0

        quantitative_metrics[name] = metrics

    print("\n定量指标汇总:")
    print("-" * 60)
    print(f"{'方案':<15} {'词对覆盖率':<10} {'覆盖率':<10} {'OOV率':<10}")
    print("-" * 60)
    for name, metrics in quantitative_metrics.items():
        print(f"{name:<15} {metrics.get('word_pair_coverage', 0):<10.3f} {metrics.get('coverage', 0):<10.3f} "
              f"{metrics.get('oov_rate', 0):<10.3f}")

    print("\n[Step 5] 研究分析：语义边界保持性与频率分层退化...")

    semantic_words = [
        'america', 'china', 'japan', 'korea', 'beijing', 'tokyo', 'seoul',
        'president', 'government', 'economic', 'financial', 'security',
        'clinton', 'obama', 'trump', 'international'
    ]

    semantic_boundary_results = {}
    for name, eval in evaluators.items():
        print(f"\n  [{name}] 语义边界保持性分析...")
        result = eval.analyze_semantic_boundary_preservation(corpus1, semantic_words, topn=10)
        semantic_boundary_results[name] = result
        print(f"    完整词覆盖率: {result['intact_coverage']:.3f}")
        print(f"    边界保持性得分: {result['boundary_preservation_score']:.3f}")

    frequency_degradation_results = {}
    for name, eval in evaluators.items():
        print(f"\n  [{name}] 频率分层退化分析...")
        result = eval.analyze_frequency_layered_degradation(corpus1, top_n=20, bottom_n=20, topn_neighbors=10)
        frequency_degradation_results[name] = result
        print(f"    高频词覆盖率: {result['high_freq_stats']['in_vocab_rate']:.3f}")
        print(f"    低频词覆盖率: {result['low_freq_stats']['in_vocab_rate']:.3f}")
        print(f"    退化比: {result['degradation_ratio']:.3f}")

    print("\n[Step 6] 词向量可视化（增强版）...")
    
    # 创建增强可视化器
    visualizer = EnhancedVisualizer(output_dir)
    
    # 准备可视化词列表
    viz_words_en = [
        'america', 'china', 'japan', 'korea', 'india', 'russia',
        'president', 'government', 'economic', 'financial', 'security',
        'clinton', 'obama', 'trump', 'world', 'international',
        'united', 'states', 'beijing', 'tokyo', 'seoul'
    ]
    
    viz_words_zh = ['美国', '中国', '日本', '韩国', '总统', '政府', '经济', '安全']
    all_viz_words = list(set(viz_words_en + viz_words_zh))
    
    # 特殊标注词
    highlight_words = ['china', 'america', 'president', '经济']
    
    print("  生成各方案的PCA可视化...")
    
    # 方案一：PCA可视化
    visualizer.plot_pca(model1, all_viz_words, 'Char+BPE (Chinese Char + English BPE)', 
                       'scheme1_pca.png', highlight_words)
    
    # 方案二：PCA可视化
    visualizer.plot_pca(model2, all_viz_words, 'PureBPE (Unified BPE)', 
                       'scheme2_pca.png', highlight_words)
    
    # 方案三：PCA可视化
    visualizer.plot_pca(model3, all_viz_words, 'WordLevel (Chinese Jieba + English Word)', 
                       'scheme3_pca.png', highlight_words)
    
    print("\n  生成各方案的t-SNE可视化...")
    
    # t-SNE可视化（需要更多词）
    tsne_words = all_viz_words + ['war', 'peace', 'money', 'bank', 'school', 'university']
    
    visualizer.plot_tsne(model1, tsne_words, 'Char+BPE - t-SNE', 
                        'scheme1_tsne.png', highlight_words, perplexity=10)
    
    visualizer.plot_tsne(model2, tsne_words, 'PureBPE - t-SNE', 
                        'scheme2_tsne.png', highlight_words, perplexity=10)
    
    visualizer.plot_tsne(model3, tsne_words, 'WordLevel - t-SNE', 
                        'scheme3_tsne.png', highlight_words, perplexity=10)
    
    print("\n  生成三方案对比PCA图...")
    
    # 三方案对比
    models_compare = {
        'Char+BPE': model1,
        'PureBPE': model2,
        'WordLevel': model3
    }
    visualizer.plot_comparison_pca(models_compare, all_viz_words, 'comparison_pca.png')
    
    # 生成词表大小对比图
    print("\n  生成词表大小对比图...")
    vocab_sizes = {
        'Char+BPE': len(model1.wv),
        'PureBPE': len(model2.wv),
        'WordLevel': len(model3.wv)
    }
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(vocab_sizes.keys(), vocab_sizes.values(), 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    
    for bar, size in zip(bars, vocab_sizes.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
               f'{int(size):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Segmentation Scheme', fontsize=12)
    ax.set_ylabel('Vocabulary Size', fontsize=12)
    ax.set_title('Vocabulary Size Comparison Across Segmentation Strategies', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    vocab_path = os.path.join(output_dir, 'vocab_size_comparison.png')
    plt.savefig(vocab_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 词表大小对比图已保存: {vocab_path}")

    print("\n  生成语义边界保持性分析可视化...")
    visualizer.plot_semantic_boundary_analysis(semantic_boundary_results, 'semantic_boundary_analysis.png')

    print("\n  生成频率分层退化分析可视化...")
    visualizer.plot_frequency_degradation_analysis(frequency_degradation_results, 'frequency_degradation_analysis.png')

    print("\n" + "="*70)
    print("实验完成！所有结果已保存至:", output_dir)
    print("\n生成的文件:")
    print("  - scheme1_pca.png / scheme2_pca.png / scheme3_pca.png (PCA可视化)")
    print("  - scheme1_tsne.png / scheme2_tsne.png / scheme3_tsne.png (t-SNE可视化)")
    print("  - comparison_pca.png (三方案对比图)")
    print("  - vocab_size_comparison.png (词表大小对比)")
    print("  - semantic_boundary_analysis.png (语义边界保持性分析)")
    print("  - frequency_degradation_analysis.png (频率分层退化分析)")
    print("  - scheme*_word2vec.model (训练好的词向量模型)")
    print("  - scheme*_tokenizer.json (BPE分词器模型)")
    print("="*70)

    return {
        'corpus_stats': processor.stats,
        'high_freq_results': high_freq_results,
        'low_freq_results': low_freq_results,
        'quantitative_metrics': quantitative_metrics,
        'semantic_boundary_results': semantic_boundary_results,
        'frequency_degradation_results': frequency_degradation_results
    }


if __name__ == "__main__":
    results = main()
