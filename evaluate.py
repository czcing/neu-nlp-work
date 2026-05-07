"""
分词策略对中英词向量学习的影响 - 评估模块
===============================================
包含评估器和可视化器定义
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedVisualizer:
    """增强版可视化器 - 使用PCA和t-SNE"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_pca(self, model, words, title, filename, color_words=None):
        """PCA降维可视化"""
        try:
            # 过滤出在词表中的词
            valid_words = [w for w in words if w in model.wv]
            if len(valid_words) < 3:
                print(f"    警告: {title} 有效词数不足3个，跳过PCA可视化")
                return False
            
            # 提取词向量
            vectors = np.array([model.wv[w] for w in valid_words])
            
            # PCA降维
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # 绘制散点
            scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                                c=range(len(valid_words)), cmap='viridis', 
                                s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # 添加标签
            for i, word in enumerate(valid_words):
                # 特殊颜色标注
                if color_words and word in color_words:
                    ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                               fontsize=11, fontweight='bold', color='red',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
                else:
                    ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                               fontsize=10, alpha=0.8)
            
            ax.set_title(f'{title} - PCA Visualization', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Word Index', fontsize=10)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ PCA可视化已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"    ✗ PCA可视化失败: {e}")
            return False
    
    def plot_tsne(self, model, words, title, filename, color_words=None, perplexity=30):
        """t-SNE降维可视化"""
        try:
            # 过滤出在词表中的词
            valid_words = [w for w in words if w in model.wv]
            if len(valid_words) < 5:
                print(f"    警告: {title} 有效词数不足5个，跳过t-SNE可视化")
                return False
            
            # 提取词向量
            vectors = np.array([model.wv[w] for w in valid_words])
            
            # 调整perplexity
            actual_perplexity = min(perplexity, len(valid_words) - 1)
            
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, 
                       perplexity=actual_perplexity, max_iter=1000)
            vectors_2d = tsne.fit_transform(vectors)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # 绘制散点
            scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                                c=range(len(valid_words)), cmap='plasma', 
                                s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # 添加标签
            for i, word in enumerate(valid_words):
                if color_words and word in color_words:
                    ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                               fontsize=11, fontweight='bold', color='red',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
                else:
                    ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                               fontsize=10, alpha=0.8)
            
            ax.set_title(f'{title} - t-SNE Visualization', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Word Index', fontsize=10)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ t-SNE可视化已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"    ✗ t-SNE可视化失败: {e}")
            return False
    
    def plot_comparison_pca(self, models_dict, words, filename):
        """对比多个模型的PCA可视化"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for idx, (name, model) in enumerate(models_dict.items()):
                valid_words = [w for w in words if w in model.wv]
                if len(valid_words) < 3:
                    print(f"    警告: {name} 有效词数不足，跳过")
                    continue
                
                vectors = np.array([model.wv[w] for w in valid_words])
                pca = PCA(n_components=2)
                vectors_2d = pca.fit_transform(vectors)
                
                ax = axes[idx]
                ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                          c=range(len(valid_words)), cmap='viridis', 
                          s=60, alpha=0.7)
                
                for i, word in enumerate(valid_words):
                    ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                               fontsize=9, alpha=0.8)
                
                ax.set_title(f'{name}', fontsize=12, fontweight='bold')
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Segmentation Strategies Comparison - PCA', fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 对比PCA可视化已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"  ✗ 对比PCA可视化失败: {e}")
            return False
    
    def plot_training_loss(self, history_dict, filename):
        """绘制训练损失曲线"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            markers = ['o', 's', '^']
            
            for i, (name, history) in enumerate(history_dict.items()):
                if history and history.get('epoch') and history.get('loss'):
                    # 过滤无效损失值
                    valid_losses = [l for l in history['loss'] if l is not None and l > 0]
                    valid_epochs = history['epoch'][:len(valid_losses)]
                    
                    if valid_losses:
                        ax.plot(valid_epochs, valid_losses,
                               label=name, color=colors[i % len(colors)],
                               marker=markers[i % len(markers)], markersize=4,
                               linewidth=1.5, alpha=0.8)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Training Loss', fontsize=12)
            ax.set_title('Word2Vec Training Loss - Segmentation Strategies Comparison', fontsize=14)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=1)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 训练损失曲线已保存: {output_path}")
            return True

        except Exception as e:
            print(f"  ✗ 训练损失曲线绘制失败: {e}")
            return False

    def plot_semantic_boundary_analysis(self, results_dict, filename):
        """可视化语义边界保持性分析结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            schemes = list(results_dict.keys())
            intact_coverages = [results_dict[s]['intact_coverage'] for s in schemes]
            boundary_scores = [results_dict[s]['boundary_preservation_score'] for s in schemes]
            subword_coverages = [results_dict[s]['subword_coverage'] for s in schemes]

            ax1 = axes[0, 0]
            x = np.arange(len(schemes))
            width = 0.35
            bars1 = ax1.bar(x - width/2, intact_coverages, width, label='Intact Word Coverage', color='#2ecc71', alpha=0.8)
            bars2 = ax1.bar(x + width/2, subword_coverages, width, label='Subword Coverage', color='#e74c3c', alpha=0.8)
            ax1.set_xlabel('Segmentation Scheme', fontsize=11)
            ax1.set_ylabel('Coverage Rate', fontsize=11)
            ax1.set_title('Semantic Boundary Preservation - Coverage', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(schemes, rotation=15)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            for bar in bars1:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=9)

            ax2 = axes[0, 1]
            bars = ax2.bar(schemes, boundary_scores, color=['#3498db', '#9b59b6', '#1abc9c'], alpha=0.8)
            ax2.set_xlabel('Segmentation Scheme', fontsize=11)
            ax2.set_ylabel('Boundary Preservation Score', fontsize=11)
            ax2.set_title('Semantic Boundary Preservation Score', fontsize=12, fontweight='bold')
            ax2.set_xticklabels(schemes, rotation=15)
            ax2.grid(axis='y', alpha=0.3)
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}',
                        ha='center', va='bottom', fontsize=10)

            ax3 = axes[1, 0]
            details_data = []
            for scheme in schemes:
                scheme_details = results_dict[scheme].get('details', [])
                for detail in scheme_details:
                    if detail.get('similarity_score') is not None:
                        details_data.append({
                            'scheme': scheme,
                            'word': detail['word'][:10],
                            'similarity': detail['similarity_score']
                        })

            if details_data:
                words = [d['word'] for d in details_data[:10]]
                similarities = [d['similarity'] for d in details_data[:10]]
                scheme_colors = {'Char+BPE': '#3498db', 'PureBPE': '#9b59b6', 'WordLevel': '#1abc9c'}
                colors = [scheme_colors.get(d['scheme'], '#95a5a6') for d in details_data[:10]]

                y_pos = np.arange(len(words))
                ax3.barh(y_pos, similarities, color=colors, alpha=0.8)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(words)
                ax3.set_xlabel('Semantic Similarity Score', fontsize=11)
                ax3.set_title('Word-Level Semantic Similarity', fontsize=12, fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)

            ax4 = axes[1, 1]
            ax4.axis('off')
            summary_text = "Semantic Boundary Preservation Analysis\n" + "=" * 40 + "\n\n"
            summary_text += "Interpretation Guide:\n"
            summary_text += "• Intact Coverage: Ratio of semantic complete words in vocabulary\n"
            summary_text += "• Subword Coverage: Ratio of subwords/characters captured\n"
            summary_text += "• Boundary Score: Cosine similarity between intact word vector\n"
            summary_text += "  and averaged subword vectors (higher = better preservation)\n\n"
            summary_text += "Key Insights:\n"
            for scheme in schemes:
                score = results_dict[scheme]['boundary_preservation_score']
                summary_text += f"• {scheme}: {score:.3f}\n"
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle('Research 1: Semantic Boundary Preservation Analysis', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 语义边界保持性分析可视化已保存: {output_path}")
            return True

        except Exception as e:
            print(f"  ✗ 语义边界保持性可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def plot_frequency_degradation_analysis(self, results_dict, filename):
        """可视化频率分层退化分析结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            schemes = list(results_dict.keys())

            high_freq_rates = [results_dict[s]['high_freq_stats']['in_vocab_rate'] for s in schemes]
            low_freq_rates = [results_dict[s]['low_freq_stats']['in_vocab_rate'] for s in schemes]

            ax1 = axes[0, 0]
            x = np.arange(len(schemes))
            width = 0.35
            bars1 = ax1.bar(x - width/2, high_freq_rates, width, label='High Freq Words', color='#27ae60', alpha=0.8)
            bars2 = ax1.bar(x + width/2, low_freq_rates, width, label='Low Freq Words', color='#c0392b', alpha=0.8)
            ax1.set_xlabel('Segmentation Scheme', fontsize=11)
            ax1.set_ylabel('In-Vocabulary Rate', fontsize=11)
            ax1.set_title('Frequency-Layered Vocabulary Coverage', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(schemes, rotation=15)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            for bar in bars1:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=9)

            degradation_ratios = [results_dict[s]['degradation_ratio'] for s in schemes]
            frequency_gaps = [results_dict[s]['frequency_gap'] for s in schemes]

            ax2 = axes[0, 1]
            x = np.arange(len(schemes))
            width = 0.35
            bars1 = ax2.bar(x - width/2, degradation_ratios, width, label='Degradation Ratio', color='#8e44ad', alpha=0.8)
            bars2 = ax2.bar(x + width/2, frequency_gaps, width, label='Frequency Gap', color='#d35400', alpha=0.8)
            ax2.set_xlabel('Segmentation Scheme', fontsize=11)
            ax2.set_ylabel('Ratio / Gap', fontsize=11)
            ax2.set_title('Frequency Degradation Metrics', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(schemes, rotation=15)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            for bar in bars1:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=9)

            high_freq_valid_counts = [results_dict[s]['high_freq_stats']['valid_count'] for s in schemes]
            low_freq_valid_counts = [results_dict[s]['low_freq_stats']['valid_count'] for s in schemes]

            ax3 = axes[1, 0]
            x = np.arange(len(schemes))
            width = 0.35
            bars1 = ax3.bar(x - width/2, high_freq_valid_counts, width, label='High Freq Valid', color='#27ae60', alpha=0.8)
            bars2 = ax3.bar(x + width/2, low_freq_valid_counts, width, label='Low Freq Valid', color='#c0392b', alpha=0.8)
            ax3.set_xlabel('Segmentation Scheme', fontsize=11)
            ax3.set_ylabel('Word Count', fontsize=11)
            ax3.set_title('Valid Words by Frequency Layer', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(schemes, rotation=15)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            for bar in bars1:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{int(bar.get_height())}',
                        ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{int(bar.get_height())}',
                        ha='center', va='bottom', fontsize=9)

            ax4 = axes[1, 1]
            ax4.axis('off')
            summary_text = "Frequency-Layered Degradation Analysis\n" + "=" * 40 + "\n\n"
            summary_text += "Interpretation Guide:\n"
            summary_text += "• Degradation Ratio: Low/High freq neighbor quality ratio\n"
            summary_text += "  (closer to 1.0 = less degradation)\n"
            summary_text += "• Frequency Gap: Difference in vocabulary coverage\n"
            summary_text += "  between high and low frequency words\n\n"
            summary_text += "Key Insights:\n"
            for scheme in schemes:
                deg_ratio = results_dict[scheme]['degradation_ratio']
                freq_gap = results_dict[scheme]['frequency_gap']
                summary_text += f"• {scheme}:\n"
                summary_text += f"  - Degradation: {deg_ratio:.2f}\n"
                summary_text += f"  - Gap: {freq_gap:.2f}\n"
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

            plt.suptitle('Research 2: Frequency-Layered Degradation Analysis', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 频率分层退化分析可视化已保存: {output_path}")
            return True

        except Exception as e:
            print(f"  ✗ 频率分层退化可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return False


class Evaluator:
    """评估器"""

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def get_word_frequency_category(self, corpus, word, top_percent=20, bottom_percent=20):
        """判断词属于高频还是低频"""
        word_counts = Counter()
        for sent in corpus:
            word_counts.update(sent)

        if not word_counts:
            return 'unknown'

        sorted_words = word_counts.most_common()
        total_words = len(sorted_words)

        top_n = int(total_words * top_percent / 100)
        bottom_n = int(total_words * bottom_percent / 100)

        for i, (w, _) in enumerate(sorted_words[:top_n]):
            if w == word:
                return 'high_freq'

        for i, (w, _) in enumerate(sorted_words[-bottom_n:]):
            if w == word:
                return 'low_freq'

        return 'mid_freq'

    def evaluate_high_freq_words(self, corpus, top_n=20):
        """评估高频词"""
        word_counts = Counter()
        for sent in corpus:
            word_counts.update(sent)
        
        high_freq_words = [w for w, _ in word_counts.most_common(top_n * 2) if len(w) > 1][:top_n]

        results = {}
        for word in high_freq_words:
            if word in self.model.wv:
                results[word] = {'in_vocab': True, 'vector_available': True}
            else:
                results[word] = {'in_vocab': False, 'vector_available': False}
        return results

    def evaluate_low_freq_words(self, corpus, bottom_n=30):
        """评估低频词"""
        word_counts = Counter()
        for sent in corpus:
            word_counts.update(sent)

        all_words = word_counts.most_common()
        low_freq_words = [w for w, c in all_words if c <= 3 and len(w) > 1][:bottom_n]

        results = {}
        for word in low_freq_words:
            if word in self.model.wv:
                results[word] = {'in_vocab': True, 'vector_available': True}
            else:
                results[word] = {'in_vocab': False, 'vector_available': False}
        return results

    def evaluate_oov_words(self, test_words, train_vocab):
        """评估OOV词的处理"""
        oov_results = {}

        for word in test_words:
            if word not in train_vocab:
                oov_results[word] = {
                    'status': 'OOV',
                    'note': 'Word not in vocabulary'
                }

        return oov_results

    def compute_similarity_coherence(self, word_pairs):
        """计算词对覆盖率（简化版，不计算相似度）"""
        scores = []
        for w1, w2, expected_sim in word_pairs:
            try:
                # 只检查词是否在词表中，不计算相似度
                if w1 in self.model.wv and w2 in self.model.wv:
                    scores.append({
                        'word1': w1,
                        'word2': w2,
                        'in_vocab': True
                    })
            except KeyError:
                continue

        if not scores:
            return None

        coverage_rate = len(scores) / len(word_pairs)

        return {
            'coverage_rate': coverage_rate,
            'details': scores
        }

    def compute_vocab_coverage(self, test_corpus):
        """计算词表覆盖率（定量指标）"""
        oov_count = 0
        total_count = 0

        for sent in test_corpus:
            for word in sent:
                total_count += 1
                if word not in self.model.wv:
                    oov_count += 1

        coverage = 1 - (oov_count / max(1, total_count))

        return {
            'oov_count': oov_count,
            'total_count': total_count,
            'coverage': coverage,
            'oov_rate': oov_count / max(1, total_count)
        }

    def analyze_semantic_boundary_preservation(self, corpus, semantic_words, topn=10):
        """
        研究功能1：语义边界保持性
        评估分词是否破坏了语义完整词的表示
        
        语义完整词（如专有名词、成语、固定搭配）在分词后可能被切分，
        导致原本完整的语义信息丢失。通过比较完整词与其被切分后的
        子词/字符的向量表示，可以评估分词对语义的保持程度。
        
        Args:
            corpus: 分词后的语料库
            semantic_words: 语义完整词列表（如 ['china', 'new york', 'united states', '美国总统'）
            topn: 相似词查询数量
            
        Returns:
            包含以下信息的字典：
            - intact_coverage: 完整词在词表中的覆盖率
            - subword_coverage: 子词/字符在词表中的覆盖率
            - semantic_similarity: 完整词与子词/字符组合的语义相似度
            - boundary_preservation_score: 边界保持性得分
        """
        from collections import Counter
        import numpy as np
        
        results = {
            'intact_coverage': 0.0,
            'subword_coverage': 0.0,
            'semantic_similarity': [],
            'boundary_preservation_score': 0.0,
            'details': []
        }
        
        word_counts = Counter()
        for sent in corpus:
            word_counts.update(sent)
        
        total_words = len(semantic_words)
        intact_in_vocab = 0
        subword_in_vocab = 0
        
        for word in semantic_words:
            word_info = {
                'word': word,
                'in_vocab': False,
                'subwords_in_vocab': [],
                'combined_vector': None,
                'similar_words': [],
                'similarity_score': None
            }
            
            if word in self.model.wv:
                intact_in_vocab += 1
                word_info['in_vocab'] = True
                
                try:
                    similar = self.model.wv.most_similar(word, topn=topn)
                    word_info['similar_words'] = [(w, float(s)) for w, s in similar]
                except:
                    word_info['similar_words'] = []
                
                if ' ' in word:
                    subwords = word.split()
                else:
                    subwords = list(word)
                
                subwords_in_vocab = []
                for sw in subwords:
                    if sw in self.model.wv:
                        subwords_in_vocab.append(sw)
                
                subword_in_vocab += len(subwords_in_vocab)
                word_info['subwords_in_vocab'] = subwords_in_vocab
                
                if len(subwords_in_vocab) > 1:
                    try:
                        subword_vectors = [self.model.wv[sw] for sw in subwords_in_vocab]
                        avg_subword_vector = np.mean(subword_vectors, axis=0)
                        
                        intact_vector = self.model.wv[word]
                        similarity = np.dot(intact_vector, avg_subword_vector) / (
                            np.linalg.norm(intact_vector) * np.linalg.norm(avg_subword_vector)
                        )
                        word_info['similarity_score'] = float(similarity)
                        results['semantic_similarity'].append(similarity)
                    except:
                        pass
            
            results['details'].append(word_info)
        
        results['intact_coverage'] = intact_in_vocab / total_words if total_words > 0 else 0.0
        results['subword_coverage'] = subword_in_vocab / total_words / 2 if total_words > 0 else 0.0
        
        if results['semantic_similarity']:
            results['boundary_preservation_score'] = np.mean(results['semantic_similarity'])
        else:
            results['boundary_preservation_score'] = 0.0
        
        return results

    def analyze_frequency_layered_degradation(self, corpus, top_n=20, bottom_n=20, topn_neighbors=10):
        """
        研究功能2：频率分层退化
        分析高频词与低频词在不同分词下的邻域差异
        
        高频词通常有更丰富的上下文，能学习到更好的向量表示；
        低频词由于上下文稀疏，向量表示可能不够鲁棒。
        此分析比较不同频率层次词的邻域质量差异。
        
        Args:
            corpus: 分词后的语料库
            top_n: 高频词数量
            bottom_n: 低频词数量
            topn_neighbors: 邻域查询数量
            
        Returns:
            包含以下信息的字典：
            - high_freq_stats: 高频词统计（邻域一致性、覆盖度等）
            - low_freq_stats: 低频词统计
            - degradation_ratio: 退化比（低频/高频邻域质量）
            - frequency_gap: 频率层差距
        """
        from collections import Counter
        import numpy as np
        
        results = {
            'high_freq_words': [],
            'low_freq_words': [],
            'high_freq_stats': {},
            'low_freq_stats': {},
            'degradation_ratio': 0.0,
            'frequency_gap': 0.0,
            'details': []
        }
        
        word_counts = Counter()
        for sent in corpus:
            word_counts.update(sent)
        
        sorted_words = word_counts.most_common()
        total_words = len(sorted_words)
        
        high_freq_threshold = int(total_words * 0.1)
        high_freq_candidates = [w for w, c in sorted_words[:high_freq_threshold] if len(w) > 1][:top_n]
        low_freq_candidates = [w for w, c in sorted_words if c <= 3 and len(w) > 1][-bottom_n:]
        
        high_freq_valid = [w for w in high_freq_candidates if w in self.model.wv]
        low_freq_valid = [w for w in low_freq_candidates if w in self.model.wv]
        
        results['high_freq_words'] = high_freq_valid[:top_n]
        results['low_freq_words'] = low_freq_valid[-bottom_n:]
        
        high_freq_neighbors = []
        high_freq_in_vocab_rate = len(high_freq_valid) / len(high_freq_candidates) if high_freq_candidates else 0.0
        
        for word in high_freq_valid:
            try:
                neighbors = self.model.wv.most_similar(word, topn=topn_neighbors)
                neighbor_words = [n[0] for n in neighbors]
                high_freq_neighbors.append(neighbor_words)
                
                neighbor_context_overlap = 0
                for sent in corpus:
                    if word in sent:
                        context_words = set(sent) - {word}
                        neighbor_overlap = len(context_words & set(neighbor_words))
                        neighbor_context_overlap += neighbor_overlap
                
                results['details'].append({
                    'word': word,
                    'frequency_category': 'high',
                    'neighbors': neighbor_words[:5],
                    'context_overlap': neighbor_context_overlap
                })
            except:
                continue
        
        low_freq_neighbors = []
        low_freq_in_vocab_rate = len(low_freq_valid) / len(low_freq_candidates) if low_freq_candidates else 0.0
        
        for word in low_freq_valid:
            try:
                neighbors = self.model.wv.most_similar(word, topn=topn_neighbors)
                neighbor_words = [n[0] for n in neighbors]
                low_freq_neighbors.append(neighbor_words)
                
                neighbor_context_overlap = 0
                for sent in corpus:
                    if word in sent:
                        context_words = set(sent) - {word}
                        neighbor_overlap = len(context_words & set(neighbor_words))
                        neighbor_context_overlap += neighbor_overlap
                
                results['details'].append({
                    'word': word,
                    'frequency_category': 'low',
                    'neighbors': neighbor_words[:5],
                    'context_overlap': neighbor_context_overlap
                })
            except:
                continue
        
        avg_high_freq_neighbors = len(high_freq_neighbors) / len(high_freq_valid) if high_freq_valid else 0.0
        avg_low_freq_neighbors = len(low_freq_neighbors) / len(low_freq_valid) if low_freq_valid else 0.0
        
        results['high_freq_stats'] = {
            'valid_count': len(high_freq_valid),
            'total_count': len(high_freq_candidates),
            'in_vocab_rate': high_freq_in_vocab_rate,
            'avg_neighbors_found': avg_high_freq_neighbors,
            'total_neighbors': sum(len(n) for n in high_freq_neighbors)
        }
        
        results['low_freq_stats'] = {
            'valid_count': len(low_freq_valid),
            'total_count': len(low_freq_candidates),
            'in_vocab_rate': low_freq_in_vocab_rate,
            'avg_neighbors_found': avg_low_freq_neighbors,
            'total_neighbors': sum(len(n) for n in low_freq_neighbors)
        }
        
        if avg_high_freq_neighbors > 0:
            results['degradation_ratio'] = avg_low_freq_neighbors / avg_high_freq_neighbors
        else:
            results['degradation_ratio'] = 0.0
        
        if high_freq_in_vocab_rate > 0:
            results['frequency_gap'] = high_freq_in_vocab_rate - low_freq_in_vocab_rate
        else:
            results['frequency_gap'] = 0.0
        
        return results
