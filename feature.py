import pandas as pd
import numpy as np
import gensim.downloader as api

# 1. 读取节点 CSV 文件，假设文件中有一列 "node"
node_df = pd.read_csv("unique_nodes.csv")
print("节点样例：")
print(node_df.head())

# 2. 加载预训练的 fastText 模型（300维）
print("加载 fastText 模型...")
model = api.load("fasttext-wiki-news-subwords-300")
print("fastText 模型加载完成。")

# 3. 定义一个函数，为一个节点生成特征向量
def get_feature_vector(text):
    # 转为小写并去除首尾空格
    text = text.lower().strip()
    # 如果完整的文本在模型中，直接返回对应向量
    if text in model:
        return model[text]
    # 否则，尝试将文本分割为多个单词，并计算均值
    tokens = text.split()
    token_vectors = [model[token] for token in tokens if token in model]
    if token_vectors:
        return np.mean(token_vectors, axis=0)
    else:
        # 如果所有分词都不在模型中，用较小随机值初始化向量（印度语模型的词向量）
        # 这里可以根据实际情况选择其他初始化方式，例如全零向量或随机小值向量
        return np.random.uniform(-0.1, 0.1, model.vector_size)

# 4. 对每个节点生成特征向量，并将结果存到 DataFrame 的新列中
node_df["feature_vector"] = node_df["node"].apply(get_feature_vector)

# 5. 构造特征矩阵：每行对应一个节点的特征向量
feature_matrix = np.stack(node_df["feature_vector"].values)
print("特征矩阵形状：", feature_matrix.shape)

# 保存特征矩阵为 CSV 文件或 NumPy 二进制文件便于后续使用
df_feature = pd.DataFrame(feature_matrix)
df_feature.to_csv("feature_matrix.csv", index=False)
print("特征矩阵已保存为 feature_matrix.csv")
