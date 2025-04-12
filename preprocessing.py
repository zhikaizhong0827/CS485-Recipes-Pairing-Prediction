import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
import itertools
import spacy
import sys

# ====================== 1. 下载数据集 ======================
import kagglehub
path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
print("Path to dataset files:", path)

# ====================== 2. 读取 CSV 文件 ======================
csv_file = os.path.join(path, "recipes.csv")
df = pd.read_csv(csv_file)
df = df.dropna(subset=["RecipeIngredientParts"])

# ====================== 3. 解析食材字符串 ======================
def parse_ingredients(text):
    text = text.strip()
    # 检查 R 中空向量的表示
    if text == "character(0)" or "character(0" in text:
        return []
    # 如果文本以 c( 开头和 ) 结尾，去除前缀和后缀
    if text.startswith("c("):
        text = text[2:]
    if text.endswith(")"):
        text = text[:-1]
    # 包裹成合法的 Python 列表形式
    literal_text = "[" + text + "]"
    try:
        result = ast.literal_eval(literal_text)
    except Exception as e:
        print("Failed parsing:", literal_text)
        raise e
    return result

df["ingredient_list"] = df["RecipeIngredientParts"].apply(parse_ingredients)
print(df["ingredient_list"].head())
output_file = "ingredient_list.csv"
df["ingredient_list"].to_csv(output_file, index=False)
print(f"{output_file} Updated.")

# ====================== 4. 初始化 spaCy======================
# 使用 spaCy 进行食材归一化处理， 比如
# use example of the first 100 rows for testing
#df = df.head(100).copy()
if spacy.prefer_gpu():
    print("Using GPU for spaCy processing.")
else:
    print("GPU not available, using CPU.")
nlp = spacy.load("en_core_web_sm")

all_ingredients = [ing for ing_list in df["ingredient_list"] for ing in ing_list]
unique_ingredients = list(set([ing.lower() for ing in all_ingredients]))
print(f"Total unique ingredients to process with pipe: {len(unique_ingredients)}")

# 使用 nlp.pipe 批处理所有独特食材，并构造映射字典：原始食材 -> 归一化后的结果
norm_map = {}
batch_size = 1000  # 可根据内存情况调整
for ing, doc in zip(unique_ingredients, nlp.pipe(unique_ingredients, batch_size=batch_size)):
    # 处理时转换为小写、去除前后空格已经在前端保证，直接做 lemma 过滤
    tokens = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    normalized = " ".join(tokens) if tokens else ing
    norm_map[ing] = normalized

# 更新 DataFrame 中每个配方的 ingredient_list 使用 norm_map 映射
df["ingredient_list"] = df["ingredient_list"].apply(
    lambda ing_list: [norm_map.get(ing.lower(), ing.lower()) for ing in ing_list]
)

# -------------------------------------------------------------
# 添加逻辑：如果较长的食材名称包含较短的食材名称，则将其替换为较短的那一个

def refine_ingredient_mapping(ingredients):
    mapping = {}
    for ing in ingredients:
        # 找出所有作为 ing 的子串的食材（不包括自身）
        candidates = [cand for cand in ingredients if cand != ing and cand in ing]
        if candidates:
            # 取候选中长度最小的一个
            mapping[ing] = min(candidates, key=len)
        else:
            mapping[ing] = ing
    return mapping

# 收集所有归一化后的独立食材
unique_ingredients = set()
for ing_list in df["ingredient_list"]:
    # 可以保证 ing_list 元素已经小写，但这里额外调用 lower() 以防万一
    unique_ingredients.update([ing.lower() for ing in ing_list])

# 构造映射字典
mapping = refine_ingredient_mapping(unique_ingredients)

# 用该映射更新每个配方的 ingredient_list
df["ingredient_list"] = df["ingredient_list"].apply(
    lambda ing_list: [mapping.get(ing, ing) for ing in ing_list]
)

# ====================== 5. 构建食材共现图 ======================
co_occurrence = defaultdict(int)

for ingredient_list in df["ingredient_list"]:
    # 去重，避免同一配方内重复计数
    unique_ingredients = set(ingredient_list)
    # 获取所有两两组合
    for ing_a, ing_b in itertools.combinations(sorted(unique_ingredients), 2):
        co_occurrence[(ing_a, ing_b)] += 1

G = nx.Graph()
for (ing_a, ing_b), weight in co_occurrence.items():
    G.add_edge(ing_a, ing_b, weight=weight)
nx.draw(G, cmap = plt.get_cmap('jet'))
plt.show()
adj_df = nx.to_pandas_adjacency(G)
print(adj_df)
print("图中节点数：", G.number_of_nodes())
print("图中边数：", G.number_of_edges())

# ====================== 6. 保存图 ======================
# 保存为 pickle 文件（后续可直接用 networkx 加载）
#gpickle.write_gpickle(G, "ingredient_cooccur_graph.gpickle")
#print("图已保存为 ingredient_cooccur_graph.gpickle")

# 将图转换为边列表，保存为 CSV 文件
edge_list = [(u, v, data["weight"]) for u, v, data in G.edges(data=True)]
edge_df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
output_file = "ingredient_cooccur_graph.csv"
edge_df.to_csv("ingredient_cooccur_graph.csv", index=False)
print(f"{output_file} Updated.")
    
node_list = list(G.nodes())

# 创建一个 DataFrame，单列保存节点名称
node_df = pd.DataFrame(node_list, columns=["node"])

# 保存文件名，可以是 CSV 或 TXT，这里使用 CSV
node_output_file = "unique_nodes.csv"
node_df.to_csv(node_output_file, index=False)
print(f"{node_output_file} Updated.")
