import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 加载数据集
df = pd.read_csv(r'C:\Users\Lenovo\.ssh\xiaoxueqi-666\nigerian-songs.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))

# 提取数值型特征
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 数据预处理 - 标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

# 将标准化后的数据转换回 DataFrame 以便可视化
df_scaled_df = pd.DataFrame(df_scaled, columns=numeric_cols)

# 创建画布，包含两个子图
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 绘制标准化前数值型特征的箱线图
df[numeric_cols].boxplot(ax=axes[0])
axes[0].set_title('标准化前数值型特征箱线图')
axes[0].set_ylabel('特征值')

# 绘制标准化后数值型特征的箱线图
df_scaled_df.boxplot(ax=axes[1])
axes[1].set_title('标准化后数值型特征箱线图')
axes[1].set_ylabel('特征值')

plt.show()

# 分解降维 - 使用 PCA 降维到 2 维
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# 聚类分析 - 使用 K-Means 聚类
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_pca)
    labels = kmeans.labels_
    score = silhouette_score(df_pca, labels)
    silhouette_scores.append(score)

# 找到最优的簇数量
best_k = silhouette_scores.index(max(silhouette_scores)) + 2

# 使用最优簇数量进行 K-Means 聚类
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(df_pca)
df['Cluster'] = kmeans.labels_

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 可视化聚类结果
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('主成分1')
plt.xticks(rotation=45)
plt.ylabel('主成分2')
plt.title('K-Means 聚类结果')

plt.show()

print('最优簇数量：', best_k)