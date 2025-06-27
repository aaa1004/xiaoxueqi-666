# xiaoxueqi-666
一、项目概述
本项目围绕尼日利亚歌曲数据集（nigerian-songs.csv）展开数据分析，运用数据预处理、降维、聚类等技术，挖掘歌曲数据潜在特征与分类模式，助力理解歌曲数据分布及内在关联 。
二、数据集介绍
（一）数据来源
数据集为 nigerian-songs.csv，存储于项目目录内，记录尼日利亚歌曲相关信息，可用于音乐特征分析、流派聚类等研究。
（二）数据字段
name：歌曲名称，字符型，标识每首歌的独特称谓。
album：专辑名称，字符型，说明歌曲所属专辑。
artist：歌手姓名，字符型，记录演唱歌手信息。
artist_top_genre：歌手主要流派，字符型，体现歌手音乐风格倾向。
release_date：发行年份，整型，反映歌曲发布时间。
length：歌曲时长（毫秒），整型，衡量歌曲长度。
popularity：流行度，整型，数值越高表示越受关注。
danceability：舞蹈性，浮点型，衡量歌曲适合跳舞的程度。
acousticness： acoustics 特性，浮点型，体现歌曲原声乐器占比等声学特征。
energy：能量感，浮点型，描述歌曲的活力、强度。
instrumentalness：乐器占比，浮点型，指歌曲中无 vocals 部分的占比。
liveness：现场感，浮点型，判断歌曲是否为现场录制的指标。
loudness：响度，浮点型，衡量歌曲声音大小。
speechiness：语音占比，浮点型，检测歌曲中语音内容的比例。
tempo：节奏（BPM），浮点型，反映歌曲节奏快慢。
time_signature：节拍类型，整型，表示歌曲节拍结构 。
三、代码功能与流程
（一）环境依赖
运行代码需安装以下 Python 库：

pandas：用于数据加载、处理与分析，版本建议 >=1.3.0 。
scikit-learn（sklearn）：提供数据预处理、降维、聚类及评估功能，版本建议 >=1.0.0 。
matplotlib：实现数据可视化，版本建议 >=3.4.0 。

可通过以下命令安装依赖（需确保已配置 Python 环境）：

bash
pip install pandas scikit-learn matplotlib
（二）代码结构与功能
数据加载与基本信息查看

python
运行
import pandas as pd
# 加载数据集，指定本地文件路径
df = pd.read_csv(r'C:\Users\Lenovo\.ssh\xiaoxueqi-666\nigerian-songs.csv')  
print('数据基本信息：')
df.info()  # 输出数据框的基本信息，包括列名、非空值数量、数据类型等

rows, columns = df.shape  # 获取数据集行数和列数
if rows < 100 and columns < 20:
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))  # 若数据量小，打印全量数据（制表符分隔，缺失值用 'nan' 表示）
else:
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))  # 数据量大时，打印前 5 行预览

功能：加载数据集，输出数据基本结构（行列数、数据类型、非空值情况等），并根据数据规模决定显示全量还是前几行数据，帮助快速了解数据概况 。

数据预处理 - 标准化

python
运行
from sklearn.preprocessing import StandardScaler
# 提取数值型特征列（int64 和 float64 类型）
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns  
scaler = StandardScaler()  # 初始化标准化器，将特征值转换为均值为 0、标准差为 1 的分布
df_scaled = scaler.fit_transform(df[numeric_cols])  # 对数值型特征进行标准化处理
df_scaled_df = pd.DataFrame(df_scaled, columns=numeric_cols)  # 转换为 DataFrame 以便后续可视化

功能：筛选数值型特征，消除不同特征量纲差异。标准化后，各特征在后续降维、聚类等操作中权重更合理，避免因数值范围差异影响结果 。

预处理结果可视化（箱线图）

python
运行
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 创建 1 行 2 列的画布，设置尺寸

# 绘制标准化前数值型特征箱线图
df[numeric_cols].boxplot(ax=axes[0])  
axes[0].set_title('标准化前数值型特征箱线图')  # 设置标题
axes[0].set_ylabel('特征值')  # 设置 y 轴标签

# 绘制标准化后数值型特征箱线图
df_scaled_df.boxplot(ax=axes[1])  
axes[1].set_title('标准化后数值型特征箱线图')
axes[1].set_ylabel('特征值')

plt.show()  # 显示箱线图

功能：通过箱线图对比标准化前后数值型特征的分布。箱线图可展示数据的中位数、四分位数、异常值等，直观呈现标准化对数据离散程度、异常值影响，辅助判断预处理效果 。

分解降维 - PCA（主成分分析）

python
运行
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  # 初始化 PCA，将数据降维到 2 维，便于后续可视化与聚类
df_pca = pca.fit_transform(df_scaled)  # 对标准化后的数值型特征进行降维

功能：降低数据维度，把高维数值特征压缩到 2 维。在保留数据主要变异信息的同时，简化数据结构，方便后续聚类结果可视化展示 。

聚类分析 - K-Means

python
运行
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []  # 存储不同簇数下的轮廓系数
for k in range(2, 11):  # 尝试簇数 2 到 10
    kmeans = KMeans(n_clusters=k, random_state=42)  # 初始化 K-Means 模型，设置随机种子保证可复现
    kmeans.fit(df_pca)  # 在降维后的数据上拟合模型
    labels = kmeans.labels_  # 获取聚类标签
    score = silhouette_score(df_pca, labels)  # 计算轮廓系数，评估聚类效果（值越接近 1 效果越好）
    silhouette_scores.append(score)

best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 找到最优簇数（轮廓系数最大对应的簇数）
kmeans = KMeans(n_clusters=best_k, random_state=42)  # 用最优簇数初始化模型
kmeans.fit(df_pca)
df['Cluster'] = kmeans.labels_  # 将聚类标签添加到原数据框

功能：采用 K-Means 算法对降维后数据聚类。通过轮廓系数评估不同簇数的聚类质量，筛选最优簇数，实现对歌曲数据的分类，挖掘潜在群体结构 。

聚类结果可视化

python
运行
plt.rcParams['figure.dpi'] = 300  # 设置图片清晰度（每英寸点数）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Cluster'], cmap='viridis')  # 绘制散点图，用颜色区分簇
plt.xlabel('主成分1')  # 设置 x 轴标签
plt.xticks(rotation=45)  # 设置 x 轴刻度旋转，避免重叠
plt.ylabel('主成分2')  # 设置 y 轴标签
plt.title('K-Means 聚类结果')  # 设置标题
plt.show()  # 显示聚类结果图

print('最优簇数量：', best_k)  # 输出最优簇数

功能：以散点图形式展示 K-Means 聚类结果，降维后的两个主成分分别作为横、纵轴，不同颜色代表不同聚类簇。直观呈现歌曲数据在二维空间的聚类分布，辅助理解分类情况 。
四、运行说明
（一）本地运行步骤
准备环境：确保已安装 Python（建议 >=3.7 版本 ），并通过 pip 安装好 pandas、scikit-learn、matplotlib 库（参考前文 “环境依赖” 部分命令）。
放置文件：将 nigerian-songs.csv 数据集文件与代码文件（如 music_analysis.py，需把上述代码整合到一个 .py 文件 ）放在同一目录（或按代码中 pd.read_csv 的路径设置存放）。
执行代码：打开终端（或命令提示符），进入代码所在目录，执行以下命令：

bash
python music_analysis.py

代码会依次执行数据加载、预处理、可视化、降维、聚类等流程，弹出箱线图、聚类结果散点图，并在终端输出数据基本信息、最优簇数等内容 。
（二）常见问题与解决
文件路径错误：若运行时报 FileNotFoundError，检查 pd.read_csv 中文件路径是否正确，确保数据集文件位置与代码设置一致 。
字体显示异常：若中文标题、标签显示为方框或乱码，确认 matplotlib 字体设置（plt.rcParams['font.sans-serif'] = ['SimHei'] ），且系统已安装对应字体（如 Windows 系统一般默认有 “黑体” ）。
库版本冲突：若运行报错与库版本相关，尝试更新库到较新版本（如 pip install --upgrade pandas scikit-learn matplotlib ），或根据报错提示调整版本适配 。
五、分析结论与拓展
（一）当前分析结论
数据预处理：标准化有效调整数值型特征分布，从箱线图可观察到，标准化后各特征离散程度更均匀，异常值影响相对降低，为后续分析奠定基础 。
聚类效果：通过轮廓系数筛选出最优簇数（代码中为 best_k ，实际运行因数据分布可能不同 ），K-Means 聚类将歌曲数据分成对应簇。聚类结果散点图呈现不同簇在二维主成分空间的分布，反映歌曲特征的群体差异，可进一步结合 artist_top_genre 等字段，分析不同簇对应的音乐风格、流行度特点 。
（二）拓展方向
特征深入挖掘：除数值型特征，可对 artist_top_genre 等类别型特征编码（如独热编码），纳入分析，丰富聚类依据 。
聚类算法对比：尝试 DBSCAN、层次聚类 等其他算法，对比不同算法在该数据集上的聚类效果，选择更优方案 。
业务关联分析：结合音乐平台实际场景，分析聚类结果与歌曲播放量、收藏量等指标关联，为音乐推荐、流派研究提供更具价值结论 。

通过本项目，可掌握从数据加载、预处理到降维聚类及可视化的完整分析流程，后续可基于实际需求持续拓展优化，深入挖掘音乐数据价值 。