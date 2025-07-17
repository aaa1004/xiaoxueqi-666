import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
import graphviz
import os

# 加载数据集
df = pd.read_csv(r'C:\Users\Lenovo\.ssh\xiaoxueqi-666\US-pumpkins.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape
if rows < 100 and columns < 20:
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))

# 二、价格分析（保持不变）
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

price_describe = df[['Low Price', 'High Price', 'Mostly Low', 'Mostly High']].describe().round(2)
df[['Low Price', 'High Price', 'Mostly Low', 'Mostly High']].plot.box()
plt.title('南瓜价格箱线图')
plt.ylabel('价格')
print('价格分布情况：')
print(price_describe)
plt.show()

# 三、城市分析（保持不变）
city_counts = df['City Name'].value_counts()
top_10_cities = city_counts.nlargest(10, keep='all')
city_mean_price = df.groupby('City Name')[['Low Price', 'High Price']].mean()
top_10_cities_mean_price = city_mean_price.loc[top_10_cities.index]

plt.figure(figsize=(10, 6))
ax1 = plt.subplot(211)
bars = top_10_cities.plot(kind='bar', ax=ax1)
plt.title('交易次数最多的前十个城市')
plt.ylabel('交易次数')
for bar in bars.patches:
    height = bar.get_height()
    ax1.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

ax2 = plt.subplot(212)
top_10_cities_mean_price.plot(kind='line', marker='o', ax=ax2)
plt.title('交易次数最多的前十个城市的平均价格')
plt.ylabel('平均价格')
plt.xlabel('城市名称')
plt.xticks(rotation=45)
plt.tight_layout()
print('交易次数最多的前十个城市：')
print(top_10_cities)
print('交易次数最多的前十个城市的平均价格：')
print(top_10_cities_mean_price.round(2))
plt.show()

# 四、包装分析（保持不变）
package_counts = df['Package'].value_counts()
top_10_packages = package_counts.nlargest(10, keep='all')
package_mean_price = df.groupby('Package')[['Low Price', 'High Price']].mean()
top_10_packages_mean_price = package_mean_price.loc[top_10_packages.index]

plt.figure(figsize=(10, 6))
ax1 = plt.subplot(211)
bars = top_10_packages.plot(kind='bar', ax=ax1)
plt.title('交易次数最多的前十个包装方式')
plt.ylabel('交易次数')
for bar in bars.patches:
    height = bar.get_height()
    ax1.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

ax2 = plt.subplot(212)
top_10_packages_mean_price.plot(kind='line', marker='o', ax=ax2)
plt.title('交易次数最多的前十个包装方式的平均价格')
plt.ylabel('平均价格')
plt.xlabel('包装方式')
plt.xticks(rotation=45)
plt.tight_layout()
print('交易次数最多的前十个包装方式：')
print(top_10_packages)
print('交易次数最多的前十个包装方式的平均价格：')
print(top_10_packages_mean_price.round(2))
plt.show()

# 五、颜色分析（保持不变）
color_counts = df['Color'].value_counts()
color_mean_price = df.groupby('Color')[['Low Price', 'High Price']].mean()

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(121)
bars = color_counts.plot(kind='bar', ax=ax1)
plt.title('不同颜色的交易次数')
plt.ylabel('交易次数')
for bar in bars.patches:
    height = bar.get_height()
    ax1.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

ax2 = plt.subplot(122)
color_mean_price.plot(kind='line', marker='o', ax=ax2)
plt.title('不同颜色的平均价格')
plt.ylabel('平均价格')
plt.xlabel('颜色')
plt.xticks(rotation=45)
print('不同颜色的交易次数：')
print(color_counts)
print('不同颜色的平均价格：')
print(color_mean_price.round(2))
plt.tight_layout()
plt.show()

# 六、使用分类特征建模（核心修改部分）
# 1. 选择特征和目标变量（改为三个分类特征）
feature_cols = ['City Name', 'Package', 'Color']
X = df[feature_cols].copy()  # 分类特征
y = df['Low Price']  # 目标变量不变

# 2. 处理低频类别（避免维度爆炸）
def merge_low_frequency_categories(df, col, threshold=10):
    """将出现次数少于threshold的类别合并为'其他'"""
    counts = df[col].value_counts()
    low_freq_categories = counts[counts < threshold].index
    df[col] = df[col].replace(low_freq_categories, '其他')
    return df

# 对城市和包装合并低频类别（颜色类别较少，不合并）
X = merge_low_frequency_categories(X, 'City Name', threshold=20)  # 城市：低频合并为'其他'
X = merge_low_frequency_categories(X, 'Package', threshold=15)    # 包装：低频合并为'其他'

# 3. 处理缺失值（分类特征用众数填充）
imputer = SimpleImputer(strategy='most_frequent')  # 众数填充分类特征
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=feature_cols)  # 转回DataFrame保留列名

# 4. 分类特征编码（独热编码）
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'避免多重共线性
X_encoded = encoder.fit_transform(X_imputed)
encoded_feature_names = encoder.get_feature_names_out(feature_cols)  # 获取编码后的特征名

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 6. 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('\n===== 分类特征建模评估结果 =====')
print(f'使用的特征：{feature_cols}（已编码为{len(encoded_feature_names)}个特征）')
print(f'均方误差（MSE）：{mse:.4f}')
print(f'决定系数（R²）：{r2:.4f}')

# 8. 查看重要特征（前10个）
feature_importance = pd.DataFrame({
    '特征名称': encoded_feature_names,
    '重要性': model.feature_importances_
}).sort_values(by='重要性', ascending=False)

print('\n===== 前10个重要特征 =====')
print(feature_importance.head(10))

# 9. 决策树可视化（选第一棵树，限制深度）
selected_tree = model.estimators_[0]
dot_data = export_graphviz(
    selected_tree,
    out_file=None,
    feature_names=encoded_feature_names,
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=3  # 限制深度，避免图过大
)

# 显示或保存决策树
try:
    from IPython.display import display
    graph = graphviz.Source(dot_data)
    display(graph)
    print("\n决策树已在当前界面显示")
except ImportError:
    graph = graphviz.Source(dot_data)
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    save_path = os.path.join(current_dir, 'random_forest_tree_categorical.pdf')
    graph.render('random_forest_tree_categorical', format='pdf', cleanup=True)
    print(f"\n决策树图片已保存至：{save_path}")