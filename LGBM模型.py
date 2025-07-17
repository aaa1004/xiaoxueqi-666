import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
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

# 二、价格分析
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei']

price_describe = df[['Low Price', 'High Price', 'Mostly Low', 'Mostly High']].describe().round(2)
df[['Low Price', 'High Price', 'Mostly Low', 'Mostly High']].plot.box()
plt.title('南瓜价格箱线图')
plt.ylabel('价格')

print('价格分布情况：')
print(price_describe)
plt.show()

# 三、城市分析
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

# 四、包装分析
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

# 五、颜色分析
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

# ======================
# 特征选择详细分析与LGBM建模（修正后）
# ======================

# 选择特征和目标变量
feature_cols = ['Low Price', 'High Price', 'Mostly Low', 'Mostly High']
X = df[feature_cols]
y = df['Low Price']

# 填充缺失值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 特征选择：基于f_regression计算得分
selector = SelectKBest(score_func=f_regression, k=2)
X_new = selector.fit_transform(X_imputed, y)

# 提取所有特征的得分并可视化
feature_scores = selector.scores_
score_df = pd.DataFrame({
    '特征名称': feature_cols,
    'f_regression得分': feature_scores.round(4)
}).sort_values(by='f_regression得分', ascending=False)

# 可视化特征得分
plt.figure(figsize=(10, 6))
bars = plt.bar(score_df['特征名称'], score_df['f_regression得分'], color='skyblue')
plt.title('各特征的f_regression得分（得分越高，与目标变量相关性越强）', fontsize=12)
plt.ylabel('f_regression得分')
plt.xlabel('特征名称')
plt.xticks(rotation=45)

# 标注得分
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# 输出特征得分详情
print('\n===== 特征选择依据：各特征的f_regression得分（降序）=====')
print(score_df)

# 获取被选中的特征名称
selected_mask = selector.get_support()
selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
print('\n被选中的特征（得分最高的2个）：', selected_feature_names)

# 将X_new转换为DataFrame（添加特征名称，解决LGBM警告）
X_new_df = pd.DataFrame(X_new, columns=selected_feature_names)

# 划分训练集和测试集（使用带特征名称的DataFrame）
X_train, X_test, y_train, y_test = train_test_split(X_new_df, y, test_size=0.2, random_state=42)

# LGBM模型训练与评估
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 评估指标（修正：使用round()函数处理float值）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\n===== LGBM模型评估结果 =====')
print('均方误差（MSE）：', round(mse, 4))  # 修正：float无round方法，用内置round函数
print('决定系数（R²）：', round(r2, 4))    # 同上

# 特征重要性
print('\nLGBM特征重要性：')
for name, importance in zip(selected_feature_names, model.feature_importances_):
    print(f'{name}: {importance:.4f}')