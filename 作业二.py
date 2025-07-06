import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 数据加载
def load_data(file_path):
    """加载南瓜数据集"""
    data = pd.read_csv(r'C:\Users\Lenovo\.ssh\xiaoxueqi-666\US-pumpkins.csv')
    print(f"数据集大小: {data.shape}")
    return data


# 2. 数据预处理
def preprocess_data(data):
    """预处理南瓜数据集"""
    # 复制数据以避免修改原始数据
    df = data.copy()

    # 查看缺失值
    print("缺失值统计:")
    print(df.isnull().sum())

    # 查看唯一值
    for col in df.columns:
        if df[col].nunique() < 20:
            print(f"{col} 唯一值: {df[col].unique()}")

    # 处理日期列
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

    # 处理价格列
    if 'Low Price' in df.columns and 'High Price' in df.columns:
        df['Average Price'] = (df['Low Price'] + df['High Price']) / 2

    # 填充缺失值
    # 分类变量用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 数值变量用中位数填充
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    return df


# 3. 数据可视化
def visualize_data(df):
    """创建南瓜数据集的可视化图表"""
    # 3.1 价格分布
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(df['Average Price'], kde=True, color='skyblue')
    plt.title('南瓜平均价格分布')
    plt.xlabel('平均价格')
    plt.ylabel('频次')

    # 3.2 月度价格趋势
    plt.subplot(2, 2, 2)
    monthly_price = df.groupby('Month')['Average Price'].mean().reset_index()
    sns.lineplot(x='Month', y='Average Price', data=monthly_price, marker='o', color='orange')
    plt.title('南瓜月度平均价格趋势')
    plt.xlabel('月份')
    plt.ylabel('平均价格')
    plt.xticks(monthly_price['Month'])

    # 3.3 不同包装的价格比较
    plt.subplot(2, 2, 3)
    package_price = df.groupby('Package')['Average Price'].mean().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(x='Average Price', y='Package', data=package_price, palette='viridis')
    plt.title('不同包装的南瓜平均价格（Top 10）')
    plt.xlabel('平均价格')
    plt.ylabel('包装')

    # 3.4 颜色与价格的关系
    plt.subplot(2, 2, 4)
    if 'Color' in df.columns:
        color_price = df.groupby('Color')['Average Price'].mean().sort_values(ascending=False).reset_index()
        sns.barplot(x='Color', y='Average Price', data=color_price, palette='coolwarm')
        plt.title('不同颜色南瓜的平均价格')
        plt.xlabel('颜色')
        plt.ylabel('平均价格')

    plt.tight_layout()
    plt.savefig('price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3.5 散点图矩阵（选择几个关键数值变量）
    plt.figure(figsize=(12, 10))
    numeric_vars = ['Low Price', 'High Price', 'Average Price', 'Mostly Low', 'Mostly High']
    numeric_vars = [var for var in numeric_vars if var in df.columns]
    sns.pairplot(df[numeric_vars], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'w'})
    plt.suptitle('南瓜价格相关变量散点图矩阵', y=1.02)
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3.6 箱线图：不同品种的价格分布
    plt.figure(figsize=(14, 8))
    if 'Variety' in df.columns and df['Variety'].nunique() < 20:
        sns.boxplot(x='Variety', y='Average Price', data=df, palette='Set3')
        plt.title('不同品种南瓜的价格分布')
        plt.xlabel('品种')
        plt.ylabel('平均价格')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('variety_price.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3.7 热力图：相关性分析
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title('南瓜数据集数值变量相关性热力图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3.8 产地与价格的关系（条形图）
    plt.figure(figsize=(14, 8))
    if 'Origin' in df.columns:
        origin_price = df.groupby('Origin')['Average Price'].mean().sort_values(ascending=False).head(15).reset_index()
        sns.barplot(x='Average Price', y='Origin', data=origin_price, palette='Blues_d')
        plt.title('不同产地南瓜的平均价格（Top 15）')
        plt.xlabel('平均价格')
        plt.ylabel('产地')
        plt.tight_layout()
        plt.savefig('origin_price.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3.9 价格的时间序列图
    plt.figure(figsize=(14, 7))
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
        df['Average Price'].resample('M').mean().plot(kind='line', marker='o', color='green')
        plt.title('南瓜平均价格的时间序列趋势')
        plt.xlabel('日期')
        plt.ylabel('平均价格')
        plt.tight_layout()
        plt.savefig('price_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        df.reset_index(inplace=True)  # 重置索引

    # 3.10 不同大小的南瓜价格比较
    plt.figure(figsize=(12, 8))
    if 'Item Size' in df.columns:
        size_order = ['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo']
        size_order = [size for size in size_order if size in df['Item Size'].unique()]
        sns.boxplot(x='Item Size', y='Average Price', data=df, order=size_order, palette='pastel')
        plt.title('不同大小南瓜的价格分布')
        plt.xlabel('南瓜大小')
        plt.ylabel('平均价格')
        plt.tight_layout()
        plt.savefig('size_price.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("可视化完成，图表已保存到当前目录")


# 4. 数据分析
def analyze_data(df):
    """分析南瓜数据集并生成报告"""
    report = {}

    # 4.1 基本统计信息
    report['基本统计'] = df[['Low Price', 'High Price', 'Average Price', 'Mostly Low', 'Mostly High']].describe()

    # 4.2 最常见的品种
    if 'Variety' in df.columns:
        report['最常见品种'] = df['Variety'].value_counts().head(5)

    # 4.3 价格最高的品种
    if 'Variety' in df.columns and 'Average Price' in df.columns:
        variety_price = df.groupby('Variety')['Average Price'].mean().sort_values(ascending=False).head(5)
        report['价格最高的品种'] = variety_price

    # 4.4 价格最高的产地
    if 'Origin' in df.columns and 'Average Price' in df.columns:
        origin_price = df.groupby('Origin')['Average Price'].mean().sort_values(ascending=False).head(5)
        report['价格最高的产地'] = origin_price

    # 4.5 价格趋势分析
    if 'Date' in df.columns and 'Average Price' in df.columns:
        df['Month'] = df['Date'].dt.month
        monthly_trend = df.groupby('Month')['Average Price'].mean()
        report['月度价格趋势'] = monthly_trend

    # 4.6 相关性分析
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        corr = df[numeric_cols].corr()
        report['相关性分析'] = corr

    # 打印分析报告
    print("\n数据分析报告:")
    for key, value in report.items():
        print(f"\n{key}:")
        print(value)

    return report


# 5. 主函数
def main():
    """主函数：执行整个南瓜数据集分析流程"""
    # 文件路径
    file_path = '/mnt/US-pumpkins.csv'

    # 加载数据
    print("正在加载数据...")
    data = load_data(file_path)

    # 数据预处理
    print("\n正在进行数据预处理...")
    processed_data = preprocess_data(data)

    # 数据可视化
    print("\n正在创建可视化图表...")
    visualize_data(processed_data)

    # 数据分析
    print("\n正在进行数据分析...")
    analysis_report = analyze_data(processed_data)

    print("\n南瓜数据集分析完成!")


if __name__ == "__main__":
    main()
