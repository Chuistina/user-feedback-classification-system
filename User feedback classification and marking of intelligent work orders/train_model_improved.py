import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
from collections import Counter

def clean_and_balance_data(input_file):
    """清洗和平衡数据集"""
    df = pd.read_csv(input_file)
    
    # 分解多标签，创建单一标签数据
    single_label_data = []
    
    for _, row in df.iterrows():
        categories = [cat.strip() for cat in row['category'].split(',')]
        
        # 为每个类别创建一个数据条目，优先选择主要类别
        primary_category = None
        
        # 定义优先级：具体问题类型 > Bug反馈 > 功能建议 > 其他
        priority_order = ['账号问题', '网络连接失败', '文档问题', '性能问题', 'Bug反馈', '功能建议', '其他']
        
        for priority_cat in priority_order:
            if priority_cat in categories:
                primary_category = priority_cat
                break
        
        if primary_category:
            single_label_data.append({
                'id': row['id'],
                'title': row['title'],
                'body': row['body'],
                'category': primary_category,
                'urgency': row['urgency'],
                'html_url': row['html_url'],
                'cleaned_title': row['cleaned_title'],
                'cleaned_body': row['cleaned_body'],
                'text_for_classification': row['text_for_classification']
            })
    
    # 创建新的DataFrame
    balanced_df = pd.DataFrame(single_label_data)
    
    print("重新平衡后的类别分布:")
    print(balanced_df['category'].value_counts())
    
    return balanced_df

def train_and_evaluate_model(input_file, model_dir="models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 清洗和平衡数据
    df = clean_and_balance_data(input_file)
    
    # 过滤掉样本数量太少的类别
    category_counts = df["category"].value_counts()
    print(f"\n原始类别计数: {category_counts}")
    
    min_samples = 5  # 至少需要5个样本
    valid_categories = category_counts[category_counts >= min_samples].index
    df = df[df["category"].isin(valid_categories)]
    
    print(f"过滤后保留的类别: {valid_categories.tolist()}")
    print(f"过滤后的类别分布:")
    print(df["category"].value_counts())

    if len(df) < 10:
        print("数据量太少，无法训练模型")
        return

    X = df["text_for_classification"]
    y = df["category"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # 使用1-2gram提高特征表达
        min_df=2,           # 忽略出现次数少于2的词
        max_df=0.95        # 忽略出现频率高于95%的词
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 训练逻辑回归模型
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # 自动平衡类别权重
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    # 评估模型
    y_pred = model.predict(X_test_tfidf)
    
    print("\n=== 模型评估结果 ===")
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 显示特征重要性
    print("\n=== 各类别的关键特征词 ===")
    feature_names = vectorizer.get_feature_names_out()
    
    for i, category in enumerate(model.classes_):
        # 获取该类别的权重最高的前10个特征
        weights = model.coef_[i]
        top_indices = weights.argsort()[-10:][::-1]
        top_features = [feature_names[idx] for idx in top_indices]
        print(f"{category}: {', '.join(top_features)}")

    # 保存模型和向量化器
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(model_dir, "logistic_regression_model.pkl"))
    
    print(f"\n模型和向量化器已保存到 {model_dir}")
    
    # 测试一些样例
    print("\n=== 测试样例 ===")
    test_samples = [
        "登录时出现错误，无法访问我的账户",
        "希望能添加一个新的搜索功能",
        "网站连接超时，无法加载页面", 
        "文档中的示例代码有错误",
        "系统运行很慢，响应时间过长",
        "这是一个普通的bug报告"
    ]
    
    for sample in test_samples:
        sample_tfidf = vectorizer.transform([sample])
        prediction = model.predict(sample_tfidf)[0]
        confidence = max(model.predict_proba(sample_tfidf)[0])
        print(f"输入: {sample}")
        print(f"预测: {prediction} (置信度: {confidence:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    input_path = "data/labeling/label_task.csv"
    train_and_evaluate_model(input_path)

