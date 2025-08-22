import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_chinese_model(input_file="data/labeling/chinese_dataset.csv", model_dir="models"):
    """使用中文数据训练模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 读取中文数据集
    df = pd.read_csv(input_file)
    
    print("数据集信息:")
    print(f"总数据量: {len(df)}")
    print(f"类别分布:")
    print(df['category'].value_counts())

    X = df["text_for_classification"]
    y = df["category"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # TF-IDF 向量化，针对中文进行优化
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),  # 1-2字符组合
        min_df=1,           # 最小文档频率
        max_df=0.9,         # 最大文档频率
        token_pattern=r'[\u4e00-\u9fa5]+|[a-zA-Z0-9]+',  # 中文字符和英文单词
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 训练逻辑回归模型
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=1.0  # 正则化参数
    )
    model.fit(X_train_tfidf, y_train)

    # 评估模型
    y_pred = model.predict(X_test_tfidf)
    
    print("\n=== 中文模型评估结果 ===")
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 计算每个类别的准确率
    from sklearn.metrics import accuracy_score
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n总体准确率: {overall_accuracy:.3f}")

    # 显示特征重要性（中文关键词）
    print("\n=== 各类别的关键特征词 ===")
    feature_names = vectorizer.get_feature_names_out()
    
    for i, category in enumerate(model.classes_):
        weights = model.coef_[i]
        top_indices = weights.argsort()[-8:][::-1]  # 前8个特征
        top_features = [feature_names[idx] for idx in top_indices]
        print(f"{category}: {', '.join(top_features)}")

    # 保存模型和向量化器
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(model_dir, "logistic_regression_model.pkl"))
    
    print(f"\n中文模型和向量化器已保存到 {model_dir}")
    
    # 测试中文样例
    print("\n=== 中文测试样例 ===")
    test_samples = [
        "登录时出现错误，无法访问我的账户",
        "希望能添加一个新的搜索功能", 
        "网站连接超时，无法加载页面",
        "文档中的示例代码有错误",
        "系统运行很慢，响应时间过长",
        "上传文件时出现崩溃",
        "建议增加批量删除功能",
        "密码重置邮件收不到",
        "API文档信息不完整",
        "页面加载速度太慢"
    ]
    
    for sample in test_samples:
        sample_tfidf = vectorizer.transform([sample])
        prediction = model.predict(sample_tfidf)[0]
        confidence = max(model.predict_proba(sample_tfidf)[0])
        print(f"输入: {sample}")
        print(f"预测: {prediction} (置信度: {confidence:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    train_chinese_model()

