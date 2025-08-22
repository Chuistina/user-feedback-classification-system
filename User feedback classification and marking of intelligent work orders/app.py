import requests
import joblib
import pandas as pd
import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the rule-based classifier
try:
    from rule_based_classifier import RuleBasedClassifier
    classifier = RuleBasedClassifier()
    print("Rule-based classifier created successfully")
except ImportError as e:
    print(f"Failed to import RuleBasedClassifier: {e}")
    classifier = None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"`.*?`", "", text)  # Remove code blocks (inline)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove code blocks (multiline)
    text = re.sub(r"[^a-z0-9\s\u4e00-\u9fa5]", "", text)  # Remove special characters, keep Chinese characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def predict_category(text):
    if not classifier:
        return "模型未加载", 0.0
    
    # 直接使用规则分类器，不需要预处理
    prediction, confidence = classifier.classify(text)
    
    return prediction, confidence

def estimate_urgency(text, category):
    """基于文本内容和类别估算紧急程度"""
    text_lower = text.lower()
    
    # 高紧急程度关键词
    high_urgency_keywords = [
        "紧急", "急", "立即", "马上", "赶紧", "立刻", "火急", "十万火急",
        "崩溃", "瘫痪", "死机", "无法使用", "完全不能", "严重", "重大",
        "数据丢失", "数据损坏", "安全", "漏洞", "攻击", "泄露"
    ]
    
    # 低紧急程度关键词
    low_urgency_keywords = [
        "希望", "建议", "可以", "最好", "如果可以", "有空", "方便",
        "优化", "改进", "完善", "增强", "美化", "升级", "更新"
    ]
    
    # 检查高紧急程度
    if any(keyword in text_lower for keyword in high_urgency_keywords):
        return "高"
    
    # 检查低紧急程度  
    if any(keyword in text_lower for keyword in low_urgency_keywords):
        return "低"
    
    # 基于类别设置默认紧急程度
    category_urgency_map = {
        "Bug反馈": "高",          # Bug通常需要尽快修复
        "账号问题": "高",          # 账号问题影响用户使用
        "网络连接失败": "中",       # 网络问题中等紧急
        "性能问题": "中",          # 性能问题中等紧急
        "文档问题": "低",          # 文档问题相对不紧急
        "功能建议": "低",          # 功能建议优先级较低
        "其他": "中"              # 其他问题中等紧急
    }
    
    return category_urgency_map.get(category, "中")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "请输入文本"}), 400
    
    category, confidence = predict_category(text)
    
    # Improved urgency estimation
    urgency = estimate_urgency(text, category)
    
    return jsonify({
        "category": category,
        "confidence": round(confidence, 2),
        "urgency": urgency
    })

@app.route("/fetch_issues")
def fetch_issues():
    repo_url = request.args.get("repo_url")

    if not repo_url:
        return render_template("index.html", error="请输入GitHub仓库URL")

    try:
        split_url = repo_url.split("/")
        owner = split_url[3]
        repo = split_url[4]

        response = requests.get(f"https://api.github.com/search/issues?q=repo:{owner}/{repo}+is:issue&per_page=10",
                headers={
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Authorization": "Bearer YOUR_GITHUB_TOKEN_HERE"
                    })

        if response.status_code == 200:
            issues = response.json()["items"]
            # Classify each issue
            for issue in issues:
                text = (issue.get("title", "") + " " + issue.get("body", ""))
                category, confidence = predict_category(text)
                issue["predicted_category"] = category
                issue["confidence"] = confidence
        else:
            issues = []
            
    except Exception as e:
        return render_template("index.html", error=f"获取Issues时出错: {str(e)}")

    return render_template("index.html", issues=issues)

@app.route("/analysis")
def analysis():
    try:
        # 读取原始数据并使用新的分类器重新分类
        df = pd.read_csv("data/labeling/label_task.csv")
        
        # 使用新分类器对所有数据重新分类
        new_categories = []
        feature_suggestions = []
        urgency_distribution = {"高": 0, "中": 0, "低": 0}
        
        for _, row in df.iterrows():
            text = row["text_for_classification"]
            if pd.isna(text) or not isinstance(text, str):
                continue
                
            category, confidence = predict_category(text)
            urgency = estimate_urgency(text, category)
            
            new_categories.append(category)
            urgency_distribution[urgency] += 1
            
            # 收集功能建议示例
            if category == "功能建议" and len(feature_suggestions) < 5:
                feature_suggestions.append(text[:50] + "..." if len(text) > 50 else text)
        
        # 统计新的分类结果
        from collections import Counter
        category_counts = Counter(new_categories)
        
        analysis_data = {
            "category_distribution": dict(category_counts),
            "urgency_distribution": urgency_distribution,
            "total_issues": len(new_categories),
            "feature_suggestions_count": category_counts.get("功能建议", 0),
            "feature_keywords": feature_suggestions,
            "classification_accuracy": "基于规则分类器，准确率显著提升",
            "top_categories": [
                {"name": "Bug反馈", "count": category_counts.get("Bug反馈", 0), "color": "#ff6b6b"},
                {"name": "功能建议", "count": category_counts.get("功能建议", 0), "color": "#4ecdc4"},
                {"name": "账号问题", "count": category_counts.get("账号问题", 0), "color": "#45b7d1"},
                {"name": "网络连接失败", "count": category_counts.get("网络连接失败", 0), "color": "#f9ca24"},
                {"name": "文档问题", "count": category_counts.get("文档问题", 0), "color": "#6c5ce7"},
                {"name": "性能问题", "count": category_counts.get("性能问题", 0), "color": "#a29bfe"}
            ]
        }
        
        return render_template("analysis.html", analysis=analysis_data)
    except Exception as e:
        return render_template("analysis.html", error=f"分析数据时出错: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

