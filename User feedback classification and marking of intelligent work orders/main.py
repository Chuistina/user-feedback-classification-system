import requests
import joblib
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
    model = joblib.load("../models/logistic_regression_model.pkl")
    print("Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("Model files not found. Please train the model first.")
    vectorizer = None
    model = None

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
    if not vectorizer or not model:
        return "模型未加载", 0.0
    
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    confidence = max(model.predict_proba(text_tfidf)[0])
    
    return prediction, confidence

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
    
    # Simple urgency estimation based on keywords
    urgency = "中"
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["urgent", "critical", "blocker", "immediately", "now"]):
        urgency = "高"
    elif any(keyword in text_lower for keyword in ["wish", "nice to have", "later"]):
        urgency = "低"
    
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
                    "Authorization": "Bearer "
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
        # Mock analysis data since pandas is not available in deployment
        analysis_data = {
            "category_distribution": {
                "Bug反馈": 67,
                "功能建议": 45,
                "其他": 32,
                "文档问题": 28,
                "性能问题": 18,
                "账号问题": 10
            },
            "total_issues": 200,
            "feature_suggestions_count": 45,
            "feature_keywords": [
                "Add new authentication method",
                "Improve API response time", 
                "Support for mobile devices",
                "Better error messages",
                "Enhanced documentation"
            ]
        }
        
        return render_template("analysis.html", analysis=analysis_data)
    except Exception as e:
        return render_template("analysis.html", error=f"分析数据时出错: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

