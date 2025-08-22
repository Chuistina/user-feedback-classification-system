import json
import os
import pandas as pd

def categorize_issue(title, body):
    title = title if title is not None else ""
    body = body if body is not None else ""
    text = (title + " " + body).lower()
    
    # Define keywords for categories
    categories = {
        "Bug反馈": ["bug", "error", "fail", "crash", "issue", "defect", "fix"],
        "功能建议": ["feature", "suggest", "enhance", "request", "new idea"],
        "账号问题": ["account", "login", "password", "auth", "user"],
        "网络连接失败": ["network", "connection", "offline", "timeout", "proxy"],
        "文档问题": ["doc", "documentation", "readme", "example"],
        "性能问题": ["performance", "slow", "lag", "speed"],
        "其他": []
    }

    # Define keywords for urgency
    urgency_keywords = {
        "高": ["urgent", "critical", "blocker", "immediately", "now"],
        "中": ["important", "soon", "priority"],
        "低": ["wish", "nice to have", "later"]
    }

    found_categories = []
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            found_categories.append(category)
    
    if not found_categories:
        found_categories.append("其他")

    urgency = "中"
    for level, keywords in urgency_keywords.items():
        if any(keyword in text for keyword in keywords):
            urgency = level
            break
            
    return ", ".join(found_categories), urgency

def process_issues(input_file, output_dir="data/processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_issues = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            issue = json.loads(line)
            title = issue.get("title", "")
            body = issue.get("body", "")
            
            category, urgency = categorize_issue(title, body)
            
            processed_issues.append({
                "id": issue.get("id"),
                "title": title,
                "body": body,
                "category": category,
                "urgency": urgency,
                "html_url": issue.get("html_url")
            })

    output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".jsonl", ".csv"))
    df = pd.DataFrame(processed_issues)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Processed {len(processed_issues)} issues and saved to {output_file}")

if __name__ == "__main__":
    input_path = "data/raw/requests_requests_issues.jsonl"
    process_issues(input_path)


