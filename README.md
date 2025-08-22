# 用户反馈分类与智能工单系统

本项目是一个用户反馈分类与智能工单系统，包含从数据收集、模型训练到Web应用的全套流程。
<img width="1749" height="775" alt="屏幕截图 2025-08-22 170857" src="https://github.com/user-attachments/assets/29ffd4ff-4cec-4ef2-ad7b-894b6b05ae03" />
<img width="1972" height="1202" alt="image" src="https://github.com/user-attachments/assets/33cfc9d1-e072-4eb2-9815-2df42be74d18" />


## 快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
pip install pandas
```

### 2. 数据准备（如果需要重新获取数据）
```bash
python fetch_issues.py  # 获取GitHub Issues数据
python process_issues.py  # 处理数据
python clean_data.py  # 清洗数据
```

### 3. 模型训练
```bash
python train_model.py  # 训练分类模型
```

### 4. 运行Web应用
```bash
python app.py  # 启动Flask应用
```
然后访问 http://localhost:5000

## API Key 配置

### GitHub API Token
如需获取更多GitHub数据或避免rate limit，请在以下文件中配置GitHub token：

1. **app.py** 第86行：
   ```python
   "Authorization": "Bearer YOUR_GITHUB_TOKEN_HERE"
   ```

2. **fetch_issues.py** 第20行：
   ```python
   "Authorization": "Bearer YOUR_GITHUB_TOKEN_HERE"
   ```

获取GitHub Token步骤：
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 选择适当的权限（repo, public_repo等）
4. 复制生成的token并替换上面的 `YOUR_GITHUB_TOKEN_HERE`

### 其他大模型API（如果需要扩展）
如果想集成其他大模型API，可以在相应文件中添加配置。

## 项目结构
```
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── labeling/          # 标注数据
├── models/                # 训练好的模型文件
├── app.py                 # Flask Web应用
├── train_model.py        # 模型训练脚本
├── fetch_issues.py       # 数据获取脚本
├── process_issues.py     # 数据处理脚本
├── clean_data.py         # 数据清洗脚本
└── requirements.txt      # 依赖包列表
```

## 故障排除

### TemplateNotFound 错误
如果访问 http://localhost:5000 时出现 `TemplateNotFound` 错误，请确保：
1. `templates` 目录存在且包含 `index.html` 和 `analysis.html` 文件
2. Flask 应用能正确找到模板文件路径

修复方法：
```bash
mkdir templates
move index.html templates/
move analysis.html templates/
```

## 使用说明

1. **文本分类**：在Web界面输入用户反馈文本，系统会自动分类并显示置信度
2. **紧急程度评估**：基于关键词自动评估反馈紧急程度
3. **GitHub Issues获取**：输入GitHub仓库URL可获取并分类该仓库的issues

---

# 用户反馈标注指南

本文件 `label_task.csv` 包含了从 GitHub Issues 抓取并初步处理过的用户反馈数据，需要手动进行分类标注。

## 标注字段说明

- `id`: 原始 Issue 的唯一标识符。
- `title`: Issue 标题。
- `body`: Issue 正文内容。
- `category`: **需要标注的字段**。请根据以下预设类别进行选择，可以多选，用逗号 `,` 分隔。如果一个反馈同时属于多个类别，请全部列出。
  - `Bug反馈`: 报告软件缺陷、错误、崩溃、异常行为等。
  - `功能建议`: 提出新功能、改进现有功能、优化用户体验等。
  - `账号问题`: 涉及用户账户、登录、注册、密码、权限等问题。
  - `网络连接失败`: 报告网络连接、断线、超时、代理设置等问题。
  - `文档问题`: 涉及文档不清晰、缺失、错误、需要更新等。
  - `性能问题`: 报告软件运行缓慢、卡顿、响应延迟等。
  - `其他`: 不属于以上任何类别的反馈。
- `urgency`: **需要标注的字段**。请根据反馈内容的紧急程度进行选择，单选。
  - `高`: 严重影响用户使用，导致功能不可用，或有数据丢失风险，需要立即处理。
  - `中`: 影响部分功能，或导致用户体验不佳，但有 workaround，需要尽快处理。
  - `低`: 不影响核心功能，或仅为优化建议，可以延后处理。
- `html_url`: 原始 Issue 的链接，方便您查看上下文。
- `cleaned_title`: 清洗后的 Issue 标题。
- `cleaned_body`: 清洗后的 Issue 正文内容。
- `text_for_classification`: 用于分类的文本，由清洗后的标题和正文拼接而成。

## 标注示例

| category           | urgency |
|--------------------|---------|
| Bug反馈, 性能问题  | 高      |
| 功能建议           | 低      |
| 账号问题           | 中      |

## 标注注意事项

1.  请仔细阅读 `title` 和 `body` 内容，理解用户反馈的真实意图。
2.  尽量选择最能准确描述反馈内容的类别。
3.  对于 `category` 字段，如果一个反馈同时涉及多个类别，请务必全部标注。
4.  对于 `urgency` 字段，请根据反馈对用户的影响程度进行判断。
5.  如果遇到难以判断的反馈，请在备注中说明，或暂时标记为 `其他`。

完成标注后，请将 `label_task.csv` 文件提交。

