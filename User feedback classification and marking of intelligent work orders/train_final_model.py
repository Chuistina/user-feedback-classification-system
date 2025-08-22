import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import numpy as np

def create_expanded_chinese_dataset():
    """创建扩展的中文数据集"""
    
    # 每个类别40个样本，总共240个
    samples = {
        "Bug反馈": [
            "系统崩溃了", "页面白屏", "按钮无响应", "数据丢失", "登录失败", 
            "上传错误", "下载失败", "保存不了", "显示异常", "功能失效",
            "程序闪退", "界面卡死", "操作无效", "系统报错", "数据错乱",
            "文件损坏", "连接断开", "权限异常", "格式错误", "计算错误",
            "验证失败", "同步失败", "导入失败", "导出失败", "刷新无效",
            "搜索失败", "删除失败", "修改失败", "创建失败", "加载失败",
            "提交失败", "发送失败", "接收失败", "解析失败", "渲染失败",
            "编码错误", "解码错误", "转换失败", "压缩失败", "解压失败"
        ],
        "功能建议": [
            "增加搜索", "添加过滤", "支持导出", "批量操作", "自定义设置",
            "多语言支持", "夜间模式", "快捷键", "拖拽功能", "自动保存",
            "数据备份", "权限管理", "消息通知", "离线支持", "移动版本",
            "API接口", "插件系统", "模板功能", "协作功能", "分享功能",
            "评论系统", "标签功能", "分类管理", "排序功能", "筛选器",
            "图表统计", "报表功能", "日志记录", "审计功能", "版本控制",
            "工作流", "自动化", "集成功能", "扩展性", "个性化",
            "智能推荐", "数据分析", "可视化", "仪表盘", "监控功能"
        ],
        "账号问题": [
            "登录失败", "密码错误", "账号锁定", "注册失败", "验证码问题",
            "邮箱验证", "手机绑定", "实名认证", "权限不足", "账号冲突",
            "密码重置", "两步验证", "第三方登录", "会话超时", "强制下线",
            "多端登录", "账号合并", "用户信息", "头像上传", "个人资料",
            "隐私设置", "安全设置", "登录记录", "操作日志", "设备管理",
            "找回账号", "注销账号", "账号申诉", "身份验证", "访问权限",
            "角色管理", "组织架构", "部门权限", "数据权限", "功能权限",
            "单点登录", "LDAP集成", "OAuth认证", "token过期", "session失效"
        ],
        "网络连接失败": [
            "连接超时", "网络异常", "请求失败", "响应超时", "连接中断",
            "网速太慢", "加载失败", "同步失败", "上传中断", "下载中断",
            "代理问题", "DNS错误", "SSL错误", "证书问题", "跨域问题",
            "防火墙", "网络延迟", "带宽不足", "CDN问题", "负载均衡",
            "服务器异常", "API不通", "接口超时", "连接池满", "网络抖动",
            "丢包严重", "路由问题", "网关错误", "隧道失败", "VPN问题",
            "移动网络", "WiFi问题", "流量限制", "网络切换", "断网重连",
            "长连接", "短连接", "websocket", "http请求", "https问题"
        ],
        "文档问题": [
            "文档缺失", "说明不清", "示例错误", "版本过旧", "链接失效",
            "内容错误", "格式混乱", "翻译问题", "图片模糊", "视频失效",
            "搜索困难", "分类不当", "更新延迟", "下载失败", "打印异常",
            "API文档", "用户手册", "安装指南", "配置说明", "故障排除",
            "常见问题", "最佳实践", "教程缺失", "代码示例", "参数说明",
            "返回值", "错误码", "异常处理", "版本兼容", "升级指南",
            "开发文档", "运维文档", "测试文档", "部署文档", "监控文档",
            "性能优化", "安全配置", "备份恢复", "灾难恢复", "运维手册"
        ],
        "性能问题": [
            "运行缓慢", "响应延迟", "加载慢", "卡顿严重", "内存泄漏",
            "CPU过高", "磁盘满", "网络慢", "数据库慢", "查询慢",
            "渲染慢", "启动慢", "关闭慢", "切换慢", "滚动卡", 
            "动画卡顿", "播放卡顿", "下载慢", "上传慢", "同步慢",
            "处理慢", "计算慢", "压缩慢", "解压慢", "转换慢",
            "导入慢", "导出慢", "备份慢", "恢复慢", "索引慢",
            "搜索慢", "排序慢", "筛选慢", "统计慢", "报表慢",
            "批处理慢", "并发问题", "锁竞争", "死锁", "资源争用",
            "缓存失效", "预加载", "懒加载", "分页慢", "无限滚动慢"
        ]
    }
    
    # 创建数据集
    data = []
    for category, texts in samples.items():
        for text in texts:
            data.append({
                'id': len(data) + 1,
                'title': text,
                'body': text,
                'category': category,
                'urgency': '中',
                'html_url': f"https://example.com/{len(data) + 1}",
                'cleaned_title': text,
                'cleaned_body': text,
                'text_for_classification': text
            })
    
    return pd.DataFrame(data)

def train_final_model(model_dir="models"):
    """训练最终的中文分类模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 创建训练数据
    df = create_expanded_chinese_dataset()
    
    print("扩展数据集信息:")
    print(f"总数据量: {len(df)}")
    print(f"类别分布:")
    print(df['category'].value_counts())

    X = df["text_for_classification"]
    y = df["category"]

    # 更大的测试集比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 优化的TF-IDF向量化器
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        token_pattern=r'[\u4e00-\u9fa5]+',  # 只要中文字符
        sublinear_tf=True,  # 使用sublinear tf scaling
        norm='l2'  # L2 normalization
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 训练逻辑回归模型
    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42,
        C=0.5,  # 增加正则化
        solver='liblinear'  # 适合小数据集
    )
    model.fit(X_train_tfidf, y_train)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=3, scoring='accuracy')
    print(f"\n交叉验证平均准确率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # 评估模型
    y_pred = model.predict(X_test_tfidf)
    
    print("\n=== 最终模型评估结果 ===")
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 计算每个类别的准确率
    from sklearn.metrics import accuracy_score
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n总体准确率: {overall_accuracy:.3f}")

    # 保存模型和向量化器
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(model_dir, "logistic_regression_model.pkl"))
    
    print(f"\n最终模型已保存到 {model_dir}")
    
    # 测试样例
    print("\n=== 模型测试 ===")
    test_samples = [
        "登录时出现错误，无法访问我的账户",
        "希望能添加一个新的搜索功能", 
        "网站连接超时，无法加载页面",
        "文档中的示例代码有错误",
        "系统运行很慢，响应时间过长",
        "上传文件时程序崩溃了",
        "建议增加批量删除功能",
        "密码重置邮件收不到",
        "API文档信息不完整",
        "页面加载速度太慢"
    ]
    
    print("详细测试结果:")
    for sample in test_samples:
        sample_tfidf = vectorizer.transform([sample])
        prediction = model.predict(sample_tfidf)[0]
        probas = model.predict_proba(sample_tfidf)[0]
        confidence = max(probas)
        
        # 显示所有类别的概率
        print(f"\n输入: {sample}")
        print(f"预测: {prediction} (置信度: {confidence:.3f})")
        
        # 显示前3个最可能的类别
        top_indices = np.argsort(probas)[::-1][:3]
        print("概率分布:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. {model.classes_[idx]}: {probas[idx]:.3f}")
        print("-" * 60)

if __name__ == "__main__":
    train_final_model()

