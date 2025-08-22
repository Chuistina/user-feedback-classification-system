from app import predict_category, estimate_urgency

def test_classification_system():
    """测试新的分类系统"""
    
    test_cases = [
        "登录时出现错误，无法访问我的账户",
        "希望能添加一个新的搜索功能", 
        "网站连接超时，无法加载页面",
        "文档中的示例代码有错误",
        "系统运行很慢，响应时间过长",
        "我感觉你们在智能识别这个地方需要提升啊，识别的一点也不准确",
        "系统崩溃了，数据全部丢失！紧急处理",
        "建议增加夜间模式，如果可以的话",
        "忘记密码了，重置邮件收不到",
        "API文档链接失效",
        "页面加载速度太慢了"
    ]
    
    print("=== 新分类系统测试结果 ===\n")
    
    for text in test_cases:
        category, confidence = predict_category(text)
        urgency = estimate_urgency(text, category)
        
        print(f"输入: {text}")
        print(f"分类: {category} (置信度: {confidence:.3f})")
        print(f"紧急程度: {urgency}")
        print("=" * 60)

if __name__ == "__main__":
    test_classification_system()

