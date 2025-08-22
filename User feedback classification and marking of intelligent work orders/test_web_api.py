import requests
import time

def test_web_api():
    """测试Web API分类功能"""
    
    # 等待Flask启动
    time.sleep(2)
    
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
        "页面加载速度太慢了"
    ]
    
    print("=== Web API 测试结果 ===\n")
    
    for text in test_cases:
        try:
            response = requests.post(
                'http://localhost:5000/classify',
                json={'text': text},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"输入: {text}")
                print(f"分类: {result['category']}")
                print(f"置信度: {result['confidence']}")
                print(f"紧急程度: {result['urgency']}")
                print("=" * 60)
            else:
                print(f"API错误: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("连接失败，请确认Flask应用正在运行")
            break
        except Exception as e:
            print(f"请求失败: {e}")

if __name__ == "__main__":
    test_web_api()

