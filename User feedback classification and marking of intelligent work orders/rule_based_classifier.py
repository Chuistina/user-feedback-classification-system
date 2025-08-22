import re
import joblib
import os
from collections import defaultdict

class RuleBasedClassifier:
    """基于规则的中文文本分类器"""
    
    def __init__(self):
        # 定义各类别的关键词
        self.category_keywords = {
            "Bug反馈": [
                # 系统问题
                "崩溃", "闪退", "卡死", "白屏", "黑屏", "死机", "异常", "错误", "失败",
                "无响应", "无反应", "不工作", "不起作用", "失效", "故障", "问题", "毛病",
                # 操作问题
                "无法", "不能", "不行", "不可以", "不支持", "不兼容", 
                "打不开", "进不去", "登不上", "用不了", "看不到", "找不到",
                # 数据问题
                "丢失", "消失", "没有了", "不见了", "清空", "删除", "损坏", "乱码",
                "数据错误", "计算错误", "显示错误", "格式错误",
                # 具体错误
                "500错误", "404错误", "超时", "中断", "中止", "停止", "终止"
            ],
            
            "功能建议": [
                # 希望类
                "希望", "建议", "推荐", "期望", "想要", "需要", "要求", "请求",
                "能否", "可否", "可以", "是否", "如果", "要是", "假如",
                # 功能类
                "增加", "添加", "新增", "加入", "引入", "集成", "整合", "融合",
                "改进", "改善", "优化", "提升", "增强", "强化", "完善", "补充",
                "支持", "兼容", "适配", "匹配", "对接", "连接", "链接",
                # 具体功能
                "搜索", "过滤", "筛选", "排序", "分类", "标签", "评论", "分享",
                "导出", "导入", "备份", "恢复", "同步", "上传", "下载",
                "批量", "多选", "全选", "快捷键", "拖拽", "预览", "打印",
                "自定义", "个性化", "配置", "设置", "选项", "参数",
                "夜间模式", "暗色", "主题", "皮肤", "样式", "界面", "布局"
            ],
            
            "账号问题": [
                # 登录相关
                "登录", "登陆", "注册", "注销", "退出", "登出", "下线",
                "用户名", "密码", "账号", "账户", "帐号", "帐户",
                # 验证相关
                "验证码", "验证", "认证", "授权", "权限", "角色", "身份",
                "绑定", "解绑", "关联", "取消关联",
                # 个人信息
                "个人信息", "个人资料", "用户信息", "头像", "昵称", "邮箱", "手机",
                "实名", "身份证", "证件", "资质",
                # 安全相关
                "两步验证", "二次验证", "安全", "保护", "隐私", "数据保护",
                "锁定", "冻结", "封号", "解封", "申诉", "找回"
            ],
            
            "网络连接失败": [
                # 连接问题
                "连接", "网络", "网速", "带宽", "信号", "WiFi", "流量",
                "超时", "中断", "断开", "断线", "掉线", "离线",
                # 加载问题
                "加载", "缓冲", "请求", "响应", "延迟", "卡顿", "缓慢",
                "打不开", "无法访问", "访问失败", "连不上",
                # 技术问题
                "DNS", "代理", "VPN", "防火墙", "端口", "协议",
                "SSL", "证书", "加密", "解密", "握手",
                "CDN", "节点", "服务器", "主机", "域名", "IP",
                # 移动网络
                "4G", "5G", "移动网络", "数据", "漫游", "信号弱"
            ],
            
            "文档问题": [
                # 文档类型
                "文档", "说明", "手册", "指南", "教程", "帮助", "文档",
                "API", "接口", "参数", "返回值", "示例", "例子",
                # 文档问题
                "不清楚", "不明确", "不详细", "不完整", "缺少", "遗漏", "漏掉",
                "错误", "有误", "不对", "不正确", "不准确", "过时", "老旧",
                "看不懂", "难理解", "复杂", "混乱", "乱", "模糊",
                # 具体内容
                "链接", "图片", "视频", "代码", "配置", "安装", "部署",
                "更新", "版本", "兼容", "依赖", "环境", "系统要求"
            ],
            
            "性能问题": [
                # 速度问题
                "慢", "缓慢", "延迟", "卡", "卡顿", "停顿", "等待",
                "响应慢", "加载慢", "运行慢", "处理慢", "启动慢",
                # 资源问题
                "内存", "CPU", "磁盘", "空间", "资源", "占用", "消耗",
                "泄漏", "溢出", "不足", "满了", "爆满",
                # 性能表现
                "卡死", "假死", "无响应", "崩溃", "闪退",
                "渲染", "绘制", "刷新", "更新", "同步",
                "并发", "并行", "队列", "阻塞", "锁定",
                # 具体场景
                "大文件", "批处理", "大数据", "高并发", "峰值",
                "数据库", "查询", "索引", "缓存", "优化"
            ]
        }
        
        # 为每个关键词计算权重（基于重要性和特异性）
        self.keyword_weights = self._calculate_keyword_weights()
    
    def _calculate_keyword_weights(self):
        """计算关键词权重"""
        weights = {}
        
        # 统计每个词在多少个类别中出现
        word_category_count = defaultdict(int)
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                word_category_count[keyword] += 1
        
        # 计算权重：出现在越少类别中的词权重越高
        for category, keywords in self.category_keywords.items():
            weights[category] = {}
            for keyword in keywords:
                # 基础权重：1.0
                # 特异性权重：1 / 出现的类别数
                # 长度权重：较长的词权重稍高
                base_weight = 1.0
                specificity_weight = 1.0 / word_category_count[keyword]
                length_weight = min(len(keyword) / 4.0, 2.0)  # 最大2倍
                
                weights[category][keyword] = base_weight * specificity_weight * length_weight
        
        return weights
    
    def classify(self, text):
        """分类文本"""
        if not text or not isinstance(text, str):
            return "其他", 0.1
        
        # 清理文本
        text = text.lower().strip()
        
        # 计算每个类别的得分
        category_scores = defaultdict(float)
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # 计算该关键词的得分
                    weight = self.keyword_weights[category][keyword]
                    
                    # 考虑出现次数
                    count = text.count(keyword)
                    
                    # 考虑位置权重（标题位置的词权重更高）
                    position_weight = 1.0
                    if len(text) > 20 and keyword in text[:20]:
                        position_weight = 1.5
                    
                    score = weight * count * position_weight
                    category_scores[category] += score
        
        # 如果没有匹配的关键词，返回其他
        if not category_scores:
            return "其他", 0.1
        
        # 找到得分最高的类别
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        # 计算置信度（归一化得分）
        total_score = sum(category_scores.values())
        confidence = best_category[1] / total_score if total_score > 0 else 0.1
        
        # 限制置信度范围
        confidence = min(max(confidence, 0.1), 0.95)
        
        return best_category[0], confidence
    
    def get_category_keywords_for_text(self, text, category):
        """获取文本中匹配该类别的关键词"""
        matched_keywords = []
        text_lower = text.lower()
        
        if category in self.category_keywords:
            for keyword in self.category_keywords[category]:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
        
        return matched_keywords

def create_rule_based_model(model_dir="models"):
    """创建并保存基于规则的分类器"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 创建分类器
    classifier = RuleBasedClassifier()
    
    # 测试分类器
    print("=== 基于规则的分类器测试 ===")
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
        "页面加载速度太慢",
        "我感觉你们在智能识别这个地方需要提升啊，识别的一点也不准确",
        "系统有bug",
        "网络不通",
        "想要夜间模式",
        "忘记密码了"
    ]
    
    for sample in test_samples:
        category, confidence = classifier.classify(sample)
        matched_keywords = classifier.get_category_keywords_for_text(sample, category)
        
        print(f"\n输入: {sample}")
        print(f"预测: {category} (置信度: {confidence:.3f})")
        if matched_keywords:
            print(f"匹配关键词: {', '.join(matched_keywords)}")
        print("-" * 60)
    
    # 保存分类器（使用pickle）
    import pickle
    classifier_path = os.path.join(model_dir, "rule_based_classifier.pkl")
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"\n基于规则的分类器已保存到: {classifier_path}")
    
    return classifier

if __name__ == "__main__":
    create_rule_based_model()
