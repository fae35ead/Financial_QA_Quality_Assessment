'''该文件实现了一个工业级的 LCPPN 推理管线，能够对公司财报问答进行智能分析和分类。核心功能包括：
1. 加载预训练的三大模型（根节点模型、Direct 子节点模型、Evasive 子节点模型），并将它们部署在 GPU 上以实现高速推理。
2. 端到端的分析接口 `analyze_qa`，输入投资者提问和董秘回答，输出根节点和子节点的分类结果以及对应的置信度。
3. 置信度保底机制：如果根节点的置信度过低，系统会自动发出预警，提示该问答可能超出常规金融语境，建议人工复核。
4. 实战演练：在 `__main__` 块中，注入了三个精心设计的测试用例，涵盖了“战略性模糊+推迟回答”、“财务表现直接回答”和“外部归因”三种典型场景，展示了 LCPPN 在实际应用中的强大分析能力和细致的分类结果。
'''

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# ================= 生产环境配置区 =================
# 绝对路径防护
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))

# 我们精心打磨的三大主力模型路径
ROOT_MODEL_DIR = os.path.join(project_root, "models", "student_100k_distilbert")
DIRECT_MODEL_DIR = os.path.join(project_root, "models", "lcppn_direct_classifier")
EVASIVE_MODEL_DIR = os.path.join(project_root, "models", "lcppn_evasive_classifier")

MAX_LEN = 384

# ================= 业务标签字典 =================
ROOT_LABELS = {0: "Direct (直接响应)", 1: "Intermediate (避重就轻)", 2: "Evasive (打太极)"}
DIRECT_LABELS = {0: "资本运作与并购", 1: "技术与研发进展", 2: "产能与项目规划", 3: "合规与风险披露", 4: "财务表现指引"}
EVASIVE_LABELS = {0: "推迟回答", 1: "转移话题", 2: "战略性模糊", 3: "外部归因"}


class LCPPNPipeline:
    """
    工业级 LCPPN (Local Classifier Per Parent Node) 推理管线
    """

    def __init__(self):
        print("⚙️ [System] 正在初始化 LCPPN 联调推理引擎...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ [System] 核心计算单元挂载完毕: {self.device}")

        # 1. 统一分词器 (三个模型基座相同，复用一个 Tokenizer 即可，节省内存)
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT_MODEL_DIR)

        # 2. 预热加载三大模型，全部打入 VRAM 并开启 eval 模式
        t0 = time.time()
        self.root_model = self._load_model(ROOT_MODEL_DIR)
        self.direct_model = self._load_model(DIRECT_MODEL_DIR)
        self.evasive_model = self._load_model(EVASIVE_MODEL_DIR)
        print(f"✅ [System] 三路大军集结完毕！耗时: {time.time() - t0:.2f}s")

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"🚨 致命错误: 找不到模型文件 {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()  # 【铁律】线上推理绝不能开启 Dropout
        return model

    def _predict_node(self, model, input_ids, attention_mask):
        """通用节点推理函数，返回预测ID和置信度"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # 使用 Softmax 压榨出概率分布
        probs = F.softmax(logits, dim=-1)
        # 获取最高概率的索引和对应的置信度
        confidence, pred_id = torch.max(probs, dim=-1)
        return pred_id.item(), confidence.item()

    def analyze_qa(self, question: str, reply: str):
        """端到端智能分析接口"""
        if not question.strip() or not reply.strip():
            return {"error": "提问或回答不能为空"}

        # 1. 文本预处理
        inputs = self.tokenizer(
            text=question,
            text_pair=reply,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        result = {
            "question": question,
            "reply": reply,
            "root_label": None,
            "root_confidence": 0.0,
            "sub_label": None,
            "sub_confidence": 0.0,
            "warning": None
        }

        # 【核心保护圈】: 切断梯度计算，防止显存泄漏
        with torch.no_grad():
            # 🚀 步骤一：过根节点 (Root Node) 宏观路由
            root_id, root_conf = self._predict_node(self.root_model, input_ids, attention_mask)
            result["root_label"] = ROOT_LABELS.get(root_id, "Unknown")
            result["root_confidence"] = round(root_conf, 4)

            # 🛡️ 步骤二：置信度保底拦截 (动态回滚预警)
            if root_conf < 0.45:
                result["warning"] = "根节点置信度过低，该问答可能超出常规金融语境，建议人工复核。"

            # 🔀 步骤三：LCPPN 动态路由与子节点解析
            if root_id == 0:
                # 路由到 Direct 节点
                sub_id, sub_conf = self._predict_node(self.direct_model, input_ids, attention_mask)
                result["sub_label"] = DIRECT_LABELS.get(sub_id, "Unknown")
                result["sub_confidence"] = round(sub_conf, 4)

            elif root_id == 2:
                # 路由到 Evasive 节点
                sub_id, sub_conf = self._predict_node(self.evasive_model, input_ids, attention_mask)
                result["sub_label"] = EVASIVE_LABELS.get(sub_id, "Unknown")
                result["sub_confidence"] = round(sub_conf, 4)

            elif root_id == 1:
                # Intermediate 节点为叶子节点，无须深入
                result["sub_label"] = "无下游细分 (部分响应)"
                result["sub_confidence"] = 1.0

        return result


# ================= 实战演练 =================
if __name__ == "__main__":
    # 初始化管线（模拟服务器启动）
    pipeline = LCPPNPipeline()

    print("\n" + "=" * 50)
    print("🔥 LCPPN 终极测谎仪已上线！正在进行实战注入测试...")
    print("=" * 50)

    # 测试用例 1：极其狡猾的“战略性模糊”+“推迟回答”
    q1 = "请问公司一季度的净利润预期是多少？二期工厂什么时候能量产出货？"
    r1 = "尊敬的投资者您好，公司目前的生产经营一切正常，各项业务正在稳步推进中。关于一季度的具体业绩数据，请您关注公司后续在指定信息披露媒体发布的定期报告。感谢您对公司的关注。"

    # 测试用例 2：干脆利落的“财务表现”直接回答
    q2 = "公司去年在新能源板块的营收占比大概达到了多少？"
    r2 = "您好！2025年公司新能源业务板块表现强劲，全年实现营收占公司总营业收入的比例约为45.2%，较去年同期有显著提升。"

    # 测试用例 3：经典的“外部归因”甩锅
    q3 = "为什么公司今年上半年的毛利率出现了如此严重的下滑？"
    r3 = "您好。今年上半年，受全球地缘政治冲突加剧以及国际海运费暴涨的宏观系统性影响，公司核心原材料的采购成本大幅攀升，导致了毛利率的阶段性承压。"

    test_cases = [(q1, r1), (q2, r2), (q3, r3)]

    for i, (q, r) in enumerate(test_cases, 1):
        print(f"\n[测试用例 {i}]")
        print(f"Q: {q}")
        print(f"A: {r}")

        t_start = time.time()
        res = pipeline.analyze_qa(q, r)
        t_cost = (time.time() - t_start) * 1000  # 转换为毫秒

        print(f"🎯 根节点判定 : {res['root_label']} (置信度: {res['root_confidence'] * 100:.1f}%)")
        print(f"🔍 子节点诊断 : {res['sub_label']} (置信度: {res['sub_confidence'] * 100:.1f}%)")
        if res['warning']:
            print(f"⚠️ 系统警告   : {res['warning']}")
        print(f"⏱️ 推理耗时   : {t_cost:.2f} ms")
        print("-" * 50)