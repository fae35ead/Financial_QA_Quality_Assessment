import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 定位我们刚刚炼制好的模型位置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
final_model_dir = os.path.join(project_root, 'models', 'final_bert_qa_model')

print("正在唤醒我们训练好的 QA 质量评估大模型...")

# 2. 加载分词器和模型权重 (从本地文件夹极速加载)
tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(final_model_dir)

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 开启评估模式，关闭 Dropout，保证输出稳定

# 3. 定义标签映射字典
label_map = {
    0: "🔴 低质量 (回答拉跨/答非所问)",
    1: "🟡 中等质量 (中规中矩)",
    2: "🟢 高质量 (逻辑清晰/信息丰富)"
}


# 4. 封装一个打分小函数
def rate_qa_quality(question, answer):
    # 核心逻辑：用 [SEP] 把问题和回答拼接起来，和训练时一模一样！
    input_text = f"{question}[SEP]{answer}"

    # 将文本转为模型认识的张量 (Tensor)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    ).to(device)

    # 不计算梯度，加速推理并省显存
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取最高概率的类别索引
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    # 提取具体的概率值 (通过 Softmax 把数字变成 0~1 的百分比)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    confidence = probabilities[predicted_class_id].item() * 100

    print("\n" + "=" * 50)
    print(f"👤 用户提问: {question}")
    print(f"🤖 AI 回答: {answer}")
    print("-" * 50)
    print(f"🏆 模型评分: {label_map[predicted_class_id]}")
    print(f"🎯 确信度: {confidence:.2f}%")
    print("=" * 50 + "\n")


# ==========================================
# 5. 现场测试！发挥你的想象力造几个句子试试
# ==========================================
if __name__ == '__main__':
    # 测试案例 1 ---> bad case
    q1 = "海南封关运作，对公司来说是否是一个机遇，公司将如何抓住这历史性机遇发展自身？？"
    a1 = "您好，公司目前研发生产的视觉检测设备主要应用于PCB行业，包括：全自动钻针刃面检、自动影像测量仪 、钻针缺陷检测机等。感谢您的关注。"
    rate_qa_quality(q1, a1)

    # 测试案例 2
    q2 = "公司的市场地位如何？有哪些核心竞争力？"
    a2 = "感谢您的关注！目前，公司在海南省共计拥有10家商品混凝土搅拌站，26条生产线，设计产能达780万立方米，覆盖海南省各主要市县，包括海口、三亚、琼海、陵水、澄迈、儋州等。公司商品混凝土业务的市场占有率约20%。公司核心竞争优势包括区位政策优势、规模品牌优势、稳定高效的管理团队、研发和技术创新优势。谢谢。"
    rate_qa_quality(q2, a2)

    # 测试案例 3
    q3 =  "公司钙钛矿涂布设备有产能规划吗？"
    a3 = "您好，公司在做技术验证的是狭缝涂布机，公司今年将积极推进功能膜、特种膜、钙钛矿等新兴技术的产业化进程。用于钙钛矿型的平板涂布设备尚处于前期技术验证阶段，后续尚需进一步技术验证，在效果验证方面，存在有效性或不达预期的风险，后续能否获得客户认可具有不确定性，未来产生的经济效益和对公司业绩的影响存在不确定性。"
    rate_qa_quality(q3, a3)

    # 测试案例 4
    q4 = "你好董秘！可用作医疗的ChatGPT，贵司会如何善加利用？"
    a4 = "您好！重大资产重组有关情况请参见公司于2023年1月18日披露在巨潮资讯网上的《浙江英特集团股份有限公司发行股份及支付现金购买资产并募集配套资金暨关联交易报告书（草案）（修订稿）》。感谢您的关注。"
    rate_qa_quality(q4, a4)
