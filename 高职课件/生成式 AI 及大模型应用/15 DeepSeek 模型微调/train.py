import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 hugging face 模型 tokenizers 并行错误，关闭并行许可。

model_path = r"/home/jovyan/work/datasets/685268a0767d61d67a4b26f3-momodel/deepseek_finetune/deepseek_r1_1b/"  # 模型路径  
data_path = r"/home/jovyan/work/datasets/685268a0767d61d67a4b26f3-momodel/deepseek_finetune/medical_o1_sft_Chinese.json"  # 数据集路径
output_path = r"/home/jovyan/work/results"
dataset = load_dataset("json", data_files=data_path, split='train')  

data_tokenizer = AutoTokenizer.from_pretrained(model_path)  
data_tokenizer.padding_side = 'right'

def dataset_process(example):
    instruction = f"你是一个有用的助手。<｜User｜>{example['Question']}<｜Assistant｜><think>\n"

    # 构建目标部分（模型需要生成的内容）
    target = f"{example['Complex_CoT']}</think>\n\n答案：{example['Response']}<｜end▁of▁sentence｜>"

    # 完整文本拼接
    full_text = instruction + target

    # 分词处理
    inputs = data_tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
        add_special_tokens=True
    )

    # 计算指令部分的token长度（不包含特殊token）
    instruction_encoded = data_tokenizer(
        instruction,
        add_special_tokens=False  # 禁用特殊token添加
    )["input_ids"]
    instruction_length = len(instruction_encoded)

    labels = inputs["input_ids"].clone().squeeze(0)
    labels[:instruction_length] = -100

    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "labels": labels
    }

update_dataset = dataset.map(dataset_process,
                             desc="Processing...",
#                              num_proc=4,
                            )

assert torch.cuda.is_available(), "必须使用GPU进行训练！"
device = torch.device("cuda")  

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="sdpa"
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 自定义回调类, 在训练时执行 log
class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.epoch = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])
        self.epoch += 1
        
loss_callback = LossCallback()

model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=1,  # 显存优化设置
    gradient_accumulation_steps=4,  # 累计梯度相当于batch_size=4
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,  # 开启混合精度
    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
    optim="adamw_torch",
    dataloader_pin_memory=False,
    remove_unused_columns=True
)

def data_collator(data):
    batch = {
        "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
        "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]),
        "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]) 
    }
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=update_dataset,
    data_collator=data_collator,
    callbacks=[loss_callback]
)

trainer.train()
model.save_pretrained(output_path)
