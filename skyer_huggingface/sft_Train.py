from transformers import (AutoModelForCausalLM,  # 用于因果语言模型的自动模型类
                          AutoTokenizer,  # 用于自动选择和加载与特定预训练模型相匹配的分词器的类
                          AutoConfig,  # 用于自动加载与特定预训练模型相匹配的配置
                          TrainingArguments,  # 用于定义训练过程中的各种参数的类
                          Trainer,
                          DataCollatorForSeq2Seq  # 用于处理序列到序列任务数据的类
                          # 如机器翻译或文本摘要。它负责将多个训练样本拼接成一个批次，以便模型可以批量处理
                          )
from datasets import load_dataset

# _model_path = "/root/workspace/skyer_huggingface/cache/skyer"
_model_path = "/root/workspace/skyer_huggingface/cache/qwen/Qwen2___5-0___5B-Instruct"
'''
加载tokenizer、config、model
并且加载配置config一定要在加载模型model之前
'''

_tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)
_config = AutoConfig.from_pretrained(_model_path, trust_remote_code=True)
# print(_config)
_config.cache_max_batch_size = None
# print(_config)
# exit()
_model = AutoModelForCausalLM.from_pretrained(
    _model_path, config=_config, trust_remote_code=True)

# print(_model) #查看模型结构
# exit()

# for k,v in _model.named_parameters(): #查看模型参数与形状
#     print(k,v.shape)

# exit()

# 加载数据集
_dataset = load_dataset(
    "json", data_files="/root/workspace/PEFT/ruozhiba_qa.json", split="train")  # 为什么写split="train"？
# print(_dataset.shape)
# print(_dataset)
# exit()


def preprocess_dataset(example):
    MAX_LENGTH = 128
    _input_ids, _attention_mask, _labels = [], [], []
    '''
    使用_tokenizer对提示语和结果进行分词
    结果报含input_ids、token_type_ids、attention_mask三部分
    '''
    _instruction = _tokenizer(
        f"<s>user\n{example['instruction']}</s>\n<s>assistant\n", add_special_tokens=False)
    _response = _tokenizer(
        example["output"] + _tokenizer.eos_token, add_special_tokens=False)

    '''
    将input_ids、attention进行拼接
    '''
    _input_ids = _instruction["input_ids"] + _response["input_ids"]
    _attention_mask = _instruction["attention_mask"] + \
        _response["attention_mask"]

    # _labels 生成了一个标签列表，其中指令部分的标签为 -100（在损失计算时被忽略），回答部分的标签为其对应的 input_ids
    _labels = [-100] * len(_instruction["input_ids"]) + _response["input_ids"]

    '''
    数据截断,在后面进行padding补齐(动态补齐)
    此处如果进行补齐，会占用很多显存
    '''
    if len(_input_ids) > MAX_LENGTH:
        _input_ids = _input_ids[:MAX_LENGTH]
        _attention_mask = _attention_mask[:MAX_LENGTH]
        _labels = _labels[:MAX_LENGTH]

    return {
        "input_ids": _input_ids,
        "attention_mask": _attention_mask,
        "labels": _labels
    }


# 对数据集进行预处理
# remove_columns=_dataset.column_names 参数表示在映射后删除原始数据集中的所有列，只保留预处理后的数据
_dataset = _dataset.map(
    preprocess_dataset, remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()

_training_args = TrainingArguments(
    output_dir="/root/workspace/skyer_huggingface/checkpoints/qwen",  # 设置检查点输出文件目录
    per_device_train_batch_size=2,  # 设置每个设备上的批量大小
    gradient_accumulation_steps=1,  # 设置梯度累积的步数
    num_train_epochs=60,  # 设置训练的总轮数
    save_steps=100,  # 设置模型保存的步数
    logging_steps=100,  # 设置日志记录的步数
    # deepspeed="deepspeed_config.json" # 用于大模型训练的深度学习优化库
    # optim="paged_adamw_32bit", #  设置优化器
)

'''
用于训练模型的类，
它提供了一个高级的API来简化训练过程。
它接受一个模型、数据、训练参数等作为输入，并处理训练循环
'''
trainer = Trainer(
    model=_model,
    args=_training_args,
    train_dataset=_dataset,
    # Pad to the longest sequence in the batch
    data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer, padding=True)
)
trainer.train()
