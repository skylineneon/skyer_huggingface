from transformers import pipeline
from transformers import (AutoModelForCausalLM,  # 用于因果语言模型的自动模型类
                          AutoTokenizer,  # 用于自动选择和加载与特定预训练模型相匹配的分词器的类
                          AutoConfig  # 用于自动加载与特定预训练模型相匹配的配置  # 用于处理序列到序列任务数据的类
                          # 如机器翻译或文本摘要。它负责将多个训练样本拼接成一个批次，以便模型可以批量处理
                          )
# _tokenizer_path = "/root/workspace/skyer_huggingface/cache/skyer"
_tokenizer_path = "/root/workspace/skyer_huggingface/cache/qwen/Qwen2___5-3B-Instruct"
_model_path = "/root/workspace/skyer_huggingface/checkpoints/train/checkpoint-2244"

_tokenizer = AutoTokenizer.from_pretrained(
    _tokenizer_path, trust_remote_code=True)
#Skyer
# _config = AutoConfig.from_pretrained(_model_path, trust_remote_code=True)
# _config.cache_max_batch_size = 1
# _model = AutoModelForCausalLM.from_pretrained(
#     _model_path,config=_config,trust_remote_code=True, device_map="cuda")

#qwen
_config= AutoConfig.from_pretrained(_tokenizer_path, trust_remote_code=True)
_model = AutoModelForCausalLM.from_pretrained(
    _tokenizer_path,config=_config,trust_remote_code=True, device_map="cuda")


pp = pipeline("text-generation", model=_model,
              tokenizer=_tokenizer, trust_remote_code=True)

print(pp("你是谁？"))
