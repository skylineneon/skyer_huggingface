


# from transformers import AutoTokenizer

# _tokenizer = AutoTokenizer.from_pretrained("./cache/skyer",trust_remote_code=True)

# print(_tokenizer.tokenize("<s>你好</s>",add_special_tokens=True))
# print(_tokenizer("<s>你好</s>"))

# test.py

from transformers import AutoModel, AutoConfig
# 打印当前工作目录
import os
print("Current working directory:", os.getcwd())

# 使用 AutoConfig 加载自定义的配置
model_dir = "./cache/skyer"
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

# 使用 AutoModel 加载自定义的模型
model = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)

# 打印模型配置
print("Model configuration:", config)

# 打印模型架构
print("Model architecture:", model)