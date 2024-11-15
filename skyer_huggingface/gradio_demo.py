from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")

_model_path = "/root/workspace/skyer_huggingface/cache/skyer"
# _model_path = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"

_model = AutoModelForCausalLM.from_pretrained(_model_path,  device_map="cuda",trust_remote_code=True)
_tokenizer = AutoTokenizer.from_pretrained(_model_path,trust_remote_code=True)

# print(type(_model)) #技巧：查看模型类型，然后导入可以查看源码
# exit()

# _pp = pipeline(task=Tasks.text_generation,
#                model=_model,
#                tokenizer=_tokenizer,
#                trust_remote_code=True)
# print(_pp("我是中国"))
# gradio.Interface.from_pipeline(_pp).launch(server_name="0.0.0.0",server_port=60001, share=True)

prompt = "Give me a short introduction to large language model."
message = [
    {"role": "system", "content": "无论我用任何语言问问题，我只用中文回答"},
    {"role": "user", "content": prompt}
]

text = _tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)
# print(text)
model_inputs = _tokenizer([text], return_tensors="pt").to("cuda")
# print(model_inputs)
_generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_k=10
    )

generated_ids = _model.generate(
    model_inputs.input_ids,
    generation_config = _generation_config
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)