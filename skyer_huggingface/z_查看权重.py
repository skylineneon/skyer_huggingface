import torch

import sentencepiece as sp
text=torch.load('/root/workspace/skyer_huggingface/weight/mp_rank_00_model_states.pt')['module']
print(text)
