import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import modeling_auto


def precompute_freqs_cis(dim, end, theta=50000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def reshape_for_broadcast(freqs_cis, x):
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None]


class Attention(nn.Module):

    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 cache_max_batch_size,
                 cache_max_seq_len
                 ):
        super().__init__()

        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads

        self._group = n_q_heads // n_kv_heads

        self._head_size = hide_dim // self._n_q_heads

        self._qw = nn.Linear(input_dim, self._head_size*self._n_q_heads)
        self._kw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._vw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._ow = nn.Linear(hide_dim, input_dim)

        self._cache_max_batch_size = cache_max_batch_size
        if self._cache_max_batch_size is not None:
            _cache_k = torch.zeros((cache_max_batch_size,
                                    cache_max_seq_len,
                                    n_kv_heads,
                                    self._head_size,
                                    ))
            self.register_buffer("_cache_k", _cache_k, persistent=False)

            _cache_v = torch.zeros((cache_max_batch_size,
                                    cache_max_seq_len,
                                    n_kv_heads,
                                    self._head_size,
                                    ))
            self.register_buffer("_cache_v", _cache_v, persistent=False)

    def forward(self, x, freq_cis, start_pos):
        _bn, _seq, _ = x.shape
        _dk = self._head_size**0.5

        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)

        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = apply_rotary_emb(_q, freq_cis[start_pos:start_pos+_seq])
        _k = apply_rotary_emb(_k, freq_cis[start_pos:start_pos+_seq])

        if self._cache_max_batch_size is not None:
            self._cache_k[:_bn, start_pos: start_pos + _seq] = _k
            self._cache_v[:_bn, start_pos: start_pos + _seq] = _v

            _k = self._cache_k[:_bn, : start_pos + _seq]
            _v = self._cache_v[:_bn, : start_pos + _seq]

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        _k = _k[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, start_pos+_seq, self._head_size)
        _v = _v[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, start_pos+_seq, self._head_size)

        # _causul = torch.ones(_seq, _seq)
        # _causul = torch.triu(_causul, diagonal=1)
        # _causul[_causul == 1] = -torch.inf
        # _causul = _causul.to(x.device)

        # _score = _q@_k.permute(0, 1, 3, 2)/_dk
        # print(_score.shape)
        # _score = torch.softmax(_score+_causul, dim=-1)

        # _o = _score@_v
        # print(_q.shape, _k.shape, _v.shape)

        if start_pos == 0:
            _o = F.scaled_dot_product_attention(
                _q, _k, _v, attn_mask=None, is_causal=True)
        else:
            _o = F.scaled_dot_product_attention(
                _q, _k, _v, attn_mask=None, is_causal=False)

        _o = _o.permute(0, 2, 1, 3)
        _o = _o.reshape(_bn, _seq, -1)

        return self._ow(_o)


class FFN(nn.Module):

    def __init__(self, input_dim, hide_dim):
        super().__init__()

        self._w0 = nn.Linear(input_dim, hide_dim)
        self._w1 = nn.Linear(input_dim, hide_dim)
        self._w2 = nn.Linear(hide_dim, input_dim)

        self._gate = nn.SiLU()

    def forward(self, x):
        _o0 = self._w0(x)
        _o1 = self._w1(x)
        _g = self._gate(_o1)
        _og = _o0*_g
        return self._w2(_og)


class RMSNormal(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        return self._w*x/((x**2).mean(-1, keepdim=True)**0.5+1e-6)


class TransformerLayer(nn.Module):

    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 cache_max_batch_size,
                 cache_max_seq_len):
        super().__init__()

        self._att_norm = RMSNormal(input_dim)

        self._att_layer = Attention(input_dim,
                                    hide_dim,
                                    n_q_heads,
                                    n_kv_heads,
                                    cache_max_batch_size,
                                    cache_max_seq_len)

        self._ffn_norm = RMSNormal(input_dim)

        self._ffn_layer = FFN(input_dim,
                              hide_dim)

    def forward(self, x, freq_cis, start_pos):
        _x = x
        _x = self._att_norm(_x)
        _x = self._att_layer(_x, freq_cis, start_pos)

        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)

        _y = _y + _x

        return _y


class TransformerDecoder(nn.Module):

    def __init__(self,
                 num_layers,  # 解码器的层数
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 max_len,
                 cache_max_batch_size=None,
                 cache_max_seq_len=None
                 ):
        super().__init__()

        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,
                              hide_dim,
                              n_q_heads,
                              n_kv_heads,
                              cache_max_batch_size,
                              cache_max_seq_len) for _ in range(num_layers)]
        )
        self._out_norm = RMSNormal(input_dim)
        _freq_cis = precompute_freqs_cis(hide_dim//n_q_heads, max_len)

        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x, start_pos):
        _x = x
        for _layer in self._layers:
            _x = _layer(_x, self.freq_cis, start_pos)
        # return _x
        return self._out_norm(_x)


class SkyerConfig(PretrainedConfig):
    model_type = "skyer"  # 自定义模型类型

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = kwargs.get("num_layers")
        self.input_dim = kwargs.get("input_dim")
        self.hide_dim = kwargs.get("hide_dim")
        self.n_q_heads = kwargs.get("n_q_heads")
        self.n_kv_heads = kwargs.get("n_kv_heads")
        self.max_pos_len = kwargs.get("max_pos_len")
        self.vocab_size = kwargs.get("vocab_size")
        self.cache_max_batch_size = kwargs.get("cache_max_batch_size")
        self.cache_max_seq_len = kwargs.get("cache_max_seq_len")

        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 3
        # 配置映射
        self.auto_map = {
            "AutoModelForCausalLM": "model.SkyerModel",  # 因为是因果推理模型，所以AutoModel后加CausalLM
            # 如果是普通的模型，写AutoModel就行了
            "AutoConfig": "model.SkyerConfig"
        }

# 创建模型


class SkyerModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self._cache_max_batch_size = config.cache_max_batch_size

        self._emb = nn.Embedding(config.vocab_size, config.input_dim)
        # 加载配置文件中的参数
        self._tf_layer = TransformerDecoder(
            num_layers=config.num_layers,
            input_dim=config.input_dim,
            hide_dim=config.hide_dim,
            n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads,
            max_len=config.max_pos_len,
            cache_max_batch_size=config.cache_max_batch_size,
            cache_max_seq_len=config.cache_max_seq_len,
        )

    def _forward(self, input_ids, start_pos):
        _tokens = self._emb(input_ids)
        _features = self._tf_layer(_tokens, start_pos)
        _outputs = _features@self._emb.weight.T
        return _outputs

    # 训练
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):  # Huggingface规定形式
        _logits = self._forward(input_ids, 0)

        _labels = labels
        _loss = None
        if _labels is not None:
            # 兼容习惯小于0的均覆盖，一般设置为-100，
            _labels[_labels < 0] = self.config.pad_token_id
            _o = _logits[:, :-1].reshape(-1, self.config.vocab_size)
            _t = _labels[:, 1:].reshape(-1)

            _loss = F.cross_entropy(
                _o, _t, ignore_index=self.config.pad_token_id)

        return CausalLMOutputWithPast(  # 不同任务输出不同，要看任务是哪一种类型
            loss=_loss,
            logits=_logits,  # 净输出，不是加softmax的输出
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # 定义默认参数
    _generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_k=5
    )
    # 推理

    def _generate(self, ids, start_pos, attention_mask, _generation_config, **kwargs):

        _outputs = self._forward(ids, start_pos)
        _output = _outputs[:, -1]
        _weight, _indices = torch.topk(
            _output, _generation_config.top_k, dim=-1)
        _probs = self._tsoftmax(_weight, _generation_config.temperature)
        _selected_indices = torch.multinomial(_probs, 1)
        _id = torch.gather(_indices, dim=-1, index=_selected_indices)
        return _id

    # 传入inputs_ids,并生成一个generation_config配置文件
    def generate(self, input_ids, generation_config=_generation_config, **kwargs):
        _ids = input_ids
        _id = self._generate(input_ids, 0, generation_config)

        for _start_pos in range(_ids.shape[1], self.config.cache_max_seq_len):
            _id = self._generate(_id, _start_pos, generation_config)
            if _id == self.config.eos_token_id:
                break
            # yield _id #如果一个字一个字输出
            _ids = torch.cat((_ids, _id), dim=-1)
        return _ids

    def get_input_embeddings(self):
        return self._emb

    def set_input_embeddings(self, value):
        self._emb = value

    @staticmethod
    def _tsoftmax(xs, temp):
        _o = xs-xs.mean()
        return torch.exp(_o/temp)/(torch.exp(_o/temp).sum(-1)+1e-5)


AutoConfig.register("skyer", SkyerConfig)  # 注册了一个名为 "skyer" 的新配置类 SkyerConfig
# 在 modeling_auto 模块的 MODEL_FOR_CAUSAL_LM_MAPPING_NAMES 字典中添加了一个映射，将 "skyer" 映射到 "SkyerModel"。
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["skyer"] = "SkyerModel"


if __name__ == "__main__":
    import os
    config = SkyerConfig(num_layers=20,
                         input_dim=2048,
                         hide_dim=1536,
                         n_q_heads=24,
                         n_kv_heads=12,
                         max_pos_len=4096,
                         vocab_size=30000,
                         cache_max_batch_size=1,
                         cache_max_seq_len=128,
                         )

    # 创建模型
    model = SkyerModel(config)
    model.load_state_dict(torch.load(
        f"/root/workspace/skyer_huggingface/weight/mp_rank_00_model_states.pt", weights_only=False)["module"])

    # 保存模型
    save_directory = "./cache/skyer"
    model.save_pretrained(save_directory,)
    os.system(f"cp model.py {save_directory}")

    print("Model and configuration saved successfully.")
