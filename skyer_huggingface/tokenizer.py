from transformers import PreTrainedTokenizer
import sentencepiece as spm
import os
import json


class SentencePieceTokenizer(PreTrainedTokenizer):
    
    def __init__(self, model_file, vocab_file, **kwargs):
        
        self._name_or_path = kwargs.get("name_or_path",".")

        self._model_file = model_file
        self._vocab_file = vocab_file

        self.sp = spm.SentencePieceProcessor(model_file=f"{self._name_or_path}/{self._model_file}")
        
        with open(f"{self._name_or_path}/{self._vocab_file}", 'r', encoding='utf-8') as f:
            self.vocab = {line.strip().split('\t')[0]: i for i, line in enumerate(f)}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        super().__init__(**kwargs)

        self.chat_template = """
            {%- for message in messages %}
                {%- if (message.role == "system") %}{{- '<s>system:\n'+ message.content + '</s>\n' }}{%- endif %}
                {%- if (message.role == "user") %}{{- '<s>user:\n'+ message.content + '</s>\n' }}{%- endif %}
            {%- endfor %}
            {{- '<s>assistant:\n' }}
        """

    @property
    def vocab_size(self):
        return len(self.vocab)    

    def _tokenize(self, text):
        """
        用于分词
        """
        return self.sp.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """
        对词进行编码
        """
        # return self.sp.Encode(token)
        return self.vocab.get(token, self.vocab['<unk>'])

    def _convert_id_to_token(self, index):
        """
        对词进行解码
        """
        # return self.sp.Decode(index)
        return self.id_to_token.get(index, '<unk>')

    def convert_tokens_to_string(self, tokens):
        return self.sp.decode(tokens)

    def get_vocab(self):
        return self.vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):

        os.makedirs(save_directory, exist_ok=True)
        os.system(f"cp {self._name_or_path}/tokenizer.py {save_directory}")
        os.system(f"cp {self._name_or_path}/{self._model_file} {save_directory}")
        os.system(f"cp {self._name_or_path}/{self._vocab_file} {save_directory}")

        tokenizer_config = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "tokenizer_class": "SentencePieceTokenizer",
            "model_file": self._model_file,
            "vocab_file": self._vocab_file,
            "auto_map": {
                "AutoTokenizer": [None, "tokenizer.SentencePieceTokenizer"]
            }
        }
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        return self._vocab_file, self._model_file


if __name__ == '__main__':
    # 使用自定义的 Tokenizer
    tokenizer = SentencePieceTokenizer(
        model_file='tokenizer.model', vocab_file='tokenizer.vocab')
    tokenizer.save_pretrained("./cache/skyer")
    # # 测试编码
    # text = '<s>这是一个测试句子</s>'
    # tokens = tokenizer.tokenize(text)
    # print(tokens)
    # # 测试解码
    # decoded_text = tokenizer.convert_tokens_to_string(tokens)
    # print("Decoded text:", decoded_text)

    
