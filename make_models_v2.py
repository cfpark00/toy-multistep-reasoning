import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import transformers
from transformers import Qwen2Config


if __name__ == "__main__":
    n_nodes=20
    n_actions=10
    i_curr=0
    vocab = {
        "<unk>": i_curr,
        "<bos>": i_curr+1,
        "<pad>": i_curr + 2,
        ":": i_curr + 3,
        "G": i_curr + 4,
        "S": i_curr + 5,
        "A": i_curr + 6,
        "F": i_curr + 7,
    }
    i_curr += len(vocab)
    for i in range(max(n_nodes,n_actions)):
        vocab[f"{i}"] = i_curr
        i_curr += 1

    model_name=f"sunnytqin/toy-multistep-v2-nn_{n_nodes}-na_{n_actions}"

    tokenizer_model = WordLevel(vocab=vocab, unk_token="<unk>")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token="<pad>",eos_token="<pad>")
    tokenizer.push_to_hub(model_name)

<<<<<<< Updated upstream
    model_config=Qwen2Config(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
=======
    model_config = Qwen2Config(
        hidden_size=256,              # narrow
        intermediate_size=1024,        # standard MLP size (4x hidden)
        num_hidden_layers=16,         # deeper
        num_attention_heads=4,        # matches 128 hidden (head_size=32)
>>>>>>> Stashed changes
        num_key_value_heads=4,
        vocab_size=len(tokenizer.vocab),
)
    model=transformers.AutoModelForCausalLM.from_config(model_config)
    model.push_to_hub(model_name)