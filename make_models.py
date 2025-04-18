import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import transformers
from transformers import Qwen2Config


if __name__ == "__main__":
    #hyper parameters
    """
    argss=[]
    for n_nodes in [5,10,20]:
        for n_actions in [5,10,20]:
            argss.append({
                "n_nodes":n_nodes,
                "n_actions":n_actions
            })
    """
    argss=[]
    for n_nodes in [10,20,50]:
        for n_actions in [5]:
            argss.append({
                "n_nodes":n_nodes,
                "n_actions":n_actions
            })

    for args in argss:
        n_nodes=args["n_nodes"]
        n_actions=args["n_actions"]
        #
        i_curr=0
        vocab = {
            "<unk>": i_curr,
            "<bos>": i_curr+1,
            "<pad>": i_curr + 2,
            ":": i_curr + 3,
        }
        i_curr += len(vocab)
        for i in range(n_nodes):
            vocab[f"S{i}"] = i_curr
            i_curr += 1
        for i in range(n_actions):
            vocab[f"a{i}"] = i_curr
            i_curr += 1

        model_name=f"cfpark00/toy-multistep-nn_{n_nodes}-na_{n_actions}"

        model = WordLevel(vocab=vocab, unk_token="<unk>")
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token="<pad>",eos_token="<pad>")
        tokenizer.push_to_hub(model_name)

        model_config=Qwen2Config(
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=len(tokenizer.vocab),
        )
        model=transformers.AutoModelForCausalLM.from_config(model_config)
        model.push_to_hub(model_name)