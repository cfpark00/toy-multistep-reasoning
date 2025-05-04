import transformers
import datasets
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os

import sys
sys.path.append("/n/home12/cfpark00/ML/tools")

import lm_tools


if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_n", type=int, default=20)
    parser.add_argument("--n_a", type=int, default=10)
    parser.add_argument("--n_ab", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_problems", type=int, default=4096)
    args = parser.parse_args()

    n_n=args.n_n
    n_a=args.n_a
    n_ab=args.n_ab
    seed=args.seed
    n_problems=args.n_problems

    model_name=f"./data/sft/v2/toy-multistep-v2-nn_{n_n}-na_{n_a}-nab_{n_ab}-seed_{seed}/checkpoint-2504"#final_model"#
    dataset_name=f"cfpark00/toy-multistep-v2-nn_{n_n}-na_{n_a}-nab_{n_ab}-seed_{seed}"

    model=transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer=transformers.AutoTokenizer.from_pretrained(model_name)
    model=model.to(device=device)
    dataset=datasets.load_dataset(dataset_name)

    eval_data={}
    for split in dataset.keys():
        print(f"Split: {split}")
        if split not in ["train", "test_nm_0"]:
            continue
        ds=dataset[split].select(range(n_problems))
        num_maskeds=np.array(ds["num_maskeds"])
        print("average num_maskeds:", num_maskeds.mean())
        prompts=ds["prompt"]
        prompt_token_ids,prompt_attention_masks=lm_tools.tokenize(tokenizer, texts=prompts,chunk_size=1024,padding_side="left",get_attention_mask=True)
        completions=ds["completion"]
        completion_token_ids,completion_attention_masks=lm_tools.tokenize(tokenizer, texts=completions,chunk_size=1024,padding_side="right",get_attention_mask=True)
        #print(prompt_token_ids.shape,prompt_attention_masks.shape,completion_token_ids.shape,completion_attention_masks.shape)

        ds_torch=torch.utils.data.TensorDataset(prompt_token_ids,
                                                prompt_attention_masks,
                                                completion_token_ids,
                                                completion_attention_masks)
        dataloader=torch.utils.data.DataLoader(ds_torch, batch_size=512, shuffle=False,drop_last=False)

        #t=0
        corrects_t0=[]
        for d in tqdm.tqdm(dataloader):
            token_ids=d[0].to("cuda")
            attention_masks=d[1].to("cuda")
            answer_ids=d[2].to("cuda")
            #answer_masks=d[3].to("cuda")
            pred=model.generate(
                input_ids=token_ids,
                attention_mask=attention_masks,
                do_sample=False,
                max_new_tokens=answer_ids.shape[1],
            )[:,len(token_ids[0]):]
            
            error=(pred!=answer_ids)#*(answer_masks==1)
            corrects_t0.extend(torch.all(~error,dim=1).cpu().numpy())
        corrects_t0=np.array(corrects_t0)
        print(f"pass@1 (t=0): {corrects_t0.mean():.4f}")

        dataloader=torch.utils.data.DataLoader(ds_torch, batch_size=16, shuffle=False,drop_last=False)
        corrects_t1=[]
        for d in tqdm.tqdm(dataloader):
            token_ids=d[0].to("cuda")
            attention_masks=d[1].to("cuda")
            answer_ids=d[2].to("cuda").repeat_interleave(32,dim=0)
            #answer_masks=d[3].to("cuda")
            pred=model.generate(
                input_ids=token_ids,
                attention_mask=attention_masks,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=answer_ids.shape[1],
                num_return_sequences=32,
            )[:,len(token_ids[0]):]
            error=(pred!=answer_ids)#*(answer_masks==1)
            corrects_t1.append(torch.all(~error,dim=1).reshape(-1,32).cpu().numpy())
        corrects_t1=np.concatenate(corrects_t1,axis=0)
        print(f"pass@1 (t=1): {corrects_t1.mean():.4f}")
        coverage_t1=np.any(corrects_t1,axis=1).astype(np.float32).mean()
        print(f"pass@32 (t=1): {coverage_t1:.4f}")
        
        eval_data_element={
            "prompt": prompts,
            "completion": completions,
            "num_maskeds": num_maskeds,
            "corrects_t0": corrects_t0,
            "accuracy_t0": corrects_t0,
            "corrects_t1": corrects_t1,
            "accuracy_t1": corrects_t1.mean(),
            "coverage_t1": coverage_t1
        }

        eval_data[split]=eval_data_element
        

    eval_data_path=os.path.join(model_name, "eval_data.pt")
    torch.save(eval_data, eval_data_path)
