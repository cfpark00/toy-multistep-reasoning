import torch
import tokenizers
import transformers
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datasets

def is_strongly_connected(adj):
    adj = adj.to(torch.int64)
    reach = adj.clone()

    prev = torch.zeros_like(reach)
    while not torch.equal(prev, reach):
        prev = reach.clone()
        intermediate = ((reach @ reach)>0).to(dtype=torch.int64)#propagate
        reach = reach | intermediate
        
    if not reach.all():
        return False
    return True

def edge_mat_to_adj(edges):
    n=edges.size(0)
    # Create an n x n Boolean tensor initialized with False.
    adj = torch.zeros(n, n, dtype=torch.bool)
    
    # Create a row index for each entry in the edge matrix.
    # This will be an n x k matrix where each row i is filled with i.
    row_indices = torch.arange(n).unsqueeze(1).expand_as(edges)

    mask= edges != -1
    
    # Use advanced indexing to set the corresponding entries to True.
    adj[row_indices[mask], edges[mask]] = True
    
    return adj

def count_paths(adj, max_length=10):
    # Convert the boolean adj matrix to int for multiplication
    A = adj.to(torch.int64)
    total_paths = 0
    A_power = A.clone()
    for L in range(1, max_length+1):
        total_paths += A_power.sum().item()
        A_power = torch.matmul(A_power, A)
    return total_paths


def get_adj_edge_mat(n_nodes, n_actions, n_ablate):
    while True:
        edge_mat=torch.randint(0,n_nodes,(n_nodes, n_actions),dtype=torch.int64)
        adj=edge_mat_to_adj(edge_mat)
        mask=torch.rand(n_nodes,n_actions)
        sorted_mask=torch.argsort(mask.flatten())
        i,j=torch.unravel_index(sorted_mask[:int(n_ablate)], mask.shape)
        mask[i,j]=-1
        mask=mask>=0
        edge_mat_masked=edge_mat.clone()
        edge_mat_masked[~mask]=-1
        adj_masked=edge_mat_to_adj(edge_mat_masked)
        if is_strongly_connected(adj) and is_strongly_connected(adj_masked):
            assert edge_mat.any(1).all(), "edge mat has no edges"
            assert edge_mat_masked.any(1).all(), "masked edge mat has no edges"
            break
    return edge_mat, edge_mat_masked, adj, adj_masked

def generate_problems(edge_mat, edge_mat_masked, n_problems):
    n_nodes=edge_mat.size(0)
    data={}
    for key,edge_mat_used in zip(["train","test_rl","test"],[edge_mat_masked,edge_mat,edge_mat]):
        edge_mat_used=edge_mat_used.clone()
        i_starts=torch.randint(0,n_nodes,(n_problems,),dtype=torch.int64)
        lengths=torch.randint(1,10,(n_problems,),dtype=torch.int64)
        paths=[]
        actionss=[]
        prompts=[]
        completions=[]
        num_maskeds=[]
        for i in tqdm.trange(n_problems):
            i_curr=i_starts[i].item()
            length=lengths[i]
            path=[i_curr]
            actions=[]
            for j in range(length):
                avail_actions=torch.where(edge_mat_used[i_curr]!= -1)[0]
                #randomly select an action
                i_action=torch.randint(0,len(avail_actions),(1,),dtype=torch.int64)
                i_action=avail_actions[i_action].item()
                #take the action
                i_next=edge_mat_used[i_curr,i_action].item()
                path.append(i_next)
                actions.append(i_action)
                i_curr=i_next
            action_dests=edge_mat_masked[torch.tensor(path[:-1]),torch.tensor(actions)]
            num_masked=torch.sum(action_dests == -1)
            #print(key, "num masked:", num_masked.item())
            assert len(path) == length+1 and len(actions) == length, "path and actions have different lengths"
            paths.append(path)
            actionss.append(actions)
            prompt="S"+str(path[0])+" "
            prompt+="".join(["a"+str(i)+" " for i in actions])
            prompt+=": "#find
            completion="".join(["S"+str(i)+" " for i in path])
            prompts.append(prompt)
            completions.append(completion)
            num_maskeds.append(num_masked.item())
            #break
        data[key] = {
            'paths': paths,
            'actionss': actionss,
            'prompts': prompts,
            'completions': completions,
            'num_maskeds': num_maskeds,
            'edge_mat': edge_mat_used.tolist(),
        }
    return data

def generate_problems_splitted(edge_mat, edge_mat_masked, n_problems_train,n_problems_rl,n_problems_test):
    n_nodes=edge_mat.size(0)
    n_actions=edge_mat.size(1)
    data={}
    keys=["train","rl_nm_0","rl_nm_1","rl_nm_2","rl_nm_3","rl_nm_4","test_nm_0","test_nm_1","test_nm_2","test_nm_3","test_nm_4"]
    for key in keys:
        if key=="train":
            n_problems=n_problems_train
            num_maskeds_allowed=[0]
        else:
            assert "_nm_" in key
            assert "rl_" in key or "test_" in key
            if "rl_" in key:
                n_problems=n_problems_rl
            else:
                assert "test_" in key
                n_problems=n_problems_test
            num_maskeds_allowed=[int(key.split("_nm_")[1])]
        paths=[]
        actionss=[]
        prompts=[]
        completions=[]
        num_maskeds=[]
        pbar=tqdm.trange(n_problems,desc=f"Split: {key}")
        while len(paths) < n_problems:
            i_curr=torch.randint(0,n_nodes,(1,),dtype=torch.int64).item()
            length=torch.randint(1,10,(1,),dtype=torch.int64).item()
            path=[i_curr]
            actions=[]
            for j in range(length):
                i_action=torch.randint(0,n_actions,(1,),dtype=torch.int64).item()
                #take the action
                i_next=edge_mat[i_curr,i_action].item()
                path.append(i_next)
                actions.append(i_action)
                i_curr=i_next
            action_dests=edge_mat_masked[torch.tensor(path[:-1]),torch.tensor(actions)]
            num_masked=torch.sum(action_dests == -1)
            if num_masked.item() not in num_maskeds_allowed:
                continue
            assert len(path) == length+1 and len(actions) == length, "path and actions have different lengths"
            prompt="S"+str(path[0])+" "
            prompt+="".join(["a"+str(i)+" " for i in actions])
            prompt+=": "#find
            completion="".join(["S"+str(i)+" " for i in path])
            paths.append(path)
            actionss.append(actions)
            prompts.append(prompt)
            completions.append(completion)
            num_maskeds.append(num_masked.item())
            pbar.update(1)
        pbar.close()
            #break
        data[key] = {
            'paths': paths,
            'actionss': actionss,
            'prompts': prompts,
            'completions': completions,
            'num_maskeds': num_maskeds,
            'edge_mat': edge_mat.tolist(),
            'edge_mat_masked': edge_mat_masked.tolist(),
        }
    return data


if __name__ == "__main__":
    #hyper parameters
    """
    n_problems=262144
    gen_func=generate_problems
    gen_func_kwargs={
        "n_problems":n_problems
    }
    argss=[]
    for n_nodes in [5,10,20]:
        for n_actions in [5,10,20]:
            for n_ablate in ["0.1","0.2","0.3"]:
                for seed in [0,1,2]:
                    argss.append({
                        "n_nodes":n_nodes,
                        "n_actions":n_actions,
                        "n_ablate":n_ablate,
                        "seed":seed
                    })
    """
    n_problems_train=65536#
    n_problems_rl=8192#per 1 out 2 out 3 out 4out
    n_problems_test=8192#per 1 out 2 out 3 out 4out
    gen_func=generate_problems_splitted
    gen_func_kwargs={
        "n_problems_train":n_problems_train,
        "n_problems_rl":n_problems_rl,
        "n_problems_test":n_problems_test
    }
    argss=[]
    for n_nodes in [10,20,50]:
        for n_actions in [5]:
            for n_ablate in ["0.2"]:
                for seed in [0]:
                    argss.append({
                        "n_nodes":n_nodes,
                        "n_actions":n_actions,
                        "n_ablate":n_ablate,
                        "seed":seed
                    })

    for args in argss:
        print(args)
        n_nodes=args["n_nodes"]
        n_actions=args["n_actions"]
        n_ablate=args["n_ablate"]
        seed=args["seed"]
        #
        n_edges=n_nodes*n_actions
        if type(n_ablate) == str:
            n_ablate = int(float(n_ablate)*n_edges)
        print("n_ablate", n_ablate)

        #set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        #generate edge mat
        edge_mat, edge_mat_masked, adj, adj_masked = get_adj_edge_mat(n_nodes, n_actions, n_ablate)
        total_paths = count_paths(adj, max_length=10)
        print("Total unique paths of length 1 to 10:", total_paths)
        #generate problems
        data = gen_func(edge_mat, edge_mat_masked, **gen_func_kwargs)
        #torch.save(data, f"./data/raw_data/data-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-seed_{seed}.pt")
        #data=torch.load(f"./data/raw_data/data-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-seed_{seed}.pt",weights_only=False)
        dataset_data={}
        for key in data.keys():
            prompts=data[key]['prompts']
            completions=data[key]['completions']
            num_maskeds=data[key]['num_maskeds']
            texts=[p+c for p,c in zip(prompts,completions)]
            data_={
                "prompts": prompts,
                "completions": completions,
                "num_maskeds": num_maskeds,
                'texts': texts,
            }
            dataset_data[key]=datasets.Dataset.from_dict(data_)
        dataset=datasets.DatasetDict(dataset_data)
        #push to hub
        dataset_name=f"cfpark00/toy-multistep-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-seed_{seed}"
        dataset.push_to_hub(dataset_name)
        print(f"Dataset pushed to hub: {dataset_name}")
        print("\n\n")