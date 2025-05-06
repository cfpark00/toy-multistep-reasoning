import torch
import tokenizers
import transformers
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datasets
import itertools
import random
import re

# Check if a directed graph is strongly connected using matrix multiplication to compute reachability.
def is_strongly_connected(adj):
    adj = adj.to(torch.int64)
    reach = adj.clone()

    prev = torch.zeros_like(reach)
    # Repeatedly propagate connections until no further changes
    while not torch.equal(prev, reach):
        prev = reach.clone()
        # Propagate reachability using Boolean matrix multiplication
        intermediate = ((reach @ reach)>0).to(dtype=torch.int64)
        reach = reach | intermediate

    # If every node can reach every other node, the graph is strongly connected
    return reach.all().item()

# Convert edge matrix representation (with -1 as masked edges) into an adjacency matrix
def edge_mat_to_adj(edges):
    n = edges.size(0)
    adj = torch.zeros(n, n, dtype=torch.bool)  # Initialize adjacency matrix

    row_indices = torch.arange(n).unsqueeze(1).expand_as(edges)  # Row indices for scatter
    mask = edges != -1  # Only keep non-masked edges
    adj[row_indices[mask], edges[mask]] = True  # Set connections

    return adj

# Count number of paths up to max_length in the graph
def count_paths(adj, max_length=10):
    A = adj.to(torch.int64)
    total_paths = 0
    A_power = A.clone()
    for L in range(1, max_length+1):
        total_paths += A_power.sum().item()
        A_power = torch.matmul(A_power, A)  # Next power of adjacency matrix
    return total_paths


# Generate a valid random edge matrix and a masked version, both strongly connected
def get_adj_edge_mat(n_nodes, n_actions, n_ablate):
    while True:
        # Randomly generate edge matrix (each node has n_actions edges)
        edge_mat = torch.randint(0, n_nodes, (n_nodes, n_actions), dtype=torch.int64)
        adj = edge_mat_to_adj(edge_mat)

        # Generate random mask for ablation
        mask = torch.rand(n_nodes, n_actions)
        sorted_mask = torch.argsort(mask.flatten())
        i, j = torch.unravel_index(sorted_mask[:int(n_ablate)], mask.shape)
        mask[i, j] = -1
        mask = mask >= 0

        # Apply mask to edge matrix
        edge_mat_masked = edge_mat.clone()
        edge_mat_masked[~mask] = -1
        adj_masked = edge_mat_to_adj(edge_mat_masked)

        # Ensure both graphs are strongly connected and all rows have valid edges
        if is_strongly_connected(adj) and is_strongly_connected(adj_masked):
            assert edge_mat.any(1).all(), "edge mat has no edges"
            assert edge_mat_masked.any(1).all(), "masked edge mat has no edges"
            break
    return edge_mat, edge_mat_masked, adj, adj_masked

def make_graph_bank(n_graphs: int,
                    n_nodes: int,
                    n_actions: int,
                    n_ablate: int):
    """
    Returns two lists (len = n_graphs):
        - edge_mats[g]          : full edge matrix for graph g
        - edge_mats_masked[g]   : masked  edge matrix for graph g
    """
    edge_mats, edge_mats_masked = [], []
    for g in range(n_graphs):
        em, em_masked, *_ = get_adj_edge_mat(
            n_nodes, n_actions, n_ablate
        )
        edge_mats.append(em)
        edge_mats_masked.append(em_masked)
    return edge_mats, edge_mats_masked


# Generate a dataset of graph navigation problems
def generate_problems(edge_mat, edge_mat_masked, n_problems):
    n_nodes = edge_mat.size(0)
    data = {}

    for key, edge_mat_used in zip(["train", "test_rl", "test"], [edge_mat_masked, edge_mat, edge_mat]):
        edge_mat_used = edge_mat_used.clone()
        i_starts = torch.randint(0, n_nodes, (n_problems,), dtype=torch.int64)
        lengths = torch.randint(1, 10, (n_problems,), dtype=torch.int64)
        paths = []
        actionss = []
        prompts = []
        completions = []
        num_maskeds = []

        for i in tqdm.trange(n_problems):
            i_curr = i_starts[i].item()
            length = lengths[i]
            path = [i_curr]
            actions = []

            # Generate a random path of fixed length
            for j in range(length):
                avail_actions = torch.where(edge_mat_used[i_curr] != -1)[0]
                # Randomly select an action
                i_action = torch.randint(0, avail_actions.size(0), (1,)).item()
                action = avail_actions[i_action].item()
                i_curr = edge_mat_used[i_curr, action].item()

                actions.append(action)
                path.append(i_curr)

            paths.append(path)
            actionss.append(actions)
            num_maskeds.append((torch.tensor(actions) == -1).sum().item())  # Count masked actions

        # Store data for this split
        data[key] = {
            "paths": paths,
            "actions": actionss,
            "num_maskeds": num_maskeds
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


def generate_problems_splitted_multi_fillers(
        edge_mats, edge_mats_masked,
        n_problems_train: int,
        n_problems_rl:    int,
        n_problems_test:  int,
        *,
        max_path_len        : int   = 9,

        # ───────── filler hyper-parameters ────────────────────────────────
        p_fg                : float = 0.60,   # prob. insert Type-1  FGi
        p_fg_match          : float = 0.90,   # …and use i == g with this prob.

        p_fs                : float = 0.60,   # prob. insert Type-2  FSi
        p_fs_match          : float = 0.90,   # …and use i == start-state

        p_fa                : float = 0.60,   # prob. insert Type-3  FAi between any two actions
        p_fa_correct        : float = 0.90,   # …and use i == *true* next-state

        balance_graphs      : bool  = True,   # round-robin across graphs
):
    """
    Multi-graph data generator with three kinds of *filler* tokens that create
    spurious correlations:

        Type-1:  FGi  (likely but not always i == graph-id)
        Type-2:  FSi  (likely but not always i == start-state)
        Type-3:  FAi  (inserted between actions; likely but not always i ==
                       *actual* next state)

    g_idx   (graph index)   – unchanged from earlier version.
    """
    assert len(edge_mats) == len(edge_mats_masked)
    n_graphs   = len(edge_mats)

    # ---------- split bookkeeping ------------------------------------------
    split_keys = (
        ["train"] +
        [f"rl_nm_{k}" for k in range(1, 2)] +
        [f"test_nm_{k}" for k in range(4)]
    )
    n_per_split = {
        "train":      n_problems_train,
        **{f"rl_nm_{k}": n_problems_test for k in range(1, 4)},
        **{f"test_nm_{k}": n_problems_test for k in range(5)},
    }
    data = {k: dict(paths=[], actionss=[], prompts=[], completions=[],
                    num_maskeds=[], g_idx=[]) for k in split_keys}

    # graph iterator
    if balance_graphs:
        graph_cycle = itertools.cycle(range(n_graphs))
    else:
        graph_cycle = (random.randrange(n_graphs)
                       for _ in itertools.count())

    # ---------- main loop ---------------------------------------------------
    for key in split_keys:
        n_targets          = n_per_split[key]
        if key == "train":
            num_maskeds_allowed = [0]
            p_fg_match_used = p_fg_match
            p_fs_match_used = p_fs_match
            p_fa_correct_used = p_fa_correct
        elif "rl" in key:
            num_maskeds_allowed = [int(key.split("_nm_")[1])]
            p_fg_match_used = p_fg_match
            p_fs_match_used = p_fs_match
            p_fa_correct_used = p_fa_correct
        else:
            num_maskeds_allowed = [int(key.split("_nm_")[1])]
            p_fg_match_used = p_fg_match
            p_fs_match_used = p_fg_match
            p_fa_correct_used = p_fg_match

        
        pbar = tqdm.trange(n_targets, desc=f"Split: {key}")

        while len(data[key]["paths"]) < n_targets:
            g  = next(graph_cycle)                 # graph id
            em, em_mask = edge_mats[g], edge_mats_masked[g]
            n_nodes, n_actions = em.size()

            start   = int(torch.randint(0, n_nodes, ()).item())
            L       = int(torch.randint(1, max_path_len + 1, ()).item())
            path, actions = [start], []

            i_curr = start
            for _ in range(L):
                i_action = int(torch.randint(0, n_actions, ()).item())
                i_next   = int(em[i_curr, i_action].item())
                actions.append(i_action)
                path.append(i_next)
                i_curr = i_next

            # how many actions are masked in masked-graph?
            masked_hits = em_mask[torch.tensor(path[:-1]),
                                  torch.tensor(actions)]
            num_masked  = int((masked_hits == -1).sum())
            if num_masked not in num_maskeds_allowed:
                continue  # reject sample

            # ------------- build prompt with fillers -------------------------
            parts = []

            # ---------- FGi and Gg token placement (Type-1 filler) ----------------
            fg_token = f"FG{g}" if random.random() < p_fg_match_used else f"FG{random.randrange(n_graphs)}"
            if random.random() < p_fg:
                if random.random() < 0.5:
                    parts.append(fg_token)
                    parts.append(f"G{g}")
                else:
                    parts.append(f"G{g}")
                    parts.append(fg_token)
            else:
                parts.append(f"G{g}")

            # ---------- FSi and S<start> token placement (Type-2 filler) ----------
            sg_token = f"FS{start}" if random.random() < p_fs_match_used else f"FS{random.randrange(n_nodes)}"
            if random.random() < p_fs:
                if random.random() < 0.5:
                    parts.append(sg_token)
                    parts.append(f"S{start}")
                else:
                    parts.append(f"S{start}")
                    parts.append(sg_token)
            else:
                parts.append(f"S{start}")

            # actions (+ optional Type-3 FAi)
            for j, a in enumerate(actions):
                parts.append(f"A{a}")
                if j < len(actions) - 1 and random.random() < p_fa:
                    # decide whether FAi is 'correct'
                    if random.random() < p_fa_correct_used:
                        ag_i = path[j + 1]             # the *true* next state
                    else:
                        ag_i = random.randrange(n_nodes)
                    parts.append(f"FA{ag_i}")

            prompt = " ".join(parts) + " : "
            completion = " ".join(f"S{s}" for s in path) + " "

            # store
            for fld, val in [("paths", path),
                             ("actionss", actions),
                             ("prompts", prompt),
                             ("completions", completion),
                             ("num_maskeds", num_masked),
                             ("g_idx", g)]:
                data[key][fld].append(val)
            pbar.update(1)
        pbar.close()

    # persist graph definitions once
    data["graphs"] = {
        "edge_mats":        [em.tolist() for em in edge_mats],
        "edge_mats_masked": [mm.tolist() for mm in edge_mats_masked],
    }
    return data

def space_out(text):
    text=text.replace("FG","F G")
    text=text.replace("FS","F S")
    text=text.replace("FA","F A")
    text=text.replace("G","G ")
    text=text.replace("S","S ")
    text=text.replace("A","A ")
    text=text.strip()
    return text

def preprocess(element):
    data={}
    for key in ["prompts","completions","texts"]:
        data[key]=space_out(element[key])
    data["prompt"]=data["prompts"]
    data["completion"]=data["completions"]
    data["text"]=data["texts"]
    return data

if __name__ == "__main__":
    #hyper parameters
    n_problems_train = 100_000 #
    n_problems_rl = 10_000 #per 1 out 2 out 3 out 4out
    n_problems_test = 5_000 #per 1 out 2 out 3 out 4out
    gen_func=generate_problems_splitted_multi_fillers
    gen_func_kwargs={
        "n_problems_train":n_problems_train,
        "n_problems_rl":n_problems_rl,
        "n_problems_test":n_problems_test,
        
        "max_path_len": 9,
        "p_fg": 0.7,   # prob. insert Type-1  FGi
        "p_fg_match": 0.7,   # …and use i == g with this prob.

        "p_fs": 0.7,   # prob. insert Type-2  FSi
        "p_fs_match": 0.7,   # …and use i == start-state

        "p_fa": 0.7,   # prob. insert Type-3  FAi between any two actions
        "p_fa_correct": 0.7,   # …and use i == *true* next-state

        "balance_graphs": True,   # round-robin across graphs
    }
    
    argss=[]
    for n_nodes in [20]:
        for n_actions in [10]:
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
        # generate edge mat
        # edge_mat, edge_mat_masked, adj, adj_masked = get_adj_edge_mat(n_nodes, n_actions, n_ablate)
        # total_paths = count_paths(adj, max_length=10)
        # print("Total unique paths of length 1 to 10:", total_paths)
        
        # generate graph bank
        edge_mats, edge_mats_masked = make_graph_bank(
            n_graphs=10, n_nodes=n_nodes, n_actions=n_actions, n_ablate=n_ablate
        )
        # generate problems
        data = gen_func(edge_mats, edge_mats_masked, **gen_func_kwargs)

        #torch.save(data, f"./data/raw_data/data-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-seed_{seed}.pt")
        #data=torch.load(f"./data/raw_data/data-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-seed_{seed}.pt",weights_only=False)
        dataset_data = {}
        for key in data.keys():
            if key == "graphs":
                continue
            prompts=[p for p in data[key]['prompts']]
            completions=[c for c in data[key]['completions']]
            num_maskeds=data[key]['num_maskeds']
            texts=[p+c for p, c in zip(prompts,completions)]
            data_={
                "prompt": prompts,
                "completion": completions,
                'text': texts,
                "prompts": prompts,
                "completions": completions,
                "num_maskeds": num_maskeds,
                'texts': texts,   
            }
            dataset_data[key]=datasets.Dataset.from_dict(data_)
        dataset=datasets.DatasetDict(dataset_data)
        dataset=dataset.map(preprocess)
        #push to hub
        dataset_name=f"sunnytqin/toy-multistep-v2-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-seed_{seed}"
        dataset.push_to_hub(dataset_name)
        print(f"Dataset pushed to hub: {dataset_name}")
        print("\n\n")