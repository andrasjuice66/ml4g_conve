#!/usr/bin/env python
import json
import os
from os.path import join
from collections import defaultdict

def preprocess_data(dataset_name):
    """
    Reads train.txt, valid.txt, test.txt from data/{dataset_name}/
    Generates JSON files with integer IDs for e1, rel, e2, e2_multi1, e2_multi2:
        e1rel_to_e2_train.json
        e1rel_to_e2_ranking_dev.json
        e1rel_to_e2_ranking_test.json
        e1rel_to_e2_full.json
    Also writes out entity2id.json, relation2id.json for reference.
    """

    base_path = f'/Users/andrasjoos/Documents/AI_masters/Period_9/ML4G/Project/LinkPred/data/{dataset_name}'
    train_file = join(base_path, 'train.txt')
    valid_file = join(base_path, 'valid.txt')
    test_file  = join(base_path, 'test.txt')

    # 1) Read raw triples from train, valid, test
    files = [train_file, valid_file, test_file]
    names = ['train.txt', 'valid.txt', 'test.txt']
    data = {}
    for name, fpath in zip(names, files):
        with open(fpath, 'r') as f:
            lines = [line.strip().split('\t') for line in f]
            # lines => list of [e1, rel, e2]
        data[name] = lines

    print("Debug: Number of lines in each split:")
    for split_name in data:
        print(f"  {split_name}: {len(data[split_name])} triples")

    # 2) We'll collect all unique entities and relations (including reverse)
    all_entities = set()
    all_relations = set()

    # 3) We also keep the adjacency structures we had
    label_graph = defaultdict(set)   # (e1, rel) -> set of e2
    train_graph = {n: defaultdict(set) for n in names}
    test_cases  = {n: [] for n in names}

    # Helper to add edges
    def add_edge(e1, rel, e2, split_name):
        rrev = rel + '_reverse'
        # Forward
        label_graph[(e1, rel)].add(e2)
        # Reverse
        label_graph[(e2, rrev)].add(e1)

        train_graph[split_name][(e1, rel)].add(e2)
        train_graph[split_name][(e2, rrev)].add(e1)

        test_cases[split_name].append((e1, rel, e2))

        # Collect entities and relations
        all_entities.add(e1)
        all_entities.add(e2)
        all_relations.add(rel)
        all_relations.add(rrev)

    # 4) Populate structures + collect all entities/relations
    for name in names:
        for (e1, rel, e2) in data[name]:
            add_edge(e1, rel, e2, name)

    # --- CHANGED: Build entity2id and relation2id ---
    all_entities = sorted(list(all_entities))
    all_relations = sorted(list(all_relations))

    entity2id = {ent: idx for idx, ent in enumerate(all_entities)}
    relation2id = {rel: idx for idx, rel in enumerate(all_relations)}

    # 5) Functions to write out JSON with integer IDs
    def write_training_graph(graph_dict, out_path):
        """
        For each (e1, rel), write a JSON line with integer IDs for e1, rel, e2_multi1.
        We'll set e2= -1 (dummy) and e2_multi2= [] for training.
        """
        with open(out_path, 'w') as f:
            for (e1, rel), tails in graph_dict.items():
                e1_id  = entity2id[e1]
                rel_id = relation2id[rel]
                e2_list = list(tails)
                e2_id_list = [entity2id[x] for x in e2_list]

                data_point = {
                    'e1': e1_id,
                    'rel': rel_id,
                    'rel_eval': -1,   # Not used in training
                    'e2': -1,         # No single gold for training
                    'e2_multi1': e2_id_list,
                    'e2_multi2': []
                }
                f.write(json.dumps(data_point) + '\n')

    def write_evaluation_graph(cases, graph, out_path):
        """
        For each triple (e1, rel, e2), save integer IDs.
          e2_multi1 = all valid tails for (e1, rel)
          e2_multi2 = all valid tails for the reverse (e2, rel+'_reverse')
        """
        with open(out_path, 'w') as f:
            for (e1, rel, e2) in cases:
                e1_id  = entity2id[e1]
                rel_id = relation2id[rel]
                rrev   = rel + '_reverse'
                rrev_id = relation2id[rrev]

                e2_id  = entity2id[e2]
                e2_list_1 = graph[(e1, rel)]
                e2_list_2 = graph[(e2, rrev)]

                e2_id_list_1 = [entity2id[x] for x in e2_list_1]
                e2_id_list_2 = [entity2id[x] for x in e2_list_2]

                data_point = {
                    'e1': e1_id,
                    'rel': rel_id,
                    'rel_eval': rrev_id,   # integer ID of the reverse relation
                    'e2': e2_id,
                    'e2_multi1': e2_id_list_1,
                    'e2_multi2': e2_id_list_2,
                }
                f.write(json.dumps(data_point) + '\n')

    # 6) Write out the four JSON files
    out_train = join(base_path, 'e1rel_to_e2_train.json')
    out_dev   = join(base_path, 'e1rel_to_e2_ranking_dev.json')
    out_test  = join(base_path, 'e1rel_to_e2_ranking_test.json')
    out_full  = join(base_path, 'e1rel_to_e2_full.json')

    # - Train
    write_training_graph(train_graph['train.txt'], out_train)

    # - Dev & Test
    write_evaluation_graph(test_cases['valid.txt'], label_graph, out_dev)
    write_evaluation_graph(test_cases['test.txt'],  label_graph, out_test)

    # - Full: mimic training style but for all (e1, rel) in the entire label_graph
    with open(out_full, 'w') as f:
        for (e1, rel), tails in label_graph.items():
            e1_id  = entity2id[e1]
            rel_id = relation2id[rel]
            e2_id_list = [entity2id[x] for x in tails]

            data_point = {
                'e1': e1_id,
                'rel': rel_id,
                'rel_eval': -1,
                'e2': -1,
                'e2_multi1': e2_id_list,
                'e2_multi2': []
            }
            f.write(json.dumps(data_point) + '\n')

    # 7) Also write entity2id.json, relation2id.json for reference
    with open(join(base_path, 'entity2id.json'), 'w') as f:
        json.dump(entity2id, f, indent=2)
    with open(join(base_path, 'relation2id.json'), 'w') as f:
        json.dump(relation2id, f, indent=2)

    # 8) Debugging prints
    print("\nDebugging output:")
    print(f"  Wrote training JSON --> {out_train}")
    print(f"  Wrote dev JSON      --> {out_dev}")
    print(f"  Wrote test JSON     --> {out_test}")
    print(f"  Wrote full JSON     --> {out_full}")
    print(f"  entity2id.json and relation2id.json are also saved.")

    # Show a sample line from each
    for path in [out_train, out_dev, out_test, out_full]:
        with open(path, 'r') as f:
            sample_line = next(f).strip()
        print(f"  Sample line from {os.path.basename(path)}: {sample_line}")


if __name__ == '__main__':
    """
    Example usage:
       python preprocessing.py FB15k-237
    """
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <dataset_name>")
        sys.exit(0)
    dataset_name = sys.argv[1]
    preprocess_data(dataset_name)