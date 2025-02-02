#!/usr/bin/env python
import json
import os
from os.path import join
from collections import defaultdict

def preprocess_data(dataset_name):
    base_path = f'/Users/andrasjoos/Documents/AI_masters/Period_9/ML4G/Project/LinkPred/data/{dataset_name}'
    train_file = join(base_path, 'train.txt')
    valid_file = join(base_path, 'valid.txt')
    test_file  = join(base_path, 'test.txt')

    files = [train_file, valid_file, test_file]
    names = ['train.txt', 'valid.txt', 'test.txt']
    data = {}
    for name, fpath in zip(names, files):
        with open(fpath, 'r') as f:
            lines = [line.strip().split('\t') for line in f]
        data[name] = lines

    print("Debug: Number of lines in each split:")
    for split_name in data:
        print(f"  {split_name}: {len(data[split_name])} triples")

    all_entities = set()
    all_relations = set()

    label_graph = defaultdict(set)   
    train_graph = {n: defaultdict(set) for n in names}
    test_cases  = {n: [] for n in names}

    def add_edge(e1, rel, e2, split_name):
        rrev = rel + '_reverse'
        label_graph[(e1, rel)].add(e2)
        label_graph[(e2, rrev)].add(e1)

        train_graph[split_name][(e1, rel)].add(e2)
        train_graph[split_name][(e2, rrev)].add(e1)

        test_cases[split_name].append((e1, rel, e2))

        all_entities.add(e1)
        all_entities.add(e2)
        all_relations.add(rel)
        all_relations.add(rrev)

    for name in names:
        for (e1, rel, e2) in data[name]:
            add_edge(e1, rel, e2, name)

    all_entities = sorted(list(all_entities))
    all_relations = sorted(list(all_relations))

    entity2id = {ent: idx for idx, ent in enumerate(all_entities)}
    relation2id = {rel: idx for idx, rel in enumerate(all_relations)}

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
                    'head': e1_id,
                    'relation': rel_id,
                    'reverse_relation': -1,  # -1 for training data
                    'tail': -1,  # -1 for training data
                    'valid_tails': e2_id_list,
                    'valid_heads': []  # empty for training data
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
                    'head': e1_id,
                    'relation': rel_id,
                    'reverse_relation': rrev_id,
                    'tail': e2_id,
                    'valid_tails': e2_id_list_1,
                    'valid_heads': e2_id_list_2
                }
                f.write(json.dumps(data_point) + '\n')

    out_train = join(base_path, 'train.json')
    out_dev   = join(base_path, 'valid.json')
    out_test  = join(base_path, 'test.json')
    out_full  = join(base_path, 'full.json')

    write_training_graph(train_graph['train.txt'], out_train)

    write_evaluation_graph(test_cases['valid.txt'], label_graph, out_dev)
    write_evaluation_graph(test_cases['test.txt'],  label_graph, out_test)

    with open(out_full, 'w') as f:
        for (e1, rel), tails in label_graph.items():
            e1_id  = entity2id[e1]
            rel_id = relation2id[rel]
            e2_id_list = [entity2id[x] for x in tails]

            data_point = {
                'head': e1_id,
                'relation': rel_id,
                'reverse_relation': -1,
                'tail': -1,
                'valid_tails': e2_id_list,
                'valid_heads': []
            }
            f.write(json.dumps(data_point) + '\n')

    with open(join(base_path, 'entity.json'), 'w') as f:
        json.dump(entity2id, f, indent=2)
    with open(join(base_path, 'relation.json'), 'w') as f:
        json.dump(relation2id, f, indent=2)

    print("\nDebugging output:")
    print(f"  Wrote training JSON --> {out_train}")
    print(f"  Wrote dev JSON      --> {out_dev}")
    print(f"  Wrote test JSON     --> {out_test}")
    print(f"  Wrote full JSON     --> {out_full}")
    print(f"  entity2id.json and relation2id.json are also saved.")

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