# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px
import json

import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import dgl
from dgl import save_graphs, load_graphs


# %% [markdown]
# # Load Data
import sys

# %%
data_path = sys.argv[
    1
]  # "/Users/jl102430/Documents/study/anomaly_detection/data/dynamic/DGraph/DGraphFin/dgraphfin.npz"

save_freq = int(sys.argv[2])

# %%
data = np.load(data_path)
data

# %%
X = data["x"]
y = data["y"]

edge_index = data["edge_index"]
edge_type = data["edge_type"]
edge_timestamp = data["edge_timestamp"]

train_mask = data["train_mask"]
valid_mask = data["valid_mask"]
test_mask = data["test_mask"]


print(
    f"""
X shape: {X.shape},
y shape: {y.shape}

edge_index shape: {edge_index.shape}
edge_type shape: {edge_type.shape}
edge_timestamp shape: {edge_timestamp.shape}

train_mask shape: {train_mask.shape}
valid_mask shape: {valid_mask.shape}
test_mask shape: {test_mask.shape}
"""
)

# %%
edge_index = pd.DataFrame(edge_index, columns=[f"src_id", "dst_id"])
edge_index["edge_type"] = edge_type
edge_index["edge_timestamp"] = edge_timestamp

edge_index = edge_index.sort_values("edge_timestamp")

edge_index


# %%
def resolve_node_type(df):  # update df in-place
    node_type_feat_idx = 0
    _type_map = {t: i for i, t in enumerate(df[f"feat_{node_type_feat_idx}"].unique())}
    df["node_type"] = df[f"feat_{node_type_feat_idx}"].apply(lambda x: _type_map[x])
    return _type_map


node_feature = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
node_feature["y"] = y
node_feature = node_feature.reset_index().rename(columns={"index": "node_id"})
node_type_map = resolve_node_type(node_feature)

node_feature

# %%
node_feature[node_feature["node_id"] == 3683606]

# %%
full_data = (
    edge_index.merge(
        node_feature[["node_id", "node_type"]], left_on=["src_id"], right_on=["node_id"]
    )
    .rename(columns={"node_type": "src_type"})
    .merge(
        node_feature[["node_id", "node_type"]], left_on=["dst_id"], right_on=["node_id"]
    )
    .rename(columns={"node_type": "dst_type"})
    .drop(["node_id_x", "node_id_y"], axis=1)
)

full_data

# %%
(
    full_data[full_data.index.isin(train_mask)]
    .groupby(["edge_timestamp", "src_type", "edge_type", "dst_type"])[
        ["src_id", "dst_id"]
    ]
    .count()
    # .agg({
    #     "src_id": lambda x: list(x),
    #     "dst_id": lambda x: list(x)
    # })
    .reset_index()
)

# %%
full_data

# %%
(
    full_data[full_data.index.isin(train_mask)]
    .groupby(
        [
            "edge_timestamp"
            # , "src_type", "edge_type", "dst_type"
        ]
    )[["src_id", "dst_id"]]
    .max()
    # .agg({
    #     "src_id": lambda x: list(x),
    #     "dst_id": lambda x: list(x)
    # })
    .reset_index()
)

# %%
tmp = (
    full_data[full_data.index.isin(train_mask)]
    .groupby("edge_timestamp")
    .agg({"src_id": lambda x: list(x), "dst_id": lambda x: list(x)})
    .reset_index()
)

tmp["node_list"] = tmp["src_id"] + tmp["dst_id"]
tmp["node_list"].apply(lambda x: len(set(x)))

# %%
tmp = (
    full_data.groupby(["edge_timestamp", "src_type", "edge_type", "dst_type"])[
        ["src_id", "dst_id"]
    ]
    .agg({"src_id": lambda x: list(x), "dst_id": lambda x: list(x)})
    .reset_index()
    .sort_values("edge_timestamp")
)


# %%
def resolve_node_list(df):
    _all_nodes = []
    cnt = 0
    for i in tqdm((df["src_id"] + df["dst_id"])):
        _all_nodes = _all_nodes + i
        if cnt % 10000 == 0:
            _all_nodes = list(set(_all_nodes))
        cnt += 1
    return set(_all_nodes)


def resolve_node_mapping_by_types(_node_feature):
    _mapping = {}
    for ntype in _node_feature["node_type"].unique():
        index2node = (
            _node_feature[_node_feature.node_type == ntype]
            .reset_index(drop=True)["node_id"]
            .to_dict()
        )
        node2index = {v: k for k, v in index2node.items()}
        _mapping[ntype] = node2index
    return _mapping


def apply_node_reindex_by_map(_map):
    def apply_reindex(_type, _node_id):
        return _map[_type][_node_id]

    return apply_reindex


def create_dgl_graph(graph_data_dict, num_nodes_dict, node_features):
    g = dgl.heterograph(graph_data_dict, num_nodes_dict=num_nodes_dict)

    for ntype in node_features["node_type"].unique():
        g.nodes[f"vtype_{ntype}"].data["features"] = torch.tensor(
            node_features[node_features["node_type"] == ntype][
                [f"feat_{i}" for i in range(1, 17)]
            ].values
        )
    return g


def resolve_lables_by_types(node_features):
    node_labels = {}
    for ntype in node_features["node_type"].unique():
        node_labels[f"vtype_{ntype}"] = torch.tensor(
            node_features[node_features["node_type"] == ntype]["y"].values
        )
    return node_labels


# %%
def construct_dgl_dataset(mask, name, save=False, save_interval=10):
    """
    Train/Val/Test needs to be re-indexed
    """
    if mask is not None:
        _data = full_data[full_data.index.isin(mask)]
    else:
        _data = full_data

    _tmp_data = (
        _data.groupby(["edge_timestamp"])[["src_id", "dst_id"]]
        .agg({"src_id": lambda x: list(x), "dst_id": lambda x: list(x)})
        .reset_index()
        .sort_values("edge_timestamp")
    )

    print("Reindex all the nodes..")
    node_list = resolve_node_list(_tmp_data)
    _node_feature = node_feature[node_feature["node_id"].isin(node_list)]

    node2id = resolve_node_mapping_by_types(_node_feature)
    node2id_apply = apply_node_reindex_by_map(node2id)

    # node_idx_map = {nid: i for i, nid in enumerate(node_list)}+

    _data["src_id"] = _data.apply(
        lambda x: node2id_apply(x["src_type"], x["src_id"]), axis=1
    )
    _data["dst_id"] = _data.apply(
        lambda x: node2id_apply(x["dst_type"], x["dst_id"]), axis=1
    )
    _node_feature["node_id"] = _node_feature.apply(
        lambda x: node2id_apply(x["node_type"], x["node_id"]), axis=1
    )

    # resolve labels
    node_labels = resolve_lables_by_types(_node_feature)

    print("Agg..")
    graph_data = (
        _data.groupby(["edge_timestamp", "src_type", "edge_type", "dst_type"])[
            ["src_id", "dst_id"]
        ]
        .agg({"src_id": lambda x: list(x), "dst_id": lambda x: list(x)})
        .reset_index()
        .sort_values("edge_timestamp")
    )
    num_nodes_dict = {}
    for _i, (_t, _n) in (
        _node_feature.groupby("node_type")[["node_id"]].count().reset_index().iterrows()
    ):
        num_nodes_dict[f"vtype_{_t}"] = _n

    output_prefix = "../dataset/dgl_format_1"

    print(f"num_nodes_dict: {num_nodes_dict}")

    g_list = []
    graph_data_dict = {}
    current_ts = -1
    for idx, (
        edge_timestamp,
        src_type,
        edge_type,
        dst_type,
        src_list,
        dst_list,
    ) in tqdm(graph_data.iterrows()):
        # Start a new graph construction
        if (edge_timestamp > current_ts) and (current_ts != -1):
            g = create_dgl_graph(
                graph_data_dict,
                num_nodes_dict=num_nodes_dict,
                node_features=_node_feature,
            )
            g_list.append(g)

            # save the graph list
            if save:
                if edge_timestamp % save_interval == 0:
                    save_by_parts(g_list, None, edge_timestamp, name, output_prefix)
                    g_list = []

            graph_data_dict = {}

        graph_data_dict[
            (f"vtype_{src_type}", f"etype_{edge_type}", f"vtype_{dst_type}")
        ] = (
            torch.tensor(src_list),
            torch.tensor(dst_list),
        )
        current_ts = edge_timestamp

    if len(graph_data_dict.keys()) > 0:
        g = create_dgl_graph(
            graph_data_dict, num_nodes_dict=num_nodes_dict, node_features=_node_feature
        )
        g_list.append(g)

        if save:
            save_by_parts(g_list, None, edge_timestamp, name, output_prefix)

    torch.save(node_labels, f"{output_prefix}/dgraph_{name}_dgl_node_labels.pt")
    with open(f"{output_prefix}/dgraph_{name}_dgl_num_nodes_dict.json", "w") as fout:
        fout.write(json.dumps(num_nodes_dict))
        fout.write("\n")
    return g_list, _data, _node_feature, node_labels


def save_by_parts(g_list, node_labels, edge_timestamp, name, output_prefix):
    part_num = str(edge_timestamp).zfill(3)
    save_graphs(
        f"{output_prefix}/dgraph_{name}_dgl.bin.{part_num}", g_list, node_labels
    )
    g_list = []
    print(f"Save to {output_prefix}")


# %%

train_graphs, train_data, train_feature, train_node_labels = construct_dgl_dataset(
    train_mask, name="train", save=True, save_interval=save_freq
)

valid_graphs, valid_data, valid_feature, valid_node_labels = construct_dgl_dataset(
    valid_mask, name="valid", save=True, save_interval=save_freq
)

test_graphs, test_data, test_feature, test_node_labels = construct_dgl_dataset(
    test_mask, name="test", save=True, save_interval=save_freq
)
# len(train_graphs)
# len(train_graphs)

# len(train_graphs)
# python prepare_dataset_temporal.py ../dataset/raw/dgraphfin.npz 10
# %%


# %%
