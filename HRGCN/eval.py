import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from data_loader import HetGCNEventGraphDataset


def evaluate_node(model, data_root_dir, ignore_weight, include_edge_type, device):
    valid_dataset = HetGCNEventGraphDataset(
        node_feature_csv=f"{data_root_dir}/valid_node_feature_norm.csv",
        edge_index_csv=f"{data_root_dir}/valid_edge_index.csv",
        node_type_txt=f"{data_root_dir}/valid_node_types.txt",
        edge_ratio_csv=None,
        ignore_weight=ignore_weight,
        include_edge_type=include_edge_type,
        edge_ratio_percentile=None,
        # n_known_abnormal=n_known_abnormal,
        # trace_info_csv=f"{self.data_root_dir}/trace_info.csv",
    )

    dataset_size = valid_dataset.size()

    model.eval()
    # set validation dataset
    model.dataset = valid_dataset
    for i in range(dataset_size):
        # 1. get node embeddings
        _, _, _node_embed = model([i], train=False, return_node_embed=True)

        _node_labels = (
            torch.from_numpy(
                valid_dataset.node_feature_df[
                    valid_dataset.node_feature_df.trace_id == i
                ]["y"].values
            )
            .float()
            .to(device)
        )
        # 2. filter on0/1 nodes
        _mask = (_node_labels == 0) or (_node_labels == 1)
        node_labels = _node_labels[_mask]
        node_embed = _node_embed[0][_mask]

        # Calc svdd score node level
        svdd_score = torch.mean(torch.square(node_embed - model.svdd_center), 1)
        # 3. Evaluate on AUC and AP between these

        fpr, tpr, roc_thresholds = roc_curve(node_labels, svdd_score)
        precision, recall, pr_thresholds = precision_recall_curve(
            node_labels, svdd_score
        )
        roc_auc = auc(fpr, tpr)
        ap = auc(recall, precision)

        return roc_auc, ap, -1, -1
