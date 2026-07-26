"""Dataset adapter for the multiplex DMGI trainer."""

import torch

from layers import AvgReadout
from utils import process


class embedder:
    """Load graph views and expose the legacy trainer attributes."""

    def __init__(self, args):
        args.batch_size = 1
        args.sparse = True
        args.metapaths_list = args.metapaths.split(",")
        args.device = torch.device(
            f"cuda:{args.gpu_num}" if args.gpu_num != "cpu" and torch.cuda.is_available() else "cpu"
        )
        graphs, features, labels, train, validation, test = process.loads(args)
        features = [process.preprocess_features(view) for view in features]
        args.nb_nodes, args.ft_size = features[0].shape
        args.nb_classes = labels.shape[1]
        args.nb_graphs = len(graphs)
        self.adj = [process.sparse_mx_to_torch_sparse_tensor(process.normalize_adj(graph)) for graph in graphs]
        self.features = [torch.as_tensor(view.toarray()).unsqueeze(0).float() for view in features]
        self.labels = torch.as_tensor(labels).unsqueeze(0).float().to(args.device)
        self.idx_train = torch.as_tensor(train, dtype=torch.long, device=args.device)
        self.idx_val = torch.as_tensor(validation, dtype=torch.long, device=args.device)
        self.idx_test = torch.as_tensor(test, dtype=torch.long, device=args.device)
        self.train_lbls = self.labels[0, self.idx_train].argmax(dim=1)
        self.val_lbls = self.labels[0, self.idx_val].argmax(dim=1)
        self.test_lbls = self.labels[0, self.idx_test].argmax(dim=1)
        args.readout_func = AvgReadout()
        args.readout_act_func = torch.nn.Sigmoid()
        self.args = args
