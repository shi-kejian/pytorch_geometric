import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import LayerNorm,BatchNorm1d, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from tqdm.notebook import tqdm
import logging

print("Loading OGB modules...")
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
print("OGB loaded")

# Load the dataset 
dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='../data')
split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=2)

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, graph_pool='mean', conv_enc_edge=True):
        super().__init__()
        self.conv_enc_edge = conv_enc_edge
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=7, norm='batch')
            norm = BatchNorm1d(hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.2)
            self.layers.append(layer)

        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        self.bond_encoder = BondEncoder(emb_dim=hidden_channels) # unused when default conv_enc_edge=True

        self.pool = global_mean_pool if graph_pool == "mean" \
                                     else global_add_pool if graph_pool == "sum" \
                                     else global_max_pool
    
        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, batch_data):

        x, edge_index, edge_attr, batch = \
                    batched_data.x, batched_data.edge_index, batch_data.edge_attr, batched_data.batch
        h = self.atom_encoder(x)
        edge_attr = edge_attr if self.conv_enc_edge else self.bond_encoder(edge_attr)

        h = self.layers[0].conv(h, edge_index, edge_attr)
        for layer in self.layers[1:]:
            h = layer(h, edge_index, edge_attr)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h, p=0.2, training=self.training)

        h_graph = self.pool(h,batch)
        return self.lin(h_graph)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=256, num_layers=7).to(device)
evaluator = Evaluator(name='ogbg-molhiv')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()   

def train(model, device, loader, optimizer, criterion):
    loss_lst = []
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)
            is_labeled = batch.y == batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            loss_lst.append(loss.item())
    return np.array(loss_lst).mean()

@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}
    return evaluator.eval(input_dict)


results = {'highest_valid': 0,
            'final_train': 0,
            'final_test': 0,
            'highest_train': 0}

for epoch in range(1, 301):
    print(epoch)
    logging.info("=====Epoch {}".format(epoch))
    logging.info('Training...')
    epoch_loss = train(model, device, train_loader, optimizer, criterion)
    logging.info('Evaluating...')
    train_rocauc = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
    valid_rocauc = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
    test_rocauc = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

    logging.info({'Train': train_rocauc,
                    'Validation': valid_rocauc,
                    'Test': test_rocauc})

    # if train_rocauc > results['highest_train']:
    #     results['highest_train'] = train_rocauc
    # if valid_rocauc > results['highest_valid']:
    #     results['highest_valid'] = valid_rocauc
    #     results['final_train'] = train_rocauc
    #     results['final_test'] = test_rocauc

    #     save_ckpt(model, optimizer,
    #                 round(epoch_loss, 4), epoch,
    #                 args.model_save_path,
    #                 sub_dir, name_post='valid_best')

logging.info("%s" % results)

