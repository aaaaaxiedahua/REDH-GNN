import torch
import torch.nn as nn
from torch_scatter import scatter


class DiagonalSheafLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(DiagonalSheafLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.s_src_r = nn.Parameter(torch.empty(in_dim))
        self.s_src_q = nn.Parameter(torch.empty(in_dim))
        self.s_src_rq = nn.Parameter(torch.empty(in_dim))
        self.s_src_b = nn.Parameter(torch.empty(in_dim))
        self.s_tgt_r = nn.Parameter(torch.empty(in_dim))
        self.s_tgt_q = nn.Parameter(torch.empty(in_dim))
        self.s_tgt_rq = nn.Parameter(torch.empty(in_dim))
        self.s_tgt_b = nn.Parameter(torch.empty(in_dim))

        self.W_node = nn.Linear(in_dim, in_dim, bias=False)
        self.W_query_node = nn.Linear(in_dim, in_dim, bias=False)
        self.w_node = nn.Linear(in_dim, 1)

        self.W_force = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for param in [
            self.s_src_r,
            self.s_src_q,
            self.s_src_rq,
            self.s_src_b,
            self.s_tgt_r,
            self.s_tgt_q,
            self.s_tgt_rq,
            self.s_tgt_b,
        ]:
            nn.init.normal_(param, mean=0.0, std=0.02)

    def _sheaf_scales(self, rel, q_rel, edge_batch_idx):
        hr = self.rela_embed(rel)
        h_qr = self.rela_embed(q_rel)[edge_batch_idx]
        src = (
            hr * self.s_src_r
            + h_qr * self.s_src_q
            + (hr * h_qr) * self.s_src_rq
            + self.s_src_b
        )
        tgt = (
            hr * self.s_tgt_r
            + h_qr * self.s_tgt_q
            + (hr * h_qr) * self.s_tgt_rq
            + self.s_tgt_b
        )
        return torch.nn.functional.softplus(src), torch.nn.functional.softplus(tgt), hr, h_qr

    def _filter_topk_nodes(self, nodes, edges, node_score, topk_nodes):
        if topk_nodes is None or topk_nodes <= 0 or nodes.size(0) == 0:
            return nodes, edges, self._old_nodes_new_idx(edges)

        obj = edges[:, 5]
        identity_mask = edges[:, 2] == (2 * self.n_rel)
        selected = torch.zeros(nodes.size(0), dtype=torch.bool, device=nodes.device)
        selected[obj[identity_mask]] = True

        batch_ids = torch.unique(nodes[:, 0], sorted=True)
        for batch_id in batch_ids:
            batch_mask = nodes[:, 0] == batch_id
            candidate_idx = torch.nonzero(batch_mask & (~selected), as_tuple=False).view(-1)
            if candidate_idx.numel() == 0:
                continue
            if candidate_idx.numel() <= topk_nodes:
                selected[candidate_idx] = True
                continue
            _, top_pos = torch.topk(node_score[candidate_idx], k=topk_nodes)
            selected[candidate_idx[top_pos]] = True

        edge_mask = selected[obj]
        filtered_edges = edges[edge_mask].clone()
        selected_idx = torch.nonzero(selected, as_tuple=False).view(-1)
        filtered_nodes = nodes[selected_idx]

        remap = torch.empty(nodes.size(0), dtype=torch.long, device=nodes.device)
        remap[selected_idx] = torch.arange(selected_idx.numel(), device=nodes.device)
        filtered_edges[:, 5] = remap[filtered_edges[:, 5]]
        old_nodes_new_idx = self._old_nodes_new_idx(filtered_edges)
        return filtered_nodes, filtered_edges, old_nodes_new_idx

    def _old_nodes_new_idx(self, edges):
        identity_mask = edges[:, 2] == (2 * self.n_rel)
        identity_edges = edges[identity_mask]
        if identity_edges.size(0) == 0:
            return torch.LongTensor([]).cuda()
        _, old_idx = identity_edges[:, 4].sort()
        return identity_edges[:, 5][old_idx]

    def forward(self, q_rel, hidden, nodes, edges, topk_nodes=0, eps=1e-6):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        edge_batch_idx = edges[:, 0]

        hs = hidden[sub]
        s_src, s_tgt, hr, h_qr = self._sheaf_scales(rel, q_rel, edge_batch_idx)

        x_src = hs + hr
        alpha = torch.sigmoid(
            self.w_alpha(
                nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
            )
        )
        transported = (s_src / (s_tgt + eps)) * x_src
        messages = alpha * transported

        preview = scatter(messages, index=obj, dim=0, dim_size=nodes.size(0), reduce="sum")
        node_query = self.rela_embed(q_rel)[nodes[:, 0]]
        node_score = self.w_node(
            torch.tanh(self.W_node(preview) + self.W_query_node(node_query))
        ).squeeze(-1)

        nodes, edges, old_nodes_new_idx = self._filter_topk_nodes(
            nodes, edges, node_score, topk_nodes
        )

        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        edge_batch_idx = edges[:, 0]

        hs = hidden[sub]
        s_src, s_tgt, hr, h_qr = self._sheaf_scales(rel, q_rel, edge_batch_idx)
        x_src = hs + hr
        alpha = torch.sigmoid(
            self.w_alpha(
                nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
            )
        )
        transported = (s_src / (s_tgt + eps)) * x_src
        messages = alpha * transported
        message_agg = scatter(messages, index=obj, dim=0, dim_size=nodes.size(0), reduce="sum")
        force = self.act(self.W_force(message_agg))

        target_proj = s_tgt * force[obj]
        source_proj = s_src * x_src
        sheaf_loss = (alpha.squeeze(-1) * (source_proj - target_proj).pow(2).sum(dim=1)).mean()
        return nodes, edges, old_nodes_new_idx, force, sheaf_loss


class SheafMomentumREDGNN(torch.nn.Module):
    def __init__(self, params, loader):
        super(SheafMomentumREDGNN, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        self.topk_nodes = getattr(params, "topk_nodes", 0)
        self.gamma = getattr(params, "gamma", 0.7)
        self.beta = getattr(params, "beta", 1.0)
        self.lambda_sheaf = getattr(params, "lambda_sheaf", 0.0)
        self.lambda_dyn = getattr(params, "lambda_dyn", 0.0)
        self.extra_loss = torch.tensor(0.0).cuda()

        acts = {"relu": nn.ReLU(), "tanh": torch.tanh, "idd": lambda x: x}
        act = acts[params.act]

        self.gnn_layers = nn.ModuleList(
            [
                DiagonalSheafLayer(
                    self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act
                )
                for _ in range(self.n_layer)
            ]
        )

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim * 2, 1, bias=False)

    def forward(self, subs, rels, mode="train"):
        n = len(subs)
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()
        momentum = torch.zeros_like(hidden)

        sheaf_losses = []
        dyn_losses = []

        for i in range(self.n_layer):
            nodes, edges, _ = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            nodes, edges, old_nodes_new_idx, force, sheaf_loss = self.gnn_layers[i](
                q_rel, hidden, nodes, edges, topk_nodes=self.topk_nodes
            )

            hidden_old = torch.zeros(nodes.size(0), hidden.size(1)).cuda()
            hidden_old.index_copy_(0, old_nodes_new_idx, hidden)
            momentum_old = torch.zeros(nodes.size(0), momentum.size(1)).cuda()
            momentum_old.index_copy_(0, old_nodes_new_idx, momentum)

            force = self.dropout(force)
            momentum = self.gamma * momentum_old + (1.0 - self.gamma) * force
            hidden = hidden_old + self.beta * momentum

            sheaf_losses.append(sheaf_loss)
            dyn_losses.append((momentum - momentum_old).pow(2).sum(dim=1).mean())

        readout = torch.cat([hidden, momentum], dim=1)
        scores = self.W_final(readout).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores

        extra_loss = torch.tensor(0.0, device=scores_all.device)
        if sheaf_losses and self.lambda_sheaf > 0:
            extra_loss = extra_loss + self.lambda_sheaf * torch.stack(sheaf_losses).mean()
        if dyn_losses and self.lambda_dyn > 0:
            extra_loss = extra_loss + self.lambda_dyn * torch.stack(dyn_losses).mean()
        self.extra_loss = extra_loss
        return scores_all
