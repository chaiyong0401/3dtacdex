from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from robomimic.models.base_nets import MAEGAT
from robomimic.models.utils import data_to_gnn_batch, create_activation

from .loss_func import sce_loss
from pretraining.pretrain_utils import create_norm
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import add_self_loops, remove_self_loops


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = MAEGAT(
            input_channel=in_dim,
            num_hidden=num_hidden,
            output_channel=out_dim,
            # out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            mask_index=6,
            resultant_type=None,
            num_nodes=None,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.mask_index = mask_index
        self.resultant_type = resultant_type
        self.num_nodes = num_nodes
        print('==============')
        print(self.mask_index)
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        if self.resultant_type is not None:
            if self.resultant_type == 'force':
                self.resultant_predictor = nn.Sequential(
                    nn.Linear(dec_in_dim*self.num_nodes, int(dec_in_dim*self.num_nodes/16)),
                    create_activation(activation),
                    nn.Linear(int(dec_in_dim*self.num_nodes/16), 3),
                )
        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes, self.mask_index:] = -1.0
            out_x[noise_nodes, self.mask_index:] = x[noise_to_be_chosen, self.mask_index:]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes, self.mask_index:] = -1.0

        return out_x, (mask_nodes, keep_nodes)

    def forward(self, batch, return_predict=False, eval=False):
        # ---- attribute reconstruction ----
        x, edge_index = batch['tactile_data'].x, batch['tactile_data'].edge_index
        if return_predict:
            assert eval == True, "return_predict should be used in eval mode due to clone of input"
            if self.resultant_type is not None:
                if self.resultant_type == 'force':
                    if self._mask_rate > 0:
                        x, recon, mask_nodes, keep_nodes = self.mask_attr_prediction(x, edge_index, True)
                    else:
                        x, recon, mask_nodes, keep_nodes = x, None, None, None
                    predicted_resultant_force = self.resultant_prediction(x.clone(), edge_index, recon, mask_nodes, batch['resultant_data'].shape[0])
                    return x, recon, mask_nodes, keep_nodes, predicted_resultant_force
            else:
                x, recon, mask_nodes, keep_nodes = self.mask_attr_prediction(x, edge_index, True)
                return x, recon, mask_nodes, keep_nodes, None
        elif not return_predict:
            if self.resultant_type is not None:
                if self.resultant_type == 'force':
                    if self._mask_rate > 0:
                        # local force prediction
                        x, recon, mask_nodes, keep_nodes = self.mask_attr_prediction(x, edge_index, True)
                        recon_loss = self.compute_recon_loss(x, recon, mask_nodes)
                        loss_item = {"recon_loss": recon_loss.item()}
                        # net force prediction
                        predicted_resultant_force = self.resultant_prediction(x, edge_index, recon, mask_nodes, batch['resultant_data'].shape[0])
                        predict_loss = self.compute_resultant_loss(batch['resultant_data'], predicted_resultant_force)
                        loss_item["predict_loss"] = predict_loss.item()
                        # combine losses
                        loss = recon_loss + predict_loss
                    else:
                        x, recon, mask_nodes, keep_nodes = x, None, None, None
                        loss_item = {"recon_loss": 0}
                        predicted_resultant_force = self.resultant_prediction(x, edge_index, recon, mask_nodes, batch['resultant_data'].shape[0])
                        predict_loss = self.compute_resultant_loss(batch['resultant_data'], predicted_resultant_force)
                        loss_item["predict_loss"] = predict_loss.item()
                        loss = predict_loss
                    
            else:
                loss = self.mask_attr_prediction(x, edge_index, return_predict)
                loss_item = {"recon_loss": loss.item()}
                loss_item["predict_loss"] = 0
            return loss, loss_item
    
    def resultant_prediction(self, x, edge_index, recon, mask_nodes, batch_size):
        if mask_nodes is not None:
            # replace mask_nodes in x using recon values
            x[mask_nodes, self.mask_index:] = recon[mask_nodes, self.mask_index:]

        # encode x
        enc_rep = self.encoder(x, edge_index)
        
        # reshape to each batch
        enc_rep = enc_rep.view(batch_size, -1)

        # predict resultant force
        predicted_resultant_force = self.resultant_predictor(enc_rep)
        return predicted_resultant_force

    def mask_attr_prediction(self, x, edge_index, return_predict=False):
        # mask the force
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self._mask_rate)

        # encode the input
        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep = self.encoder(use_x, use_edge_index)

        assert not self._concat_hidden

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, use_edge_index)

        if return_predict:
            return x, recon, mask_nodes, keep_nodes
        elif not return_predict:
            loss = self.compute_recon_loss(x, recon, mask_nodes)
            return loss

    def compute_recon_loss(self, x, recon, mask_nodes):
        x_init = x[mask_nodes, self.mask_index:]
        x_rec = recon[mask_nodes, self.mask_index:]

        loss = self.criterion(x_rec, x_init)
        return loss
    
    def compute_resultant_loss(self, ori, predict):
        loss = self.criterion(predict, ori)
        return loss
    
    def embed(self, x, edge_index):
        rep = self.encoder(x, edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
