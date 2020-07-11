from __future__ import print_function
import torch.nn as nn
import torch
from fairseq.models.roberta import RobertaModel

class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, pad_idx, nhid,
                 nentries=0, seqlen=0, lstmenc=False, pretrain=False, pretrain_dim=0):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.pretrain = pretrain
        self.pretrain_dim = pretrain_dim
        if pretrain:
            self.entry_encoder = nn.Linear(pretrain_dim, embedding_dim)
        if lstmenc:
            self.entry_encoder = nn.LSTM(embedding_dim, embedding_dim, 1, dropout=dropout, batch_first=True)
            self.key_encoder = nn.LSTM(embedding_dim, embedding_dim, 1, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # self.Qproj = nn.Linear(nhid, embedding_dim)
        self.LW_H = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=1)
        self.pad_idx = pad_idx
        self.lstmenc = lstmenc
        self.init_weights()

    def get_state(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(bsz, self.embedding_dim)

    def init_weights(self):
        initrange = 0.1
        # self.Qproj.bias.data.zero_()
        # self.Qproj.weight.data.uniform_(-initrange, initrange)
        self.LW_H.bias.data.zero_()
        self.LW_H.weight.data.uniform_(-initrange, initrange)
        if self.pretrain:
            self.entry_encoder.bias.data.zero_()
            self.entry_encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, entries, query, wordemb, true_ents=None):
        # b: batch size
        # m: number of entries
        # e: embedding size
        # s: sequence length
        mask = (1.0 - (entries == self.pad_idx).float()).unsqueeze(2).to(torch.float32)
        # query = self.Qproj(query) # b * e
        entry_size = entries.size() # m * s
        us = [query] # b * e
        prob_list = []
        # u = [self.get_state(story.size(0))]
        # entries = entries.contiguous().view(entries.size(0), -1).long()
        for hop in range(self.max_hops):
            # embed_A = self.C[hop](entries) * mask # m * s * e
            # embed_A = self.KBembA(entries) * mask
            if self.pretrain:
                m_A = self.dropout(self.entry_encoder(torch.sum(entries, 1)))
            else:
                embed_A = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
                if self.lstmenc:
                    embed_A, _ = self.key_encoder(embed_A)
                    m_A = embed_A[:,-1,:]
                else:
                    m_A = torch.sum(embed_A, 1) # m * e

            # b * m * e
            # Dot product attention: (b * e) x (m * e) -> (b * m)
            logits = torch.einsum('ik,jk->ij', us[-1], m_A)
            prob = self.softmax(logits) # b * m
            prob_list.append(logits)

            if self.pretrain:
                m_C = self.dropout(self.entry_encoder(torch.sum(entries, 1)))
            else:
                embed_C = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
                if self.lstmenc:
                    embed_C, _ = self.key_encoder(embed_C)
                    m_C = embed_C[:,-1,:]
                else:
                    m_C = torch.sum(embed_C, 1)

            if true_ents is not None:
                o_k = torch.index_select(m_C, 0, true_ents)
            else:
                # (b * m) x (m * e) -> (b * e)
                o_k = torch.einsum('bm,me->be', prob, m_C)
            # u_k = nn.functional.relu(self.LW_H(us[-1]))
            # u_k = u_k + o_k
            u_k = self.LW_H(us[-1]) + o_k
            us.append(u_k)
        return us[-1], prob_list[-1]

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
