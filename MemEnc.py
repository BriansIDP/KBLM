from __future__ import print_function
import torch.nn as nn
import torch
import math

class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, pad_idx, nhid,
                 lstmenc=False, pretrain=False, pretrain_dim=0):
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
        self.Qgate = nn.Linear(nhid, embedding_dim)
        self.softmax = nn.Softmax(dim=1)
        self.pad_idx = pad_idx
        self.lstmenc = lstmenc
        self.init_weights()

    def get_state(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(bsz, self.embedding_dim)

    def init_weights(self):
        initrange = 0.1
        self.Qgate.bias.data.zero_()
        self.Qgate.weight.data.uniform_(-initrange, initrange)
        if self.pretrain:
            self.entry_encoder.bias.data.zero_()
            self.entry_encoder.weight.data.uniform_(-initrange, initrange)

    def value_atten(self, query, entries, mask):
        # (bsize * nemb), (nentries * entry_size * nemb) -> (bsize * nentries * entry_size)
        logits = torch.einsum('ik,mnk->imn', query, entries)
        mask = mask.squeeze(-1).unsqueeze(0).repeat(query.size(0), 1, 1) == 0
        logits.masked_fill_(mask, -1e9)
        prob = nn.functional.softmax(logits, dim=-1)
        # (bsize * nentries * entry_size), (nentries * entry_size * nemb) -> (bsize, nentries, nemb)
        output = torch.einsum('imn,mnk->imk', prob, entries)
        return output

    def single_step_atten(self, query, entries, wordemb, mask, true_ents=None, forcing_p=0.0):
        if self.pretrain:
            m_A = self.dropout(self.entry_encoder(torch.sum(entries, 1)))
        else:
            embed_A = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
            embed_A = embed_A * mask
            if self.lstmenc:
                embed_A, _ = self.key_encoder(embed_A)
                embed_A = self.dropout(embed_A)
                indices = nn.functional.relu(mask.sum(dim=1).to(torch.long) - 1)
                dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), embed_A.size(2))
                m_A = torch.gather(embed_A, 1, dummy).squeeze(1)
                # m_A = embed_A[:,-1,:]
            else:
                m_A = torch.sum(embed_A, 1) # m * e

            # Dot product attention: (b * e) x (m * e) -> (b * m)
            logits = torch.einsum('ik,jk->ij', query, m_A)
            # Scale by sqrt(e)
            logits /= torch.sqrt(torch.tensor(m_A.size(1), dtype=m_A.dtype, device=m_A.device))
            prob = self.softmax(logits) # b * m

        if self.pretrain:
            m_C = self.dropout(self.entry_encoder(torch.sum(entries, 1)))
        else:
            embed_C = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
            embed_C = embed_C * mask
            if self.lstmenc:
                embed_C, _ = self.entry_encoder(embed_C)
                m_C = torch.gather(embed_C, 1, dummy).squeeze(1)
            else:
                m_C = torch.sum(embed_C, 1)
        if true_ents is not None:
            o_k_pred = torch.einsum('bm,me->be', prob, m_C)
            o_k_true = torch.index_select(m_C, 0, true_ents.view(-1))
            force_mask = (torch.rand(o_k_true.size(0), 1) < forcing_p).to(o_k_true.dtype).to(o_k_true.device)
            o_k = o_k_true * force_mask + o_k_pred * (1 - force_mask)
        else:
            o_k = torch.einsum('bm,me->be', prob, m_C)

        return o_k, logits

    def double_step_atten(self, query, entries, wordemb, mask, true_ents=None):
        # get word embeddings
        embed_A_1 = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
        # embed_A_1 = embed_A_1 * mask
        # first attention to get entry embeddings
        m_A = self.value_atten(query, embed_A_1, mask)
        # second attention to get the KB embedding
        # Dot product attention: (b * e) x (b * m * e) -> (b * m)
        logits = torch.einsum('ik,ijk->ij', query, m_A)
        # Scale by sqrt(e)
        logits /= torch.sqrt(torch.tensor(m_A.size(1), dtype=m_A.dtype, device=m_A.device))
        prob = self.softmax(logits) # b * m
        if true_ents is not None:
            indices = true_ents.view(-1, 1)
            dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), m_A.size(2))
            o_k = torch.gather(m_A, 1, dummy).squeeze(1)
        else:
            o_k = torch.einsum('bm,bme->be', prob, m_A)
        return o_k, logits

    def hierarchical_atten(self, query, entries, wordemb, levels):
        # reshape and pad KB entries
        total_entries = entries.size(0)
        n_entries = math.ceil(total_entries ** (1/levels))
        to_pad = n_entries ** levels - total_entries
        padding = torch.ones(int(to_pad), entries.size(1), dtype=entries.dtype, device=entries.device) * self.pad_idx
        entries = torch.cat([entries, padding], dim=0).view([n_entries] * levels + [-1])
        mask = (entries == self.pad_idx).squeeze(-1)

        embed_A = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
        embed_C = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
        if self.lstmenc:
            embed_A, _ = self.key_encoder(embed_A)
            m_A = embed_A[:,-1,:]
            embed_C, _ = self.key_encoder(embed_C)
            m_C = embed_C[:,-1,:]
        else:
            m_A = torch.sum(embed_A, -2) # m^levels * e
            m_C = torch.sum(embed_C, -2)
        problist = []
        for i in range(levels):
            tensor_size = 'mnacdfgh'[:levels-i]
            if i > 0:
                batched_tensor_size = 'i' + tensor_size
            else:
                batched_tensor_size = tensor_size
            # Dot product attention: (b * e) x (m^level * e) -> (b * m^level)
            logits = torch.einsum('ik,{}k->i{}'.format(batched_tensor_size, tensor_size), query, m_A)
            logits.masked_fill_(mask, -1e9)
            prob = torch.nn.functional.softmax(logits, dim=-1) # b * m^level
            problist.append(prob)
            m_A = torch.einsum('i{},{}e->i{}e'.format(tensor_size, batched_tensor_size, tensor_size[:-1]), prob, m_A)
            m_C = torch.einsum('i{},{}e->i{}e'.format(tensor_size, batched_tensor_size, tensor_size[:-1]), prob, m_C)
            mask = torch.min(mask, dim=-1)[0]
        if levels == 2:
            final_prob = problist[0] * problist[1].unsqueeze(-1)
            final_prob = final_prob.view(final_prob.size(0), -1)[:, :total_entries]
        return m_C, final_prob

    def forward(self, entries, query, wordemb, true_ents=None, hierarchical=1, forcing_p=0.0):
        # b: batch size
        # m: number of entries
        # e: embedding size
        # s: sequence length
        mask = (1.0 - (entries == self.pad_idx).float()).unsqueeze(2).to(torch.float32)
        # query = self.Qproj(query) # b * e
        entry_size = entries.size() # m * s
        # query = nn.functional.sigmoid(self.Qgate(query))
        query = self.Qgate(query)
        us = [query] # b * e
        prob_list = []
        # u = [self.get_state(story.size(0))]
        # entries = entries.contiguous().view(entries.size(0), -1).long()
        for hop in range(self.max_hops):
            if hierarchical == 1:
                o_k, logits = self.single_step_atten(us[-1], entries, wordemb, mask, true_ents, forcing_p)
            else:
                o_k, logits = self.double_step_atten(us[-1], entries, wordemb, mask, true_ents, forcing_p)
                # o_k, logits = self.hierarchical_atten(us[-1], entries, wordemb, hierarchical)
            prob_list.append(logits)

            # gating = self.Qgate(torch.cat(us[-1], dim=-1))
            u_k = o_k
            # u_k = self.LW_H(us[-1]) + o_k
            # u_k = o_k * nn.functional.sigmoid(gating)
            us.append(u_k)
        return us[-1], prob_list[-1]


class EncoderTreeMemNN(nn.Module):
    def __init__(self, embedding_dim, hop, dropout, pad_idx, tree_depth=1):
        super(EncoderTreeMemNN, self).__init__()
        self.max_hops = hop
        self.nemb = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.Qproj = nn.Linear(embedding_dim, embedding_dim)
        # self.entry_query_proj = nn.Linear(embedding_dim*2, embedding_dim)
        self.LW_H = nn.Linear(embedding_dim, embedding_dim)
        self.pad_idx = pad_idx
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.Qproj.bias.data.zero_()
        self.Qproj.weight.data.uniform_(-initrange, initrange)
        self.LW_H.bias.data.zero_()
        self.LW_H.weight.data.uniform_(-initrange, initrange)

    def entry_atten(self, query, keys, values):
        # (bsize * nemb), (nentries * nemb) -> (bsize * nentries)
        logits = torch.einsum('ik,jk->ij', query, keys)
        prob = nn.functional.softmax(logits, dim=-1)
        # (bsize * nentries), (nentries * nemb) -> (bsize, nemb)
        o_k = torch.einsum('bm,me->be', prob, keys)
        # (bsize * nentries), (bsize, nentries, nemb) -> (bsize, nemb)
        values = torch.einsum('bm,bme->be', prob, values)

        return o_k, logits, values

    def value_atten(self, query, entries):
        # (bsize * nemb), (nentries * entry_size * nemb) -> (bsize * nentries * entry_size)
        logits = torch.einsum('ik,mnk->imn', query, entries)
        prob = nn.functional.softmax(logits, dim=-1)
        # (bsize * nentries * entry_size), (nentries * entry_size * nemb) -> (bsize, nentries, nemb)
        output = torch.einsum('imn,mnk->imk', prob, entries)
        return output

    def forward(self, keys, entries, query, wordemb, true_ents=None):
        """params
        query: (bsize, nemb)
        keys: (nentries,)
        entries: (nentries, entry_size)
        """
        # mask = (1.0 - (entries == self.pad_idx).float()).unsqueeze(2).to(torch.float32)
        # query = self.Qproj(query) # bsize * nemb
        nentries, entry_size = entries.size() # nentries * entry_size
        bsize = query.size(0)
        us = [query] # bsize * nemb
        os = []
        prob_list = []
        # nentries * nemb
        keys = nn.functional.embedding(keys, wordemb, padding_idx=self.pad_idx)
        # nentries * entry_size * nemb
        entries = nn.functional.embedding(entries, wordemb, padding_idx=self.pad_idx)
        for hop in range(self.max_hops):
            # perform attention over entries
            # expanded_query = us[-1].unsqueeze(1).repeat(1, n_entries, 1) # bsize * nentries * nemb
            # expanded_key = keys.unsqueeze(0).repeat(bsize, 1, 1) # bsize * nentries * nemb
            # bsize * nentries * nemb
            # entry_keys = self.entry_query_proj(torch.cat([expanded_query, keys]))
            value_query = self.Qproj(us[-1])
            entries = self.value_atten(value_query, entries)
            o_k, logits, combined_entries = self.entry_atten(us[-1], keys, entries)
            prob_list.append(logits)

            if true_ents is not None:
                combined_entries = torch.index_select(entries, 0, true_ents)
                o_k = torch.index_select(keys, 0, true_ents)
            # u_k = nn.functional.relu(self.LW_H(us[-1]))
            # u_k = u_k + o_k
            # u_k = self.LW_H(us[-1]) + o_k
            u_k = o_k
            us.append(u_k)
        return us[-1], prob_list[-1], combined_entries


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


if __name__ == "__main__":
    """Test
    bsize = 2
    nentries = 3
    entry_size = 2
    nemb = 5
    ntokens = 10
    """
    # test tree attn encoder
    bsize = 2
    nentries = 3
    entry_size = 2
    nemb = 5
    ntokens = 10
    test_encoder = EncoderTreeMemNN(nemb, 1, 0.0, 0)
    keys = torch.LongTensor([7, 8, 9])
    entries = torch.LongTensor([[2, 3], [1, 5], [6, 0]])
    query = torch.rand(bsize, nemb)
    wordemb = torch.rand(ntokens, nemb)
    output = test_encoder(keys, entries, query, wordemb)
    # test hierarchical attention
    test_encoder = EncoderMemNN(ntokens, nemb, 1, 0.0, 0, nemb, nentries)
    output = test_encoder(keys.unsqueeze(1), query, wordemb, hierarchical=2)
