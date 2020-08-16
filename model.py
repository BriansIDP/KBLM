from __future__ import print_function
import torch.nn as nn
import math
from torch import cat, randn_like, einsum, zeros, stack
from torch.autograd import Variable
from MemEnc import EncoderMemNN, EncoderTreeMemNN


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, nclass, ninp, nhid_class, class_masks, nhid, nlayers,
                 KB=None, rnndrop=0.5, dropout=0.5, reset=0, useKB=False, pad=0, hop=1,
                 lstm_enc=False, post=True, pretrain=False, pretrain_dim=0,
                 KBvalues=None, usesurface=True, classnorm=False, tied=False, levels=1):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.dropclass = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=rnndrop)
        self.class_encoder = nn.Embedding(nclass, nhid_class)
        self.LMhead = nn.Linear(nhid, ninp)
        if (useKB or tied) and ninp != nhid:
            self.LMdecoder = nn.Linear(ninp, ntoken)
            self.CLdecoder = nn.Linear(nhid_class, nclass)
        else:
            self.LMdecoder = nn.Linear(nhid, ntoken)
            self.CLdecoder = nn.Linear(nhid_class, nclass)
        self.KBgate = nn.Linear(ninp+nhid, nclass)
        if tied:
            # Tie weight
            self.LMdecoder.weight = self.encoder.weight
        self.init_weights()
        # Memory network
        if (useKB and KB is not None):
            if usesurface:
                self.MemNet = EncoderMemNN(ntoken, ninp, hop, dropout, pad, nhid,
                                           lstm_enc, pretrain, pretrain_dim)
            else:
                self.MemNet = EncoderTreeMemNN(ntoken, ninp, hop, dropout, pad, nhid,
                                             lstm_enc, pretrain, pretrain_dim)

        self.tied = tied
        self.nhid = nhid
        self.class_masks = class_masks
        self.nclass = nclass
        self.nhid_class = nhid_class
        self.nlayers = nlayers
        self.reset = reset
        self.useKB = useKB
        self.KB = KB
        self.ninp = ninp
        self.post = post
        self.classnorm = classnorm
        self.levels = levels

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.class_encoder.weight.data.uniform_(-initrange, initrange)
        self.LMhead.bias.data.zero_()
        self.LMhead.weight.data.uniform_(-initrange, initrange)
        self.LMdecoder.bias.data.zero_()
        self.LMdecoder.weight.data.uniform_(-initrange, initrange)
        self.CLdecoder.bias.data.zero_()
        self.CLdecoder.weight.data.uniform_(-initrange, initrange)
        self.KBgate.bias.data.zero_()
        self.KBgate.weight.data.uniform_(-initrange, initrange)

    def forward(self, words, classes, hidden, gating=False, rampup=False, ramp_factor=1,
                true_ents=None, true_class=None, entities=None, forcing_p=0.0):
        seq_len = words.size(0)
        bsize = words.size(1)
        emb = self.drop(self.encoder(words))
        # class_emb = self.class_encoder(classes)
        output_list = []
        class_output_list = []
        ramp_rate = 1.0
        if rampup:
            ramp_rate = ramp_factor
        if true_ents is not None:
            true_ents = true_ents.view(seq_len, bsize)

        KB_embedding = zeros(1, bsize, self.ninp).to(device=emb.device)
        eachoutput = emb[0:1, :, :]
        output = []
        class_distributions = []
        att_list = []
        mem_list = []
        for i in range(seq_len):
            to_input = emb[i:i+1, :, :]
            # if self.useKB and not self.post:
            #     to_input = cat([to_input, eachoutput], dim=-1)
                # hidden = (eachoutput, hidden[-1])
            eachoutput, hidden = self.rnn(to_input, hidden)

            if (self.tied or self.useKB) and self.ninp != self.nhid:
                eachoutput = self.LMhead(eachoutput)

            if self.useKB == "KB" and not self.post:
                KB_embedding, att = self.MemNet(self.KB if entities is None else entities,
                                                hidden[1][-1],
                                                self.encoder.weight.data,
                                                true_ents[i, :] if true_ents is not None else None,
                                                hierarchical=self.levels)
                class_distribution = nn.functional.softmax(self.KBgate(cat([eachoutput, KB_embedding], dim=-1)), dim=1)
                gate = class_distribution[:,0:1]
                KB_embedding = math.floor(ramp_factor) * KB_embedding.unsqueeze(0)
                eachoutput = (1 - gate) * eachoutput + gate * KB_embedding
                att_list.append(att)
            class_decoded = self.CLdecoder(eachoutput.view(bsize, -1))
            class_distribution = nn.functional.softmax(class_decoded, dim=1)
            class_distributions.append(class_distribution)
            output.append(eachoutput)
            mem_list.append(hidden[1][-1].unsqueeze(0))
        output = cat(output, dim=0).view(seq_len*bsize, -1)
        att = stack(att_list).view(seq_len*bsize, -1) if len(att_list) > 0 else None
        mem_list = cat(mem_list, dim=0).view(seq_len*bsize, -1)
        class_distribution = stack(class_distributions).view(seq_len*bsize, -1)

        # Query the knowledge base
        if self.useKB == "KB" and self.post:
            # [seq_len*bsize, ninp]
            KB_embedding, att = self.MemNet(self.KB if entities is None else entities,
                                            mem_list.detach(),
                                            self.encoder.weight.data,
                                            true_ents=true_ents,
                                            hierarchical=self.levels,
                                            forcing_p=forcing_p)
            # gate = nn.functional.softmax(self.KBgate(cat([output, KB_embedding.detach()], dim=-1)), dim=1)[:,0:1]
            gate = class_distribution[:,0:1]
            # gate = 1 - true_class.unsqueeze(-1).to(KB_embedding.dtype)
            output = output  + math.floor(ramp_factor) * gate * KB_embedding
        output = self.drop(output)
        # [seq_len*bsize, ntoken]
        decoded = self.LMdecoder(output)
        if self.classnorm:
            # [seq_len*bsize, nclass, ntoken]
            expanded_wordout = decoded.unsqueeze(1).repeat(1, self.nclass, 1)
            expanded_wordout.masked_fill_(self.class_masks, float(-1e9))
            masked_worddist = nn.functional.softmax(expanded_wordout, dim=2)
            # [seq_len*bsize, 1, nclass] * [seq_len*bsize, nclass, ntoken] -> [seq_len*bsize, 1, ntoken]
            summed_prob = einsum('bij,bjk->bik', class_distribution.unsqueeze(1), masked_worddist).squeeze(1)
        else:
            summed_prob = decoded
        return summed_prob, class_distribution.squeeze(1), hidden, att

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def resetsent(self, hidden, input, eosidx, noiselevel=1):
        if self.rnn_type == 'LSTM':
            outputcell = hidden[0]
            memorycell = hidden[1]
            mask = input != eosidx
            expandedmask = mask.unsqueeze(-1).expand_as(outputcell)
            expandedmask = expandedmask.float()
            reversemask = 1.0 - expandedmask
            return (outputcell*expandedmask, memorycell*expandedmask)
