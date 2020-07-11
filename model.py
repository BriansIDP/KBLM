from __future__ import print_function
import torch.nn as nn
from torch import cat, randn_like, einsum, zeros, stack
from torch.autograd import Variable
from MemEnc import EncoderMemNN


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, nclass, ninp, nhid_class, class_masks, nhid, nlayers,
                 KB=None, rnndrop=0.5, dropout=0.5, reset=0, useKB=False, pad=0, hop=1,
                 lstm_enc=False, post=True, pretrain=False, pretrain_dim=0, KBvalues=None):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.dropclass = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if not post:
            self.rnn = nn.LSTM(ninp * 2, nhid, nlayers, dropout=rnndrop)
        else:
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=rnndrop)
        self.class_encoder = nn.Embedding(nclass, nhid_class)
        self.LMhead = nn.Linear(nhid, ninp)
        if useKB:
            self.LMdecoder = nn.Linear(ninp, ntoken)
            self.CLdecoder = nn.Linear(nhid_class, nclass)
        else:
            self.LMdecoder = nn.Linear(nhid, ntoken)
            self.CLdecoder = nn.Linear(nhid_class, nclass)
        # self.CLhead = nn.Linear(nhid_class, nhid)
        # self.CLdecoder.weight = self.class_encoder.weight
        # Memory network
        if useKB and KB is not None:
            nentries = KB.size(0)
            seqlen = KB.size(1)
            self.MemNet = EncoderMemNN(ntoken, ninp, hop, dropout, pad, nhid,
                                       nentries, seqlen, lstm_enc, pretrain, pretrain_dim)
            # Tie weight
            self.LMdecoder.weight = self.encoder.weight

        self.init_weights()

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

    def forward(self, words, classes, hidden, gating=False, rampup=False, ramp_factor=1, true_ents=None):
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

        KB_embedding = None
        eachoutput = emb[0:1, :, :]
        output = []
        class_distributions = []
        att_list = []
        for i in range(seq_len):
            if self.useKB and not self.post:
            #     query = self.LMhead(hidden[0][-1:,:,:].view(bsize, -1))
            #     KB_embedding, att = self.MemNet(self.KB, query, self.encoder.weight.data)
            #     att_list.append(att)
                to_input = cat([emb[i:i+1, :, :], eachoutput], dim=2)
            else:
                to_input = emb[i:i+1, :, :],
            eachoutput, hidden = self.rnn(to_input, hidden)

            # forward class distribution
            class_output = self.dropclass(eachoutput)
            class_decoded = self.CLdecoder(class_output.view(bsize, -1))
            class_distribution = nn.functional.softmax(class_decoded, dim=1)
            class_distributions.append(class_distribution)

            if self.useKB:
                eachoutput = self.LMhead(eachoutput)
                if not self.post:
                    KB_embedding, att = self.MemNet(self.KB,
                                                    eachoutput.view(bsize, -1),
                                                    self.encoder.weight.data,
                                                    true_ents[i, :] if true_ents is not None else None)
                    eachoutput = eachoutput + 0.5 * ramp_factor * class_distribution[:,1:] * KB_embedding
                    att_list.append(att)

            output.append(eachoutput)
        output = cat(output, dim=0).view(seq_len*bsize, -1)
        att = stack(att_list).view(seq_len*bsize, -1) if len(att_list) > 0 else None
        class_distribution = stack(class_distributions).view(seq_len*bsize, -1)

        # class_output = self.dropclass(output)
        # class_decoded = self.CLdecoder(class_output.view(seq_len*bsize, -1))
        # [seq_len*bsize, nclass] -> [seq_len*bsize, 1, nclass]
        # class_distribution = nn.functional.softmax(class_decoded, dim=1)
        # Query the knowledge base
        if self.useKB and self.post:
            # [seq_len*bsize, ninp]
            KB_embedding, att = self.MemNet(self.KB, output, self.encoder.weight.data)
            gate = class_distribution[:,1:]
            output = output + 0.5 * ramp_factor * gate * KB_embedding
        output = self.drop(output)
        # [seq_len*bsize, ntoken]
        decoded = self.LMdecoder(output)
        # [seq_len*bsize, nclass, ntoken]
        expanded_wordout = decoded.unsqueeze(1).repeat(1, self.nclass, 1)
        expanded_wordout.masked_fill_(self.class_masks, float(-1e9))
        masked_worddist = nn.functional.softmax(expanded_wordout, dim=2)
        # [seq_len*bsize, 1, nclass] * [seq_len*bsize, nclass, ntoken] -> [seq_len*bsize, 1, ntoken]
        summed_prob = einsum('bij,bjk->bik', class_distribution.unsqueeze(1), masked_worddist)
        return summed_prob.squeeze(1), class_distribution.squeeze(1), hidden, att

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
