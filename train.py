# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
from operator import itemgetter

import dataloader
from model import RNNModel
from KB_utils import KB_manager

arglist = []
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nhid_class', type=int, default=200,
                    help='number of LSTM hidden units for class labels')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to embeddings (0 = no dropout)')
parser.add_argument('--rnndrop', type=float, default=0.2,
                    help='dropout applied to rnns (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--evalmode', action='store_true',
                    help='Evaluation only mode')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--reset', type=int, default=0,
                    help='reset on the sentence boundaries')
parser.add_argument('--loss', type=str, default='ce',
                    help='loss functions to be used')
parser.add_argument('--tagloss-scale', type=float, default=0.1,
                    help='tag loss scale')
parser.add_argument('--attloss-scale', type=float, default=0.05,
                    help='KB attention loss scale')
parser.add_argument('--noise_ratio', type=int, default=50,
                    help='set the noise ratio of NCE sampling, the noise')
parser.add_argument('--norm_term', type=int, default=9,
                    help='set the log normalization term of NCE sampling')
parser.add_argument('--factor', type=float, default=0.5,
                    help='interpolation value')
parser.add_argument('--interp', action='store_true',
                    help='Linear interpolate with Ngram')
parser.add_argument('--stream_out', action='store_true',
                    help='Write out stream')
parser.add_argument('--logfile', type=str, default='LOGs/rnn.log',
                    help='path to save the final model')
parser.add_argument('--use_extra', action='store_true',
                    help='Use duplicated part of training set')
parser.add_argument('--useKB', default='',
                    help='Type of Knowledge Base')
parser.add_argument('--usesurface', action='store_true',
                    help='use knowledge surface form')
parser.add_argument('--gating', action='store_true',
                    help='Use class probability to gate the KB')
parser.add_argument('--rampup', action='store_true',
                    help='Learning rate adjustment for Mem Net')
parser.add_argument('--pre_epochs', type=float, default=1,
                    help='How many epochs to perform ramp up')
parser.add_argument('--nhop', type=int, default=1,
                    help='number of attention hops')
parser.add_argument('--use_dsc', action='store_true',
                    help='Use DSC loss')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='DSC gamma smoothing factor')
parser.add_argument('--post', action='store_true',
                    help='include KB after LSTM')
parser.add_argument('--lstmenc', action='store_true',
                    help='use LSTM to encode KB entries')
parser.add_argument('--resume', action='store_true',
                    help='include KB after LSTM')
parser.add_argument('--KBforcing', action='store_true',
                    help='teacher forcing on KB entries')
parser.add_argument('--from-pretrain', action='store_true',
                    help='use pretrained LM for context/KB embeddings')
parser.add_argument('--pretrain-dim', type=int, default=0,
                    help='output dimension from pretrained system')
parser.add_argument('--classnorm', action='store_true',
                    help='use class distribution to normalise probs')
parser.add_argument('--attn-levels', type=int, default=1,
                    help='to perform a multilevel attention')
parser.add_argument('--load-from', type=str, default='',
                    help='load RNNLM from trained model')
parser.add_argument('--KBsize', type=int, default=0,
                    help='The size of the KB for each minibatch')

args = parser.parse_args()

arglist.append(('Data', args.data))
arglist.append(('Model', args.model))
arglist.append(('Embedding Size', args.emsize))
arglist.append(('Hidden Layer Size', args.nhid))
arglist.append(('Layer Number', args.nlayers))
arglist.append(('Learning Rate', args.lr))
arglist.append(('Update Clip', args.clip))
arglist.append(('Max Epochs', args.epochs))
arglist.append(('BatchSize', args.batch_size))
arglist.append(('Sequence Length', args.bptt))
arglist.append(('Dropout', args.dropout))
arglist.append(('Loss Function', args.loss))
arglist.append(('Class hidden dim', args.nhid_class))
arglist.append(('Use class prob to normalise', args.classnorm))
arglist.append(('Use KB', args.useKB))
arglist.append(('Use Gating', args.gating))
arglist.append(('Use LR Rampup', args.rampup))
arglist.append(('No. of epochs for ramp up', args.pre_epochs))
arglist.append(('No. of hops', args.nhop))
arglist.append(('Use DSC loss', args.use_dsc))
arglist.append(('After LSTM', args.post))
arglist.append(('Teacher forcing to train KB', args.KBforcing))
arglist.append(('Attn loss scaling factor', args.attloss_scale))
arglist.append(('Use LSTM encoder', args.lstmenc))
arglist.append(('Use pretrained LM', args.from_pretrain))
arglist.append(('KB retrieval size', args.KBsize))

def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logging("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
train_loader, val_loader, test_loader, dictionary, class_dict = dataloader.create(
    args.data, batchSize=100000000, use_label=True)
eosidx = dictionary.get_eos()
KB = None
if args.useKB in ["KB", "None"]:
    KB = dataloader.KB(args.data, dictionary, eosidx, args.from_pretrain).to(device)
elif args.useKB == "KG":
    KB, KGvalues = dataloader.KG(args.data, dictionary, graphlen=3)
    KB = KB.to(device)
    KGvalues = KGvalues.to(device)

# Small KB collection
KBmanager = KB_manager(KB, args.bptt)

# Get sub-dictionary masks
subdictmask = torch.ones(len(class_dict), len(dictionary))
for tag in class_dict.idx2word:
    dictfile = os.path.join(args.data, 'dicts', 'dict.{}.txt'.format(tag))
    tagidx = class_dict.word2idx[tag]
    with open(dictfile) as fin:
        for line in fin:
            dictidx = dictionary.word2idx[line.strip()]
            subdictmask[tagidx,dictidx] = 0
subdictmask = subdictmask == 1

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    data = torch.LongTensor(data)
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = args.eval_batch_size

# Stream writeout mode
if args.stream_out and eval_batch_size != 1:
    logging('Batch size must be 1 in stream writeout mode!')
    raise

###############################################################################
# Build the model
###############################################################################
ntokens = len(dictionary)
nclasses = len(class_dict)
if args.useKB != "":
    if args.resume:
        model = torch.load(args.save)
    elif args.useKB in ["KB", "None"]:
        model = RNNModel(ntokens, nclasses, args.emsize, args.nhid_class, subdictmask.to(device),
                        args.nhid, args.nlayers, KB, args.rnndrop, args.dropout, reset=args.reset,
                        useKB=args.useKB, pad=eosidx, hop=args.nhop, lstm_enc=args.lstmenc,
                        post=args.post, pretrain=args.from_pretrain, pretrain_dim=args.pretrain_dim,
                        classnorm=args.classnorm, tied=args.tied, levels=args.attn_levels)
    elif args.useKB == "KG":
        model = RNNModel(ntokens, nclasses, args.emsize, args.nhid_class, subdictmask.to(device),
                        args.nhid, args.nlayers, KB, args.rnndrop, args.dropout, reset=args.reset,
                        useKB=True, pad=eosidx, hop=args.nhop, lstm_enc=args.lstmenc,
                        post=args.post, KBvalues=KGvalues, usesurface=args.usesurface,
                        classnorm=args.classnorm, tied=args.tied, levels=args.attn_levels)
else:
    model = RNNModel(ntokens, nclasses, args.emsize, args.nhid_class, subdictmask.to(device), args.nhid,
                     args.nlayers, None, args.rnndrop, args.dropout, reset=args.reset,
                     classnorm=args.classnorm, tied=args.tied)
# Initialise with trained parameters
if args.load_from != '':
    pretrained_dict = torch.load(args.load_from).state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if args.classnorm:
    criterion = nn.NLLLoss()
    interpCrit = nn.NLLLoss(reduction='none')
else:
    criterion = nn.CrossEntropyLoss()
    interpCrit = nn.CrossEntropyLoss(reduction='none')
att_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

if args.cuda:
    model.cuda()
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, train_label, i, train_att=None):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    inputlabel = train_label[i:i+seq_len]
    label = train_label[i+1:i+1+seq_len].view(-1)
    if train_att is not None:
        attlabel = train_att[i:i+seq_len]
        atttarget = train_att[i+1:i+1+seq_len].view(-1)
    else:
        attlabel = None
        atttarget = None
    return data, target, inputlabel, label, attlabel, atttarget

def DSC(probs, targets):
    self_adjusting_factor = probs[:,1] * (1 - probs[:,1])
    targets = targets.type(torch.cuda.FloatTensor)
    dsc_loss_num = 2 * self_adjusting_factor * targets + args.gamma
    dsc_loss_denom = self_adjusting_factor + targets + args.gamma
    dsc_loss = 1 - dsc_loss_num / dsc_loss_denom
    return torch.mean(dsc_loss)

def DL(probs, targets):
    self_adjusting_factor = probs[:,1] * (1 - probs[:,1])
    targets = targets.type(torch.cuda.FloatTensor)
    dsc_loss_num = 2 * probs[:,1] * targets + args.gamma
    dsc_loss_denom = probs[:,1] ** 2 + targets ** 2 + args.gamma
    dsc_loss = 1 - dsc_loss_num / dsc_loss_denom
    return torch.mean(dsc_loss)

def evaluate(data_source, eval_labels, eval_att=None, rampfactor=1.0):
    # Turn on evaluation mode which disables dropout.
    model.to(device)
    model.eval()
    total_tag_loss = 0.
    total_word_loss = 0.
    total_oracle_loss = 0.
    total_atten_loss = 0.
    total_NE = 0.
    stout = []
    ntokens = len(dictionary)
    hidden = model.init_hidden(eval_batch_size)
    att_criterion.reduction = "none"
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets, inputlabel, labels, _, atttarget = get_batch(data_source, eval_labels, i, train_att=eval_att)
            # get att targets
            # thisKB, atttarget = KBmanager.get_this_KB_test(atttarget)
            # Use true KB entries or predicted entries
            # print(atttarget)
            if args.KBforcing:
                worddist, classdist, hidden, attprob = model(
                    data, inputlabel, hidden, args.gating, args.rampup, rampfactor,
                    true_ents=atttarget, true_class=labels, entities=KBmanager.testKB)
            else:
                worddist, classdist, hidden, attprob = model(
                    data, inputlabel, hidden, args.gating, args.rampup, rampfactor,
                    true_class=labels, entities=KBmanager.testKB)
            # Determine the output type
            if args.classnorm:
                word_loss = criterion(torch.log(worddist+1e-9), targets)
                oracle_loss = interpCrit(torch.log(worddist), targets)
            else:
                word_loss = criterion(worddist, targets)
                oracle_loss = interpCrit(worddist, targets)
            if args.use_dsc:
                tag_loss = DSC(classdist, labels)
            else:
                tag_loss = criterion(torch.log(classdist), labels)
            attention_loss = att_criterion(attprob, atttarget) if attprob is not None else torch.zeros(1)
            # calculate NE loss
            label_mask = labels != 1
            oracle_loss = torch.sum(oracle_loss * label_mask.float())
            total_oracle_loss += oracle_loss
            total_NE += torch.sum(label_mask)

            if args.stream_out:
                final_prob = torch.gather(worddist, 1, targets.unsqueeze(1))
                final_class_prob = classdist[:,0]
                stout += list(zip(final_prob.squeeze(1).tolist(), final_class_prob.tolist(), targets.tolist()))
            total_word_loss += word_loss * data.size(0)
            total_tag_loss += tag_loss * data.size(0)
            total_atten_loss += attention_loss.sum()
            hidden = repackage_hidden(hidden)
        total_word_loss /= len(data_source)
        total_tag_loss /= len(data_source)
        total_oracle_loss /= total_NE
        total_atten_loss /= total_NE
    return total_word_loss, stout, total_tag_loss, total_oracle_loss, total_atten_loss

def train(epoch, model, train_data, train_label, train_att, lr, use_dsc=False, att_loss_scale=0.0):
    # Turn on training mode which enables dropout.
    model.train()
    total_tag_loss = 0.
    total_word_loss = 0.
    total_att_loss = 0.
    ramp_factor = 1.
    n_batches = len(train_data) // args.bptt
    start_time = time.time()
    ntokens = len(dictionary)
    hidden = model.init_hidden(args.batch_size)
    att_criterion.reduction = "mean"
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wdecay)
    forcing_p = (1 - epoch / args.epochs) ** 2 * 0.5
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets, inputlabel, labels, attlabel, atttarget = get_batch(
            train_data, train_label, i, train_att=train_att)
        # get small KB for better attention
        thisKB, atttarget = KBmanager.get_this_KB(batch, atttarget)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        # gs534 add sentence resetting
        eosidx = dictionary.get_eos()
        if args.rampup and args.pre_epochs != 0 and epoch < args.pre_epochs:
            ramp_factor = (epoch - 1) / args.pre_epochs
        if args.KBforcing:
            worddist, classdist, hidden, att_prob = model(
                data, inputlabel, hidden, args.gating, args.rampup, ramp_factor,
                true_ents=atttarget, true_class=labels, entities=thisKB, forcing_p=forcing_p)
        else:
            worddist, classdist, hidden, att_prob = model(
                data, inputlabel, hidden, args.gating, args.rampup, ramp_factor,
                true_class=labels, entities=thisKB)
        # Determine the output type
        if args.classnorm:
            word_loss = criterion(torch.log(worddist+1e-9), targets)
        else:
            word_loss = criterion(worddist, targets)

        if args.use_dsc:
            tag_loss = DSC(classdist, labels)
        else:
            tag_loss = criterion(torch.log(classdist+1e-9), labels)

        loss = word_loss + args.tagloss_scale * tag_loss
        # calcualte attention loss
        if att_prob is not None:
            attention_loss = att_criterion(att_prob, atttarget)
            loss += attention_loss * att_loss_scale
            # import pdb; pdb.set_trace()
        if torch.isnan(loss).any():
            import pdb; pdb.set_trace()
            raise
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)
        optimizer.step()

        total_word_loss += word_loss.item()
        total_tag_loss += tag_loss.item()
        total_att_loss += attention_loss.item() if att_prob is not None else 0

        if batch % args.log_interval == 0 and batch > 0:
            cur_word_loss = total_word_loss / args.log_interval
            cur_tag_loss = total_tag_loss / args.log_interval
            cur_att_loss = total_att_loss / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'tag_loss {:5.2f} | att_loss {:.2f} | ppl {:8.2f} | ramp {:2.5f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_tag_loss, cur_att_loss, math.exp(cur_word_loss), ramp_factor))
            total_word_loss = 0
            total_tag_loss = 0
            total_att_loss = 0
            start_time = time.time()
    return ramp_factor

def loadNgram(path):
    probs = []
    with open(path) as fin:
        for line in fin:
            probs.append(float(line.strip()))
    return torch.Tensor(probs)

logging('Training Start!')
for pairs in arglist:
    logging(pairs[0] + ': ' + str(pairs[1]))
# Loop over epochs.
lr = args.lr
best_val_loss = None
att_loss_scale = args.attloss_scale
best_val_att_loss = None

# At any point you can hit Ctrl + C to break out of training early.
if not args.evalmode:
    try:
        for epoch in range(1, args.epochs+int(args.pre_epochs)+1):
            epoch_start_time = time.time()
            rampfactor = 0.
            for train_batched in train_loader:
                train_data, train_label, train_att = list(zip(*train_batched))
                train_data = batchify(train_data, args.batch_size)
                train_label = batchify(train_label, args.batch_size)
                train_att = batchify(train_att, args.batch_size)
                # split into small KBs
                KBmanager.split_KB_entries(args.KBsize, train_att)
                rampfactor = train(epoch, model, train_data, train_label, train_att, lr, att_loss_scale=att_loss_scale)
            
            # start cross-validation
            aggregate_valloss = 0.
            aggregate_valtagloss = 0.
            aggregate_valattloss = 0.
            aggregate_valoracleloss = 0.
            total_valset = 0
            for val_batched in val_loader:
                val_data, val_label, val_att = list(zip(*val_batched))
                databatchsize = len(val_batched)
                val_data = batchify(val_data, eval_batch_size)
                val_label = batchify(val_label, eval_batch_size)
                val_att = batchify(val_att, eval_batch_size)
                # split into small KBs
                val_att = KBmanager.split_KB_entries_test(val_att)
                val_loss, _, val_tag_loss, val_oracle_loss, val_att_loss = evaluate(val_data, val_label, val_att, rampfactor)
                aggregate_valloss += databatchsize * val_loss
                aggregate_valtagloss += databatchsize * val_tag_loss
                aggregate_valattloss += databatchsize * val_att_loss
                aggregate_valoracleloss += databatchsize * val_oracle_loss
                total_valset += databatchsize
            # average losses over pieces of validation data
            val_loss = aggregate_valloss / total_valset    
            val_tag_loss = aggregate_valtagloss / total_valset
            val_oracle_loss = aggregate_valoracleloss / total_valset
            val_att_loss = aggregate_valattloss / total_valset
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid tag loss {:5.2f} | '
                    'valid att loss {:2.2f} | valid oracle ppl {:2.3f} | valid ppl {:8.2f}'.format(
                        epoch, (time.time() - epoch_start_time),
                        val_tag_loss, val_att_loss, math.exp(val_oracle_loss), math.exp(val_loss)))
            logging('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 2.0
            if not best_val_att_loss or val_att_loss < best_val_att_loss:
                best_val_loss = val_loss
            else:
                att_loss_scale /= 4
    except KeyboardInterrupt:
        logging('-' * 89)
        logging('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Set cpu evaluate mode
device = torch.device("cuda")

if args.evalmode:
    # start cross-validation
    aggregate_valloss = 0.
    aggregate_valtagloss = 0.
    aggregate_valattloss = 0.
    aggregate_valoracleloss = 0.
    total_valset = 0
    for val_batched in val_loader:
        val_data, val_label, val_att = list(zip(*val_batched))
        databatchsize = len(val_batched)
        val_data = batchify(val_data, eval_batch_size)
        val_label = batchify(val_label, eval_batch_size)
        val_att = batchify(val_att, eval_batch_size)
        # split into small KBs
        val_att = KBmanager.split_KB_entries_test(val_att)
        val_loss, _, val_tag_loss, val_oracle_loss, val_att_loss = evaluate(val_data, val_label, val_att)
        aggregate_valloss += databatchsize * val_loss
        aggregate_valtagloss += databatchsize * val_tag_loss
        aggregate_valattloss += databatchsize * val_att_loss
        aggregate_valoracleloss += databatchsize * val_oracle_loss
        total_valset += databatchsize
    # average losses over pieces of validation data
    val_loss = aggregate_valloss / total_valset
    val_tag_loss = aggregate_valtagloss / total_valset
    val_oracle_loss = aggregate_valoracleloss / total_valset
    val_att_loss = aggregate_valattloss / total_valset
    logging('-' * 89)
    logging('| valid tag loss {:5.2f} | '
            'valid att loss {:2.2f} | valid oracle ppl {:2.3f} | valid ppl {:8.2f}'.format(
                val_tag_loss, val_att_loss, math.exp(val_oracle_loss), math.exp(val_loss)))
    logging('-' * 89)

# Run on test data.
test_start_time = time.time()
# Write out probabilities
if args.stream_out and args.evalmode:
    evalstfile = open('eval_1.st', 'w')
# Start test evaluation
total_testset = 0
aggregate_testloss = 0.
aggregate_testtagloss = 0
aggregate_testattloss = 0
aggregate_testoracleloss = 0.
for test_batched in test_loader:
    test_data, test_label, test_att = list(zip(*test_batched))
    databatchsize = len(test_batched)
    test_data = batchify(test_data, eval_batch_size)
    test_label = batchify(test_label, eval_batch_size)
    test_att = batchify(test_att, eval_batch_size)
    # split into small KBs
    test_att = KBmanager.split_KB_entries_test(test_att)
    test_loss, stout, test_tag_loss, oracle_loss, test_att_loss = evaluate(test_data, test_label, test_att)
    if args.stream_out:
        evalstfile.write('P_word\tP_class\tWORD\n')
        for p1, p2, idx in stout:
            word = dictionary.idx2word[idx]
            evalstfile.write('{:1.8f}\t{:1.8f}\t{}\n'.format(p1, p2, word))
    aggregate_testloss += databatchsize * test_loss
    aggregate_testtagloss += databatchsize * test_tag_loss
    aggregate_testattloss += databatchsize * test_att_loss
    aggregate_testoracleloss += databatchsize * oracle_loss
    total_testset += databatchsize
test_loss = aggregate_testloss / total_testset
test_tag_loss = aggregate_testtagloss / total_testset
test_att_loss = aggregate_testattloss / total_testset
test_oracle_loss = aggregate_testoracleloss / total_testset
logging('-' * 89)
logging('| test tag loss {:5.2f} | '
        'test att loss {:2.2f} | test oracle loss {:5.3f} | test ppl {:8.2f}'.format(
            test_tag_loss, test_att_loss, math.exp(test_oracle_loss), math.exp(test_loss)))
logging('-' * 89)
logging('Test time cost: {:5.2f} ms'.format(time.time() - test_start_time))

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

