import sys, os
import random
import json

from torch.utils.data import Dataset, DataLoader
import torch
# from fairseq.models.roberta import RobertaModel

class Dictionary(object):
    def __init__(self, dictfile):
        self.word2idx = {}
        self.idx2word = []
        self.build_dict(dictfile)

    def build_dict(self, dictfile):
        with open(dictfile, 'r', encoding="utf8") as f:
            for line in f:
                index, word = line.strip().split(' ')
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def get_eos(self):
        return self.word2idx['<eos>']

    def __len__(self):
        return len(self.idx2word)

class LMdata(Dataset):
    def __init__(self, datafile, labelfile, dictionary, label_dict, split, atten_label_file=None):
        '''Load data_file'''
        self.data = []
        self.label = []
        self.split = split
        with open(datafile, 'r') as f:
            datalines = f.readlines()
        with open(labelfile, 'r') as f:
            labellines = f.readlines()
        self.attenlabel = []
        if atten_label_file is not None:
            with open(atten_label_file, 'r') as f:
                attenlabels = f.readlines()
        for i, dataline in enumerate(datalines):
            labelline = labellines[i]
            # Attention labels
            if atten_label_file is not None:
                attenline = attenlabels[i].split()
            for j, tup in enumerate(zip(dataline.split(), labelline.split())):
                word, label = tup
                if atten_label_file is not None:
                    # self.attenlabel.append(int(random.choice(attenline[j].split(','))))
                    self.attenlabel.append(int(attenline[j].split(',')[0]))
                if word in dictionary.word2idx:
                    self.data.append(dictionary.word2idx[word])
                else:
                    self.data.append(dictionary.word2idx['OOV'])
                self.label.append(label_dict.word2idx[label] if label in label_dict.word2idx else label_dict.word2idx['OOV'])

            self.data.append(dictionary.word2idx['<eos>'])
            self.label.append(label_dict.word2idx['O'])
            if atten_label_file is not None:
                self.attenlabel.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.attenlabel != []:
            return self.data[idx], self.label[idx], self.attenlabel[idx]
        else:
            return self.data[idx], self.label[idx], None

def collate_fn(batch):
    return batch

def create(datapath, batchSize=1, shuffle=False, workers=0, use_label=True):
    loaders = []
    dictfile = os.path.join(datapath, 'dictionary.txt')
    dictionary = Dictionary(dictfile)
    class_dict = Dictionary(os.path.join(datapath, 'label_dict.txt'))
    dirs = ['Train', 'Val', 'Test']
    for i, split in enumerate(['train', 'valid', 'test']):
        data_file = os.path.join(datapath, dirs[i], '%s_nlp.txt' %split)
        label_file = os.path.join(datapath, dirs[i], 'new_%s_labels.txt' %split)
        if use_label:
            atten_label_file = os.path.join(datapath, dirs[i], 'atten_labels.txt')
            dataset = LMdata(data_file, label_file, dictionary, class_dict, split, atten_label_file)
        else:
            dataset = LMdata(data_file, label_file, dictionary, class_dict, split)
        loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=workers))
    return loaders[0], loaders[1], loaders[2], dictionary, class_dict

def KB(datapath, dictionary, padID, pretrain=False):
    KBfile = os.path.join(datapath, 'normalised_ents.txt')
    maxlen = 0
    KBentries = []
    with open(KBfile) as fin:
        lines = fin.readlines()

    # if pretrain and not os.path.isfile("/home/dawna/gs534/KBauxLM/data/SWBD_NE/KB.pt"):
    #     roberta = RobertaModel.from_pretrained(
    #         '/home/dawna/gs534/fairseq/models/roberta.base', checkpoint_file='model.pt')
    #     roberta.cuda()
    #     maxlen = 0
    #     for line in lines:
    #         elems = line.strip().lower()
    #         with torch.no_grad():
    #             doc = roberta.extract_features_aligned_to_words(elems)
    #         if len(doc[1:-1]) > maxlen:
    #             maxlen = len(doc[1:-1])
    #         KBentries.append(torch.stack([feat.vector for feat in doc[1:-1]]))
    #     KB = torch.nn.utils.rnn.pad_sequence(KBentries, batch_first=True)
    #     with open("/home/dawna/gs534/KBauxLM/data/SWBD_NE/KB.pt", 'wb') as f:
    #         torch.save(KB, f)
    if pretrain:
        with open("/home/dawna/gs534/KBauxLM/data/SWBD_NE/KB.pt", 'rb') as f:
            KB = torch.load(f)
    else:
        for line in lines:
            elems = line.lower().split()
            ids = []
            for elem in elems:
                if elem == '<void>':
                    ids.append(dictionary.word2idx['<eos>'])
                elif elem not in dictionary.idx2word:
                    ids.append(dictionary.word2idx['OOV'])
                else:
                    ids.append(dictionary.word2idx[elem])
            # if dictionary.word2idx['OOV'] not in ids:
            KBentries.append(ids)
            if len(ids) > maxlen:
                maxlen = len(ids)
        KB = []
        for entry in KBentries:
            if len(entry) < maxlen:
                entry = entry + [padID] * (maxlen - len(entry))
            KB.append(entry)
        KB = torch.LongTensor(KB)
    return KB

def KG(datapath, dictionary, graphlen=5):
    KBfile = os.path.join(datapath, 'entity_links', 'entity_links_no_phrase.json')
    KBroot = os.path.join(datapath, 'normalised_ents.txt')
    KBentries = [[dictionary.word2idx['OOV']] * graphlen]
    KBkeys = [dictionary.word2idx['OOV']]
    with open(KBfile) as fin:
        Kgraph = json.load(fin)
    with open(KBroot) as fin:
        ents = [line.strip() for line in fin]
    # assert len(ents) == len(list(Kgraph.keys()))
    for entity in ents:
        KBkeys.append(dictionary.word2idx[entity])
        links = Kgraph[entity] if entity in Kgraph else []
        if len(links) > graphlen:
            links = links[:graphlen]
        else:
            links += [entity] * (graphlen - len(links))
        KBentries.append([dictionary.word2idx[link] for link in links])
    return torch.LongTensor(KBkeys).unsqueeze(1), torch.LongTensor(KBentries)


if __name__ == "__main__":
    datapath = sys.argv[1]
    traindata, valdata, testdata, dictionary, class_dict = create(datapath, batchSize=10, workers=0)
    for i_batch, sample_batched in enumerate(traindata):
        import pdb; pdb.set_trace()
        print(i_batch, sample_batched.size())
