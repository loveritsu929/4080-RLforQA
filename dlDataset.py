import json, os, pickle
from tqdm import tqdm
import torch
import torch.utils.data as data
#from bert_serving.client import BertClient
#bert = BertClient(check_length=False)
train_file = '/media/data1/hotpot/hotpot_train_v1.1.json'
dev_file = '/media/data1/hotpot/hotpot_dev_fullwiki_v1.json'

# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def load_file(file_path, file_type):
    if file_type == "txt":
        with open(file=file_path, mode="rt", encoding="utf-8") as file_stream:
            return file_stream.read().splitlines()

    elif file_type == "jsn":
        with open(file=file_path, mode="rt", encoding="utf-8") as file_stream:
            return json.load(file_stream)

    elif file_type == "obj":
        with open(file=file_path, mode="rb") as file_stream:
            return pickle.load(file_stream)

    else:
        pass

def dump_data(data_buffer, file_path, file_type):
    if file_type == "txt":
        with open(file=file_path, mode="wt", encoding="utf-8") as file_stream:
            file_stream.write("\n".join(data_buffer))

    elif file_type == "jsn":
        with open(file=file_path, mode="wt", encoding="utf-8") as file_stream:
            json.dump(obj=data_buffer, fp=file_stream)

    elif file_type == "obj":
        with open(file=file_path, mode="wb") as file_stream:
            pickle.dump(obj=data_buffer, file=file_stream)

    else:
        pass
    
class SentDataset(data.Dataset):
    # 215662 pos / 3703344
    def __init__(self, mode = 'train'):
        self.mode = mode
        self.fp = load_file(train_file, 'jsn') if mode == 'train' else load_file(dev_file, 'jsn') # a list of qusetions
        self.num_positive, self.dataset_len, self.dataset = self.make_dataset()
        
    def make_dataset(self):
        # need: question, para_title+sentence, label
        # tensor n * 2048 + label
        ds_len = 0
        num_positive= 0
        ds = []
        for qdict in self.fp:
            supports = qdict['supporting_facts'] # a list of lists of facts with form [title, sent_id]
            question = qdict['question']
            context = qdict['context'] # a list of 2-element list [title, paragraph]
        
            answer = qdict['answer'] # do not really need it?
            qid = qdict['_id']
            qtype = qdict['type']
            
            # some articles in the fullwiki dev/test sets have zero paragraphs
            if len(context) == 0:
                context = [['some random title', 'some random stuff']]
            
            for title, paragraph in context:
                ds_len += len(paragraph)
                for sent_index, sent in enumerate(paragraph):
                    sent_sample = title + ": " + sent
                    label = 1 if [title, sent_index] in supports else 0
                    num_positive += 1 if label == 1 else 0
                    ds.append((question, sent_sample, label))
        return num_positive, ds_len, ds
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        return self.dataset[index]
    
class ParaDataset(data.Dataset):
    ''' a dataset for hotpotQA
        pure DL approach
        899667 paras
        load the entire dataset file into memory
    '''
    
    def __init__(self, mode='train'):
        self.mode = mode
        self.fp = load_file(train_file, 'jsn') if mode == 'train' else load_file(dev_file, 'jsn') # a list of qusetions
        self.dataset= self.make_dataset()
        
    def para_padding(self, para_list, maxLen):
        para = para_list
        paraLen = len(para_list)
        diff = maxLen - paraLen
        times = int(maxLen/paraLen)
        if diff > 0:
            para = (para * (times+1))[:maxLen]
        return para
    
    def make_dataset(self):
        # need: question, para_title+sentence, label
        # tensor n * 2048 + label
        ds = []
        for qdict in self.fp:
            supports = qdict['supporting_facts'] # a list of 2-element lists of facts with form [title, sent_id]
            #fact_titles = [fact[0] for fact in supports] # only the titles of supporting paragraphs
            
            question = qdict['question']
            context = qdict['context'] # a list of 2-element list [title, paragraph]
            #answer = qdict['answer'] # do not really need it?
            #qid = qdict['_id']
            
            # some articles in the fullwiki dev/test sets have zero paragraphs
            if len(context) == 0:
                context = [['some random title', ['some random stuff']]]
            context = [[title, list(filter(lambda x: x != '' and x != ' ', para))] for title, para in context] # remove empty sentence in para
            #maxParaLen = max([len(para) for title, para in context])
            for title, paragraph in context:
                label = [0] * len(paragraph)
                for i, _ in enumerate(label):
                    if [title, i] in supports:
                        label[i] = 1
                #paragraph = self.para_padding(paragraph, maxParaLen)
                paragraph[0] = title + ': ' + paragraph[0]
                ds.append((question, paragraph, label))
        return ds
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        q, para, labels = self.dataset[index]
#        q_t = torch.as_tensor(bert.encode([q]))
#        para_t = torch.as_tensor(bert.encode(para))
        return q_t, para_t, labels
    
#    def __getitem__(self, index):
#        '''
#           return a tensor of all sentences with shape num_sent * bert_embedding_size, a question tensor,
#           and a label temsor with size num_sent * 1
#        '''
#        qdict = self.fp[index]
#        supports = qdict['supporting_facts'] # a list of lists of facts with form [title, sent_id]
#        question = qdict['question']
#        context = qdict['context'] # a list of 2-element list [title, paragraph], where paragraph is a list of sentences.
#        num_context = len(context)
#        
#        answer = qdict['answer'] # do not really need it?
#        qid = qdict['_id']
#        qtype = qdict['type']
#        
#        # some articles in the fullwiki dev/test sets have zero paragraphs
#        if len(context) == 0:
#            context = [['some random title', ['some random stuff']]]
#        
#        num_sent = 0
#        sents = []
#        labels = []
#        for title, paragraph in context:
#            num_sent += len(paragraph)
#            sent_list = paragraph # encode entire paragraph ,input: list of strs 
#            sents.append(sent_tensor) # size: n * 1024
##            for sent_index, sent in enumerate(paragraph):
##                sent_label = 1 if [title, sent_index] in supports else 0
##                label_tensor = torch.as_tensor(sent_label)
##                labels.append(label_tensor.unsqueeze_(0))
#                
##        sent_mat = torch.cat(sents,0)
##        label_mat = torch.cat(labels,0)
#        
##        assert num_sent == sent_mat.size(0) == label_mat.size(0)
#        question_vec = torch.as_tensor(self.bertClient.encode([question]))
#        
#        return question_vec, sent_mat, label_mat

# test case       



class SentParaDataset(data.Dataset):
    ''' a dataset for hotpotQA
        pure DL approach
        899667 paras
        load the entire dataset file into memory
    '''
    
    def __init__(self, mode='train'):
        self.mode = mode
        self.fp = load_file(train_file, 'jsn') if mode == 'train' else load_file(dev_file, 'jsn') # a list of qusetions
        self.dataset= self.make_dataset()
        
    def para_padding(self, para_list, maxLen):
        para = para_list
        paraLen = len(para_list)
        diff = maxLen - paraLen
        times = int(maxLen/paraLen)
        if diff > 0:
            para = (para * (times+1))[:maxLen]
        return para
    
    def make_dataset(self):
        # need: question, para_title+sentence, label
        # tensor n * 2048 + label
        ds = []
        for qdict in self.fp:
            supports = qdict['supporting_facts'] # a list of 2-element lists of facts with form [title, sent_id]
            fact_titles = [fact[0] for fact in supports] # only the titles of supporting paragraphs
            
            question = qdict['question']
            context = qdict['context'] # a list of 2-element list [title, paragraph]
            
            # some articles in the fullwiki dev/test sets have zero paragraphs
            if len(context) == 0:
                context = [['some random title', ['some random stuff']]]
            context = [[title, list(filter(lambda x: x != '' and x != ' ', para))] for title, para in context] # remove empty sentence in para
            #maxParaLen = max([len(para) for title, para in context])
            for title, sents in context:
                sent_labels = [0] * len(sents)
                for i, _ in enumerate(sent_labels):
                    if [title, i] in supports:
                        sent_labels[i] = 1
                #paragraph = self.para_padding(paragraph, maxParaLen)
                sents[0] = title + ': ' + sents[0]
                para = ''.join(sents)
                para_label = 1 if title in fact_titles else 0
                ds.append((question, sents, para, sent_labels, para_label))
        return ds
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
#        q, para, labels = self.dataset[index]
#        q_t = torch.as_tensor(bert.encode([q]))
#        para_t = torch.as_tensor(bert.encode(para))
#        return q_t, para_t, labels
        return self.dataset[index]





     
if __name__ == '__main__':
    #Dds1 = ParaDataset(mode='train')
    ds2 = SentParaDataset(mode='test')
    '''
    ds[0] = 
    ("Which magazine was started first Arthur's Magazine or First for Women?",
 ['Radio City (Indian radio station)'
  "Radio City is India's first private FM radio station and was started on 3 July 2001.",
  ' It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).',
  ' It plays Hindi, English and regional songs.',
  ' It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.',
  ' Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.',
  ' The Radio station currently plays a mix of Hindi and Regional music.',
  ' Abraham Thomas is the CEO of the company.'],
 0)
    '''
    #dl = data.DataLoader(ds, batch_size = 2, shuffle = False, num_workers = 0)
 
    '''
    for i, (q, sample, label) in enumerate(dl,0):
        if i > 2:
            break
        else:
            print(sample.size())
    #q, s, l = ds.__getitem__(0)
    
    RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0.
    Got 49 and 34 in dimension 1
    
     sent_mat.view(1,-1)
    '''
    #dl = data.DataLoader(ds, batch_size = 1, shuffle = False, num_workers = 0, collate_fn=my_collate,)
    # num_workers != 0 ==> Error no responding ???
    
    # dev_ds: len==314793, num_positive = 10298, pos% = 3.27 %
    # train_ds: len==3703344, num_positive = 215662, pos% = 5.82 %
    # test true_pos acc on dev_dataset: acc == 8810/10298 == 85.6%

        
        
