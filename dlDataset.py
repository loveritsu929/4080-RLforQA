import json, os, pickle
from tqdm import tqdm
import torch
import torch.utils.data as data
from bert_serving.client import BertClient

train_file = '/media/data1/hotpot/hotpot_train_v1.1.json'
dev_file = '/media/data1/hotpot/hotpot_dev_fullwiki_v1.json'

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
    
class sentDataset(data.Dataset):
    def __init__(self, mode = 'train'):
        self.mode = mode
        #self.bertClient = BertClient(check_length=False) # localhost, max_seq_lehgth = inf
        self.fp = load_file(train_file, 'jsn') if mode == 'train' else load_file(dev_file, 'jsn') # a list of qusetions
        self.dataset_len, self.dataset = self.make_dataset()
        
    def make_dataset(self):
        # need: question, para_title+sentence, label
        # tensor n * 2048 + label
        ds_len = 0
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
                    label = [0,1] if [title, sent_index] in supports else [1,0]
                    ds.append((question, sent_sample, label))
        
        assert ds_len == len(ds)
        
            
        return ds_len, ds
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    
    
    
    
    
class paraDataset(data.Dataset):
    ''' a dataset for hotpotQA
        pure DL approach
        
        load the entire dataset file into memory
    '''
    
    def __init__(self, mode='train'):
        self.mode = mode
        self.bertClient = BertClient(check_length=False) # localhost, max_seq_lehgth = inf
        self.fp = load_file(train_file, 'jsn') if mode == 'train' else load_file(dev_file, 'jsn') # a list of qusetions
        
    def __len__(self):
        return len(self.fp)
    
    def __getitem__(self, index):
        '''
           return a tensor of all sentences with shape num_sent * bert_embedding_size, a question tensor,
           and a label temsor with size num_sent * 1
        '''
        qdict = self.fp[index]
        supports = qdict['supporting_facts'] # a list of lists of facts with form [title, sent_id]
        question = qdict['question']
        context = qdict['context'] # a list of 2-element list [title, paragraph]
        num_context = len(context)
        
        answer = qdict['answer'] # do not really need it?
        qid = qdict['_id']
        qtype = qdict['type']
        
        # some articles in the fullwiki dev/test sets have zero paragraphs
        if len(context) == 0:
            context = [['some random title', 'some random stuff']]
        
        num_sent = 0
        sents = []
        labels = []
        for title, paragraph in context:
            num_sent += len(paragraph)
            sent_tensor = torch.as_tensor(self.bertClient.encode(paragraph)) # encode entire paragraph ,input: list of strs 
            sents.append(sent_tensor) # size: n * 1024
            for sent_index, sent in enumerate(paragraph):
                sent_label = 1 if [title, sent_index] in supports else 0
                
                label_tensor = torch.as_tensor(sent_label)
                labels.append(label_tensor.unsqueeze_(0))
                
        sent_mat = torch.cat(sents,0)
        label_mat = torch.cat(labels,0)
        
        assert num_sent == sent_mat.size(0) == label_mat.size(0)
        question_vec = torch.as_tensor(self.bertClient.encode([question]))
        
        return question_vec, sent_mat, label_mat

# test case            
if __name__ == '__main__':
    ds = sentDataset(mode='train')
    dl = data.DataLoader(ds, batch_size = 1, shuffle = False, num_workers = 0)
    # num_workers != 0 ==> Error no responding ???
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
        
        