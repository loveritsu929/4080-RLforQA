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
    

class dlDataset(data.Dataset):
    ''' a dataset for hotpotQA
        pure DL approach
    '''
    
    def __init__(self, mode='train'):
        self.mode = mode
        self.bertClient = BertClient() # localhost
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
        qtypr = qdict['type']
        
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
                
                #sents.append(sent_tensor.unsqueeze_(0))
                labels.append(label_tensor.unsqueeze_(0))
                
        sent_mat = torch.cat(sents,0)
        label_mat = torch.cat(labels,0)
        
        assert num_sent == sent_mat.size(0) == label_mat.size(0)
        question_vec = torch.as_tensor(self.bertClient.encode([question]))
        
        return question_vec, sent_mat, label_mat

# test case            
if __name__ == '__main__':
    ds = dlDataset(mode='train')
    #q, s, l = ds.__getitem__(0)
    
        
        