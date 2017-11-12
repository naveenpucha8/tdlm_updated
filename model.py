import torch
import numpy as np
import math
import scipy.stats
import torch.nn.functional as F
from torch.autograd import Variable

class TopicModel(torch.nn.Module):
    def __init__(self, is_training, vocab_size, batch_size, num_steps, num_classes, cf):
 		self.conv_size = len(cf.filter_sizes)*cf.filter_number
 		self.doc = Variable(torch.IntTensor(cf.doc_len),requires_grad=True)
 		self.y = Variable(torch.IntTensor(num_steps),requires_grad=True)
 		self.tm_mask = Variable(torch.FloatTensor(num_steps),requires_grad=True)
 		self.conv_word_embedding=torch.FloatTensor(vocab_size, 50).zero_()
        self.topic_output_embedding = Variable(torch.FloatTensor(cf.k,cf.topic_embedding_size),requires_grad=True)
        self.topic_input_embedding = Variable(torch.FloatTensor(cf.k,self.conv_size),requires_grad=True)
        #Now defining the conv and linear layers 
        #conv1 for forming the document vector d of size a 
        self.conv1=torch.nn.Conv2d(1,,(cf.filter_sizes,50,1,cf.filter_number), stride=(1,1,1,1), padding=1)
        #Linear layer for tm_logits from conv_hidden values #Take Care
        self.linear1=torch.nn.Linear((batch_size*num_steps, cf.topic_embedding_size),batch_size*num_steps, vocab_size)
    	self.flag=0
    def pre(self,y,m,d,t):
    	self.doc=d
    	self.y = y
        self.tm_mask = m
        self.tag = t
        self.doc_inputs=[]
        for i in self.doc:
        	for j in i:
        		self.doc_inputs.append(self.conv_word_embedding[j])
        self.doc_inputs = torch.from_numpy(np.array(doc_inputs))
	
        if is_training and cf.tm_keep_prob < 1.0:
            self.doc_inputs = torch.nn.Dropout(self.doc_inputs, cf.tm_keep_prob, seed=1)
        self.doc_inputs = doc_inputs.unsqueeze(-1)
        if self.flag==0:
        	self.flag=1
        	print self.doc_inputs
    







