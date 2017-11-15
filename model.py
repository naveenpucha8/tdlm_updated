import torch
import numpy as np
import math
import scipy.stats
from gensim import matutils
import torch.nn.functional as F
from torch.autograd import Variable
import constants as cf
class TopicModel(torch.nn.Module):
    def __init__(self, is_training, vocab_size, batch_size, num_steps, num_classes, cf):
    	super(TopicModel, self).__init__()
 	self.conv_size = cf.filter_number
 	self.doc = Variable(torch.IntTensor(cf.doc_len),requires_grad=True)
 	self.y = Variable(torch.IntTensor(num_steps),requires_grad=True)
 	self.tm_mask = Variable(torch.FloatTensor(num_steps),requires_grad=True)
 	self.conv_word_embedding=torch.FloatTensor(vocab_size, 50).zero_()
        self.topic_output_embedding = Variable(torch.rand(cf.k,cf.topic_embedding_size),requires_grad=True)
        self.topic_input_embedding = Variable(torch.rand(cf.k,self.conv_size),requires_grad=True)
        #Now defining the conv and linear layers
        #conv1 for forming the document vector d of size a
        self.conv1=torch.nn.Conv2d(1,cf.filter_number,(cf.filter_sizes,50), stride=(1,1), padding=0)
	self.conv1.weights=torch.rand(cf.filter_sizes,50)
        #Linear layer for tm_logits from conv_hidden values #Take Care
        self.linear1=torch.nn.Linear(cf.topic_embedding_size,vocab_size,bias=True)
	self.linear1.weights=torch.rand(cf.topic_embedding_size,vocab_size)
	self.tm_cost=0
	self.batch_size=batch_size
	self.num_steps=num_steps
	self.is_training=is_training
	self.tm_train_op=None
   	self.cf=cf
    def pre(self,y,m,d,t):
    	self.doc=d
    	self.y = y
        self.tm_mask = m
        self.tag = t
	count=-1
	count2=-1
        self.doc_inputs=Variable(torch.FloatTensor(64,300,50))
	for i in self.doc:
		count+=1
		count2=-1
        	for j in i:
			count2+=1
        		self.doc_inputs[count,count2]=self.conv_word_embedding[j]
	#self.doc_inputs = torch.from_numpy(np.array(self.doc_inputs))
	if self.is_training and self.cf.tm_keep_prob < 1.0:
	    model=torch.nn.Dropout(cf.tm_keep_prob)
            self.doc_inputs = model(self.doc_inputs)
	#print self.doc_inputs.size()
        self.doc_inputs = self.doc_inputs.unsqueeze(1)
	#print self.doc_inputs.size()

	return self.doc_inputs

    def forward(self,doc_inputs):

	#doc_inputs=doc_inputs.unsqueeze(-1)
	#doc_inputs=doc_inputs.repeat(1,1,1,self.cf.filter_number)
	print doc_inputs.size()
	output=self.conv1(doc_inputs)
	h=F.max_pool2d(output,(299,1), stride=(1,1), padding=0)
	pooled_outputs=[]
	pooled_outputs.append(h)
	print h.size()
	conv_pooled=torch.cat(pooled_outputs,3)
	conv_pooled = conv_pooled.view(-1, self.conv_size)
	print conv_pooled.size()
	x=torch.sum(self.topic_input_embedding.unsqueeze(0).mul(conv_pooled.unsqueeze(1)), 2)
	self.attention = F.log_softmax(x)

        self.mean_topic = torch.sum(self.attention.unsqueeze(2).mul(self.topic_output_embedding.unsqueeze(0)),1)

	if self.is_training and cf.tm_keep_prob < 1.0:
            self.mean_topic_dropped = F.dropout(self.mean_topic, cf.tm_keep_prob)
        else:
            self.mean_topic_dropped = self.mean_topic

        self.conv_hidden = self.mean_topic_dropped.repeat(1, self.num_steps)

	self.conv_hidden= self.conv_hidden.view(self.batch_size*self.num_steps, cf.topic_embedding_size)
        #print self.conv_hidden[0]
	self.tm_logits=self.linear1(self.conv_hidden)
	self.tm_logits=F.log_softmax(self.tm_logits)
        self.tm_cost=0
        return self.tm_logits
    def get_topics(self,  topn):
        topics = []
        entropy = []
	tw_dist=torch.nn.Softmax()
	tw_dist=tw_dist(self.linear1(self.topic_output_embedding))
	print tw_dist
        for ti in xrange(self.cf.topic_number):
	    zz=tw_dist[ti].data
            best = matutils.argsort(zz.numpy(), topn=topn, reverse=True)
            topics.append(best)
            entropy.append(scipy.stats.entropy(zz.numpy()))
        return topics, entropy

class LanguageModel(TopicModel):
    def __init__(self, is_training, vocab_size, batch_size, num_steps, num_classes, cf):
        if cf.topic_number > 0:
             TopicModel.__init__(self, is_training, vocab_size, batch_size, num_steps, 0, cf)
        self.lstm_word_embedding=torch.FloatTensor(vocab_size, 50).zero_()
        self.x = Variable(torch.IntTensor(num_steps),requires_grad=True)
     	self.lm_mask = Variable(torch.FloatTensor(num_steps),requires_grad=True)

        if config.topic_number > 0:
            self.w = Variable(torch.rand(cf.topic_embedding_size, cf.rnn_hidden_size),requires_grad=True)
            self.u = Variable(torch.rand(cf.rnn_hidden_size,  cf.rnn_hidden_size),requires_grad=True)
            self.b = Variable(torch.rand(cf,rnn_hidden_size),requires_grad=True)
        # self.lm_softmax_w = Variable(torch.rand(cf.rnn_hidden_size, vocab_size),requires_grad=True)
        
