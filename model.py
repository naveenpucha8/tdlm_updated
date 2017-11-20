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
        self.tm_cost=10
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
        # self.lm_softmax_w = Variable(torch.rand(cf.rnn_hidden_size, vocab_size),requires_grad=True)

        if cf.topic_number > 0:
            self.w = Variable(torch.rand(cf.topic_embedding_size, cf.rnn_hidden_size),requires_grad=True)
            self.u = Variable(torch.rand(cf.rnn_hidden_size,  cf.rnn_hidden_size),requires_grad=True)
            self.b = Variable(torch.rand(cf.rnn_hidden_size),requires_grad=True)

            #define lstm cell
            #
        print cf.rnn_hidden_size
        self.lstm_cell = torch.nn.LSTM(0,cf.rnn_hidden_size,dropout=0.6)
        print self.lstm_cell
        # if self.is_training and cf.lm_keep_prob < 1.0:
        #     self.lstm_cell = torch.nn.Dropout(cf.lm_keep_prob)
        self.cell = self.lstm_cell
        # self.cell = torch.nn.RNN([self.lstm_cell]*cf.rnn_layer_size,)
        # inp = Variable(torch.randn(64,240))
        # self.initial_state = self.cell(inp,(0,0))
        self.initial_state = self.cell(batch_size, torch.FloatTensor)
        self.linear1=torch.nn.Linear(cf.rnn_hidden_size,vocab_size,bias=True)


    def pre(self, x):
        inputs=Variable(torch.FloatTensor(64,300,50))
        count = -1
        count2 = -1
        # print self.conv_hidden
        self.x = x
        for i in self.x:
		count+=1
		count2=-1
        	for j in i:
			count2+=1
        		inputs[count,count2]=self.lstm_word_embedding[j]

        if self.is_training and self.cf.lm_keep_prob < 1.0:
            model = torch.nn.Dropout(cf.lm_keep_prob)
            inputs = model(inputs)

        inputs = [torch.squeeze(input_, 1) for input_ in torch.split(inputs, self.num_steps, 1)]

        # outputs, self.state = torch.nn.rnn(self.cell, inputs, initial_state=self.initial_state)
        # print inputs

        return inputs
    def forward(self, inputs, conv_hidden):

        self.inputs = inputs

        outputs, self.state = self.cell(self.inputs,(0,0))
        lstm_hidden = torch.cat((torch.ones(1),outputs),0)
        lstm_hidden = lstm_hidden.view(-1,cf.rnn_hidden_size)

        if config.topic_number > 0:
            z, r = array_ops.split(1, 2, linear([conv_hidden, lstm_hidden], \
                2 * cf.rnn_hidden_size, True, 1.0))
            z, r = torch.sigmoid(z), torch.sigmoid(r)
            c = torch.tanh(torch.mul(conv_hidden, self.gate_w) + torch.matmul((r * lstm_hidden), self.gate_u) + \
                self.gate_b)
            hidden = (1-z)*lstm_hidden + z*c
            self.tm_weights = torch.mean(z,1)
            self.tm_weights = self.tm_weights.view(-1,num_steps)
        else:
            hidden = lstm_hidden

        self.lm_logits=self.linear1(hidden)
        self.lm_logits=F.log_softmax(self.tm_logits)
        return lm_logits



    def sample(self, probs, temperature):
        if temperature == 0:
            return np.argmax(probs)

        probs = probs.astype(torch.FloatTensor)
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / math.fsum(np.exp(probs))
        return np.argmax(np.random.multinomial(1, probs, 1))

    def generate(self, conv_hidden, start_word_id, temperature, max_length, stop_word_id):
        x = [[start_word_id]]
        sent = [start_word_id]

        for _ in xrange(max_length):
            if type(conv_hidden) is np.ndarray:
            #if conv_hidden != None:
                probs, state = ([self.probs, self.state], \
                    {self.x: x, self.initial_state: self.state, self.conv_hidden: conv_hidden})
            else:
                probs, state = ([self.probs, self.state], \
                    {self.x: x, self.initial_state: self.state})
            sent.append(self.sample(probs[0], temperature))
            if sent[-1] == stop_word_id:
                break
            x = [[ sent[-1] ]]

        return sent

    def generate_on_topic(self, x, topic_id, start_word_id, temperature=1.0, max_length=30, stop_word_id=None):
        if topic_id != -1:
            count = -1
            count2 = -1
            self.x = x
    	    for i in self.topic_id:
        		count+=1
        		count2=-1
                	for j in i:
        			count2+=1
                		self.topic_input[count,count2]=self.topic_output_embedding[j]
            topic_emb = self.topic_emb.unsqueeze(self.topic_input,0)

        else:
            topic_emb = None

        return self.generate(topic_emb, start_word_id, temperature, max_length, stop_word_id)
