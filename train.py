import scipy.stats
import sys, random, os, time, math
import numpy as np
from gensim import matutils
import gensim.models as g
# import codecs
import constants as cf
from utilfuncs import *
from model import TopicModel as TM
from model import LanguageModel as LM
import torch

def init_embedding(model, idxvocab):
	word_emb = []
	for vi, v in enumerate(idxvocab):
		if v in model:
			word_emb.append(model[v])
		else:
			word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
	return np.array(word_emb)

def run_epoch(sents, docs, labels, tags, models, is_training):

    ####unsupervised topic and language model training####

    #generate the batches
	global tm_train

	tm_num_batches, lm_num_batches = int(math.ceil(float(len(sents[0]))/cf.batch_size)), int(math.ceil(float(len(sents[1]))/cf.batch_size))
	batch_ids = [ (item, 0) for item in range(tm_num_batches) ] + [ (item, 1) for item in range(lm_num_batches) ]
	seq_lens = (cf.tm_sent_len, cf.lm_sent_len)
	#shuffle batches and sentences
	random.shuffle(batch_ids)
	random.shuffle(sents[0])
	random.shuffle(sents[1])
	optimizer = torch.optim.Adam(tm_train.parameters(), lr=0.01)
	optimizer.zero_grad()

	lm_costs, tm_costs, lm_words, tm_words = 0.0, 0.0, 0.0, 0.0
	for bi, (b, model_id) in enumerate(batch_ids):
		if model_id==0:					#if language included comment this line

			#optimizer = torch.optim.Adam(tm_train.parameters(), lr=0.01)
			#optimizer.zero_grad()

			x, y, m, d, t = get_batch(sents[model_id], docs[model_id], tags, b, cf.doc_len, seq_lens[model_id], cf.tag_len, cf.batch_size, 0,(True if isinstance(models[model_id], LM) else False))
			doc_inputs = tm_train.pre(y,m,d,t)
			tm_logits = tm_train(doc_inputs)
		    lm_train.pre(x,m) 	
			y=torch.autograd.Variable(torch.from_numpy(np.asarray(y)))
			m=torch.autograd.Variable(torch.from_numpy(np.asarray(m)))

			loss=torch.nn.CrossEntropyLoss()
			tm_cost=loss(tm_logits,y.view(-1))
			print tm_cost
			#print tm_crossent.size()
			#tm_crossent_m = tm_crossent * m.view(-1)
			#tm_cost = torch.sum(tm_crossent_m) / batch_size
			#print tm_logits

			if is_training:
				v=torch.mul(tm_train.topic_output_embedding,tm_train.topic_output_embedding)
				vv=torch.sum(v)
				topicnorm = tm_train.topic_output_embedding / torch.sqrt(vv)
				print topicnorm.size()
				temp=torch.mm(topicnorm,torch.t(topicnorm))-torch.autograd.Variable(torch.eye(10))
				print temp.size()
				uniqueness = torch.max(torch.max(torch.mul(temp,temp),1)[0],0)[0]
				tm_cost += cf.alpha * uniqueness
			#print temp
			tm_costs += tm_cost * cf.batch_size #keep track of full batch loss (not per example batch loss)
			print tm_costs
			#tm_words += torch.autograd.Variable(np.sum(m))
			#lm_costs += lm_cost * cf.batch_size
			#lm_words += np.sum(m)

			tm_costs.backward(retain_graph=True)
			optimizer.step()


random.seed(1)
np.random.seed(1)
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
vocabxid = {}
idxvocab = []

wordvec = g.Word2Vec.load('./word2vec1/skipgram.bin')
word_embd_size = wordvec.vector_size

idxvocab, vocabxid, tm_ignore = gen_vocab(cf.dummy_symbols, cf.train_corpus, cf.stopwords, cf.vocab_minfreq, cf.vocab_maxfreq)
train_sents, train_docs, train_docids, train_stats = gen_data(vocabxid, cf.dummy_symbols, tm_ignore, cf.train_corpus)
valid_sents, valid_docs, valid_docids, valid_stats = gen_data(vocabxid, cf.dummy_symbols, tm_ignore, cf.valid_corpus)

num_classes = 0

tm_train = TM(is_training=True,  vocab_size=len(idxvocab), batch_size=cf.batch_size, num_steps=3, num_classes=num_classes, cf=cf)
tm_valid = TM(is_training=False, vocab_size=len(idxvocab), batch_size=cf.batch_size, num_steps=3, num_classes=num_classes, cf=cf)

tm_train.conv_word_embedding = torch.from_numpy(init_embedding(wordvec, idxvocab))


for i in range(cf.epoch_size):
	print "hello i am here2"
	run_epoch(train_sents, train_docs, None, None, (tm_train, None), True)
	print "hello i am here3"
	curr_ppl = run_epoch(valid_sents, valid_docs, None, None, (tm_valid, None), False)

if cf.topic_number > 0:
	print "\nTopics\n======"
	topics, entropy = tm_train.get_topics(topn=5)
	for ti, t in enumerate(topics):
		print "Topic", ti, "[", ("%.2f" % entropy[ti]), "] :", " ".join([ idxvocab[item] for item in t ])
