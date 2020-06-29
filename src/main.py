import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from config import Config
from nice_graph import Nice_graph
from model.adne import ADNE
import scipy.io  as sio
import numpy as np
import copy
from utils import *
if __name__=="__main__":
	config=Config()
    
	graph_data=Nice_graph(config.file_path,config.ng_sample_ratio)
	config.struct_u[0]=graph_data.num_item
	config.struct_t[0]=graph_data.num_usr

	model=ADNE(config)
	model.do_variables_init(graph_data,config.DBN_init)
	print model.H

	epochs=0
	batch_n=0

	origin_data=copy.deepcopy(graph_data)
	fout=open(config.embedding_filename + "-log.txt","w")
	

	graph_data.neg_m()
	loss = 0.0
	loss_link = 0.0
	loss_san = 0.0
	loss_reg = 0.0
	nagtive_link = 0.0
	while(True):
		
		#print 'really done 2'
		time0 = time.clock()
		mini_batch = graph_data.sample(config.batch_size,config.batch_size_t)
		time1 = time.clock()
		#print ('time of sample is {}s.'.format(time1-time0))
		#print mini_batch.us_it_matrix
		#print mini_batch.adjacent_matrix.toarray()
		#like = model.get_Xt_reconstruct(mini_batch)
		#print like
		#like = model.get_Xu_reconstruct(mini_batch)
                #print like

		_loss, _loss_link, _loss_san, _loss_reg, _nagtive_link = model.fit(mini_batch)
		loss += _loss
		loss_link += _loss_link
		loss_san += _loss_san
		loss_reg += _loss_reg
		nagtive_link += _nagtive_link
		#print loss
		batch_n += 1
		#print('Epoch: %d, batch : %d, loss: %3f' % (epochs,batch_n,loss))
		if graph_data.is_epoch_end:
			epochs  += 1
			print('Epoch : %d, loss : %.2f, loss_link : %.2f, loss_san : %.2f, loss_reg : %.2f, nagtive_link : %.2f' % \
				(epochs,loss/batch_n, loss_link/batch_n, loss_san/batch_n, loss_reg/batch_n, nagtive_link/batch_n))
			batch_n = 0
			if epochs % config.display ==0:
				embedding=None
				while(True):
					mini_batch= graph_data.sample(config.batch_size,config.batch_size_t, do_shuffle = False)
					loss +=model.get_loss(mini_batch)
					if embedding is None:
						embedding = model.get_embedding(mini_batch)
					else:
						embedding = np.vstack((embedding,model.get_embedding(mini_batch)))

					if graph_data.is_epoch_end:
						break
				sio.savemat(config.embedding_filename +'-'+str(epochs)+'embedding.mat',{'embedding':embedding})



			if epochs > config.epochs_limit:
				print "exceed epochs linit terminating"
				break
			last_loss = loss

			graph_data.neg_m()
			loss = 0.0
			loss_link = 0.0
			loss_san = 0.0
			loss_reg = 0.0
			nagtive_link = 0.0

fout.close()

    
