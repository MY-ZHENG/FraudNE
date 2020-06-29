import numpy as np
import tensorflow as tf
import time
import copy

class ADNE:
	def __init__(self,config):
		self.is_variables_init = False
		self.config=config
		
		tf_config=tf.ConfigProto()
		tf_config.gpu_options.allow_growth = True

		self.sess=tf.Session(config = tf_config)

		self.layers = len(config.struct_u)
		self.struct_u=config.struct_u
		self.struct_t=config.struct_t
		self.sparse_dot=config.sparse_dot
		
		self.Wu={}
		self.bu={}
		self.Wt={}
		self.bt={}

		struct_u =self.struct_u
		struct_t = self.struct_t
		for i in range(self.layers-1):
			name="user_encoder" + str(i)
			self.Wu[name]=tf.Variable(tf.random_normal([struct_u[i],struct_u[i+1]]),name=name)
			self.bu[name]=tf.Variable(tf.zeros(struct_u[i+1]),name=name)
		for i in range(self.layers-1):
			name="item_encoder" + str(i)
			self.Wt[name]=tf.Variable(tf.random_normal([struct_t[i],struct_t[i+1]]),name=name)
			self.bt[name]=tf.Variable(tf.zeros(struct_t[i+1]),name=name)
		struct_u.reverse()
		struct_t.reverse()
		for i in range(self.layers -1):
			name="user_decoder" + str(i)
			self.Wu[name]=tf.Variable(tf.random_normal([struct_u[i],struct_u[i+1]]),name=name)
			self.bu[name]=tf.Variable(tf.zeros(struct_u[i+1]),name=name)
		for i in range(self.layers -1):
			name="item_decoder" + str(i)
			self.Wt[name]=tf.Variable(tf.random_normal([struct_t[i],struct_t[i+1]]),name=name)
			self.bt[name]=tf.Variable(tf.zeros(struct_t[i+1]),name=name)
		struct_u.reverse()
		struct_t.reverse()

		self.adjacent_matrix = tf.placeholder("float",[None,None])
		#self.links=tf.placeholder("float",[None,3])
		self.ngmatrix=tf.placeholder("float",[None,None])

		self.Xu_sp_indices = tf.placeholder(tf.int64)
		self.Xu_sp_ids_val = tf.placeholder(tf.float32)
		self.Xu_sp_shape = tf.placeholder(tf.int64)
		self.Xu_sp = tf. SparseTensor(self.Xu_sp_indices,self.Xu_sp_ids_val,self.Xu_sp_shape)
		self.Xt_sp_indices = tf.placeholder(tf.int64)
		self.Xt_sp_ids_val = tf.placeholder(tf.float32)
		self.Xt_sp_shape = tf.placeholder(tf.int64)
		self.Xt_sp = tf. SparseTensor(self.Xt_sp_indices,self.Xt_sp_ids_val,self.Xt_sp_shape)
		self.X_u=tf.placeholder("float",[None,config.struct_u[0]])
		self.X_t=tf.placeholder("float",[None,config.struct_t[0]])
		
		self.__make_compute_graph()
		self.loss =self.__make_loss(config)
		#self.optimizer=tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)
		self.optimizer=tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)


	def __make_compute_graph(self):
		def user_encoder(X_u):
			for i in range(self.layers-1):
				name="user_encoder"+str(i)
				print name
				X_u=tf.nn.sigmoid(tf.matmul(X_u,self.Wu[name]) + self.bu[name])
			return X_u
		def item_encoder(X_t):
			for i in range(self.layers-1):
				name="item_encoder"+str(i)
				print name
				X_t=tf.nn.sigmoid(tf.matmul(X_t,self.Wt[name]) + self.bt[name])
			return X_t

		def user_encoder_sp(X_u):
			for i in range(self.layers-1):
				name="user_encoder"+ str(i)
				if i==0:
					X_u=tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(X,self.Wu[name])+self.bu[name])
				else:
					X_u=tf.nn.sigmoid(tf.matmul(X,self.Wu[name])+ self.bu[name])
			return X_u
		def item_encoder_sp(X_t):
			for i in range(self.layers-1):
				name="item_encoder"+ str(i)
				if i==0:
					X_t=tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(X_t,self.Wt[name])+self.bt[name])
				else:
					X_t=tf.nn.sigmoid(tf.matmul(X_t,self.Wt[name])+ self.bt[name])
			return X_t

		def user_decoder(X_u):
			for i in range(self.layers-1):
				name="user_decoder"+ str(i)
				X_u=tf.nn.sigmoid(tf.matmul(X_u,self.Wu[name])+self.bu[name])
			return X_u
		def item_decoder(X_t):
			for i in range(self.layers-1):
				name="item_decoder"+ str(i)
				X_t=tf.nn.sigmoid(tf.matmul(X_t,self.Wt[name])+self.bt[name])
			return X_t

		if self.sparse_dot:
			self.Hu=user_encoder_sp(self.Xu_sp)
			self.Ht=item_encoder_sp(self.Xt_sp)
		else:
			print 'do it'
			self.Hu=user_encoder(self.X_u)
			self.Ht=item_encoder(self.X_t)
		self.H=tf.concat([self.Hu,self.Ht],0)
		self.Xu_reconstruct=user_decoder(self.Hu)
		self.Xt_reconstruct= item_decoder(self.Ht)

	def __make_loss(self,config):
		def get_loss_link_sample(Y1,Y2):
			return tf.reduce_sum(tf.pow(Y1-Y2,2))
		def get_link_loss(H,adj_mini_batch):    
			D=tf.diag(tf.reduce_sum(adj_mini_batch,1))
			L=D-adj_mini_batch
			return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))               #need to change,maybe ,lapalican loss

		def get_san_loss(X,newX,beta):
			B = X*(beta-1)+1
			return tf.reduce_sum(tf.pow((newX-X)*B,2))
		def get_reg_loss(weight_u,weight_t,biases_u,biases_t):
			ret=tf.add_n([tf.nn.l2_loss(wu) for wu in weight_u.itervalues()])+tf.add_n([tf.nn.l2_loss(wt) for wt in weight_t.itervalues()])
			ret=ret + tf.add_n([tf.nn.l2_loss(bu) for bu in biases_u.itervalues()]) + tf.add_n([tf.nn.l2_loss(bt) for bt in biases_t.itervalues()])
			return ret

        #def get_nagtive_loss(lin, X,Y ):
        	
        	#for i in range(len(lin)):
        		#tf.pow(X[lin[i,0]]-Y[lin[i,1]],2)
        	#sum=tf.reduce_sum(tf.pow(X[lin[i,0]]-Y[lin[i,1]],2))
        	#return sum


		self.loss_san=get_san_loss(self.X_u,self.Xu_reconstruct,config.beta)+get_san_loss(self.X_t,self.Xt_reconstruct,config.beta)
		self.loss_link=get_link_loss(self.H,self.adjacent_matrix)
		self.nagtive_link=get_link_loss(self.H,self.ngmatrix)
		self.loss_xxx=tf.reduce_sum(tf.pow((self.Xu_reconstruct),2))+tf.reduce_sum(tf.pow((self.Xt_reconstruct),2))
		self.loss_reg = get_reg_loss(self.Wu ,self.bu,self.Wt,self.bt)
		#self.sample=get_loss_link_sample(self.Huple,self.Htple)


		return config.gamma*self.loss_link+ config.alpha * self.loss_san +config.reg*self.loss_reg-config.ng *self.nagtive_link
				
		#return config.gamma*self.loss_link + config.alpha * self.loss_san+ self.loss_reg
		#return  self.loss_san
		#return  config.alpha * self.loss_san + self.loss_xxx

		
	def save_model(self,path):
		saver=tf.train.Saver()
		saver.save(self.sess,path)
	def restore_model(self,path):
		saver=tf.train.Saver()
		saver.save(self.sess,path)
		self.is_Init=True

	def do_variables_init(self, data, DBN_init):
		def assign(a,b):
			op=a.assign(b)
			self.sess.run(op)
		init= tf.global_variables_initializer()
		self.sess.run(init)
		if DBN_init:
			shape_u=self.struct_u
			#shape_t=self.struct_t
				
			myRBMs =[]
			for i in range (len(shape_u)-1):
				myRBM=rbm([shape_u[i],shape_u[i+1]],{"batch_size":self.config.dbn_batch_size_us,"learning_rate":self.config.dbn_learning_rate})
				myRBMs.append(myRBM)
				for epoch in range(self.config.dbn_epochs):
					error = 0
					for batch in range(0,data.num_usr,self.config.dbn_batch_size_us):
						mini_batch_us=data.sample(self.config.dbn_batch_size_us).X_us       #
						for k in range(len(myRBMs)-1):
							mini_batch_us=myRBMs[k].getH(mini_batch_us)
						error += myRBM.fit(mini_batch_us)
					print("rbm epochs:" ,epoch , "error: ",error)
				Wu,buv,buh=myRBM.getWb()
				name="user_encoder" + str(i)
				assign(self.Wu[name],Wu)
				assign(self.bu[name],buh)
				name="user_decoder"+str(self.layers-i-2)       # why is 2
				assign(self.Wu[name],Wu.transpose())
				assign(self.bu[name],buv)	
			'''
			myRBMts =[]
			for i in range (len(shape_t)-1):
				myRBMt=rbm([shape_t[i],shape_t[i+1]],{"batch_size":self.config.dbn_batch_size_it,"learning_rate":self.config.dbn_learning_rate})
				myRBMts.append(myRBMt)
				for epoch in range(self.config.dbn_epochs):
					error = 0	
					for batch in range(0,data.num_item,self.config.dbn_batch_size_it):
						mini_batch_it=data.sample(self.config.dbn_batch_size_it).X_it
						for k in range(len(myRBMts)-1):
							mini_batch_it=myRBMts[k].getH(mini_batch_it)
						error += myRBMt.fit(mini_batch_it)
					print("rbm epochs:", epoch,"error: ",error)

				Wt,btv,bth=myRBMt.getWb()
				name="item_encoder" + str(i)
				assign(self.Wt[name],Wt)
				assign(self.bt[name],bth)
				name="item_decoder"+str(self.layers-i-2)       # why is 2
				assign(self.Wt[name],Wt.transpose())
				assign(self.bt[name],btv)
			'''    

	def __get_feed_dict(self,data):
		#X_u=data.X_us
		#X_t=data.X_it
		if self.sparse_dot:
			Xu_ind=np.vstack(np.where(X_u).astype(np.int64)).T
			Xu_shape = np.array(X_u.shape).astype(np.int64)
			Xu_val = Xu[np.where(X_u)]
			Xt_ind=np.vstack(np.where(X_t).astype(np.int64)).T
			Xt_shape = np.array(X_t.shape).astype(np.int64)
			Xt_val = Xt[np.where(X_t)]
			return{self.X_u:data.X_us ,self.Xu_sp_indices:Xu_ind, self.Xu_sp_shape:Xu_shape,self.Xu_sp_ids_val:Xu_val,\
			self.X_t: data.X_it,self.Xt_sp_indices:Xt_ind, self.Xt_sp_shape:Xt_shape,self.Xt_sp_ids_val:Xt_val,\
		    self.adjacent_matrix:data.adjacent_matrix}
		else:
			return {self.X_u:data.X_us,self.X_t:data.X_it, self.adjacent_matrix:data.adjacent_matrix,self.ngmatrix:data.ngmatrix}
		
	
	def fit(self,data):
		#time0 = time.clock()
		feed_dict = self.__get_feed_dict(data)
		#time1 = time.clock()
		ret,ret_link,ret_san,ret_reg,ret_neg, _ =self.sess.run((self.loss,self.loss_link, self.loss_san, self.loss_reg, self.nagtive_link,self.optimizer),feed_dict = feed_dict)
		#time2 = time.clock()
		#print ('Time of feed is {}, and time of run is {}'.format(time1-time0, time2-time1))
		return ret,ret_link,ret_san,ret_reg,ret_neg
	def get_loss(self,data):
		feed_dict=self.__get_feed_dict(data)
		return self.sess.run(self.loss, feed_dict = feed_dict)

	def get_embedding(self,data):
		return self.sess.run(self.H , feed_dict = self.__get_feed_dict(data))
	def get_Xu_reconstruct(self,data):

		return self.sess.run(self.Xu_reconstruct , feed_dict = self.__get_feed_dict(data))

	def get_Xt_reconstruct(self,data):

		return self.sess.run(self.Xt_reconstruct , feed_dict = self.__get_feed_dict(data))
	def get_Wu(self):
		return self.sess.run(self.Wu)
	def get_Wt(self):
		return self.sess.run(self.Wt)
	def get_Bu(self):
		return self.sess.run(self.bu)
	def get_Bt(self):
		return self.sess.run(self.bt)	
	def close(self):
		self.sess.close()

			
