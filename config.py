class Config(object):
	def __init__(self):
		
		self.file_path="input/zomato.edgelist.400"
		#self.tdfile_path="ratingsts.dict.inject.400"
		#self.ratefile_path='ratingsrate.dict.inject.400'
		
		self.embedding_filename = "output/zomato.inject_400"
		self.batch_size=5336
		self.batch_size_t=1012
		#self.num_sample_it=4
		self.num_block=4
		
		self.struct_u=[None,600,100]
		self.struct_t=[None,500,100]

		self.gamma=10
		self.ng=10
		self.reg=0.01
		self.alpha=1
		self.beta=1

		self.epochs_limit=5000
		self.learning_rate=0.0008

		self.DBN_init=False
		self.dbn_epochs = 20
		self.dbn_batch_size_us=3
		self.dbn_batch_size_it=3
		self.dbn_learning_rate=0.1

		self.ng_sample_ratio=0.5


		self.display =50
		self.sparse_dot=False
		#self.howlen=1000
		self.howlen=195456




