# FraudNE
This repository provides an implementation of the method proposed in "[FraudNE: a Joint Embedding Approach for Fraud Detection](https://www.researchgate.net/profile/Chuan_Zhou5/publication/327164545_FraudNE_a_Joint_Embedding_Approach_for_Fraud_Detection/links/5b7df361299bf1d5a71e43dd/FraudNE-a-Joint-Embedding-Approach-for-Fraud-Detection.pdf)", Mengyu Zheng, Chuan Zhou, Jia Wu, Shirui Pan, Jinqiao Shi and Li Guo, IJCNN 2018

### Overview
-   `input/`    contains an example graphs 'zomato.edgelist.400';
-   `output/`  is the directory to store the learned node embeddings;
-  `src/` contains the implementation of the proposed FraudNE method.

### Requirement

The implementation is tested under Pyrhon 2.7, with the folowing packages installed:
 - `networkx==1.11`
 - `numpy==1.11.2`
 - `tensorflow==1.5`


### Input
The code takes a bipartite input graph composed users and items. Every row indicates an edge between two nodes, such like:

              user_node1_id_int item_node2_id_int weight_int
The file does not contain a header. Nodes can be indexed starting with any non-negative number.

The graph is assumed to be directed and weighted by default.

### Cite
If you find *FraudNE* useful for your research, please consider citing the following paper:

                    @inproceedings{zheng2018fraudne,
                      title={Fraudne: a joint embedding approach for fraud detection}, 
                      author={Zheng, Mengyu and Zhou, Chuan and Wu, Jia and Pan, Shirui and Shi, Jinqiao and Guo, Li},  
                      booktitle={2018 International Joint Conference on Neural Networks (IJCNN)}, 
                      pages={1--8}, 
                      year={2018},
                      organization={IEEE}}
