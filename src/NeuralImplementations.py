from recommenders.models.ncf.ncf_singlenode import NCF
from  recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.models.deeprec.deeprec_utils import HParams
from recommenders.evaluation.python_evaluation import rmse,mae, precision_at_k, recall_at_k
from PrepareData import GetData
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.datasets.python_splitters import python_stratified_split
import pandas as pd
from recommenders.utils.timer import Timer
from time import time
import numpy as np
from PrepareData import GetData
import torch.nn as nn
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



class NCF_Implementation:
    def __init__(self,train_file_path,test_file_path,max_users,max_items):
        self.train_df = pd.read_csv(train_file_path,delimiter="\t",engine="python",header=None)
        self.train_df.columns=['user_id','song_id','rating']
        self.train_df[['user_id','song_id','rating']].to_csv(train_file_path.split(".")[0]+".csv")
        self.train_data_filepath = train_file_path.split(".")[0]+".csv"
        self.MAX_USERS = max_users
        self.MAX_ITEMS = max_items
        self.test_df = pd.read_csv(test_file_path,delimiter="\t",engine="python",header=None)
        self.test_df.columns=['user_id','song_id','rating']
        self.test_df[['user_id','song_id','rating']].to_csv(test_file_path.split(".")[0]+".csv")
        self.test_data_filepath = test_file_path.split(".")[0]+".csv"

        self.data = NCFDataset(train_file=self.train_data_filepath, test_file=self.test_data_filepath, 
        col_user='user_id', col_item='song_id', col_rating='rating', binary=True, seed=None)


        self.model = NCF(
            n_users=self.data.n_users, 
            n_items=self.data.n_items,
            model_type="NeuMF",
            n_factors=4,
            layer_sizes=[16,8,4],
            n_epochs=10,
            batch_size=50,
            learning_rate=1e-3,
            verbose=1,
            seed=None
        )
    def train(self):
        tic = time()
        self.model.fit(self.data)
        toc = time()
        self.time_passed = toc - tic
    
    def evaluate(self):
        predictions = []
        for (_, row) in self.test_df.iterrows():
            try:
                predictions.append([row.user_id, row.song_id, row.rating,self.model.predict(int(row.user_id), int(row.song_id))])
            except:
                pass
        predictions = pd.DataFrame(predictions, columns=['user_id', 'song_id', 'rating','prediction'])
        ratings_map = [
            (predictions['prediction'] <= 0.05),
            (predictions['prediction'] > 0.05) & (predictions['prediction'] <= 0.2),
            (predictions['prediction'] > 0.2) & (predictions['prediction'] <= 0.45),
            (predictions['prediction'] > 0.45) & (predictions['prediction'] <= 0.7),
            (predictions['prediction'] > 0.7)
        ]
        predictions['prediction'] = np.select(ratings_map, range(1,6))
        RMSE = rmse(predictions[['user_id','song_id','rating']], predictions[['user_id','song_id','prediction']], col_user='user_id', col_item='song_id')
        pr = precision_at_k(predictions[['user_id','song_id','rating']], predictions[['user_id','song_id','prediction']],\
            col_user='user_id', col_item='song_id',k=20,threshold=3.5)
        re = recall_at_k(predictions[['user_id','song_id','rating']], predictions[['user_id','song_id','prediction']],\
            col_user='user_id', col_item='song_id',k=20,threshold=3.5)
        return RMSE,pr,re,self.time_passed

    def make_recommendations(self,user,recommendation_num):
        l_predictions = []
        l_item_id = []
        l_user = []
        for i in range(self.MAX_ITEMS):
            try:
                l_predictions.append(self.model.predict(user,i))
                l_item_id.append(i)
                l_user.append(user)
            except:
                pass
        
        df = pd.DataFrame()
        df["user"] = l_user
        df["item"] = l_item_id
        df["prediction"] = l_predictions
        ratings_map = [
            (df['prediction'] <= 0.05),
            (df['prediction'] > 0.05) & (df['prediction'] <= 0.2),
            (df['prediction'] > 0.2) & (df['prediction'] <= 0.45),
            (df['prediction'] > 0.45) & (df['prediction'] <= 0.7),
            (df['prediction'] > 0.7)
        ]
        df['prediction'] = np.select(ratings_map, range(1,6))
        df =  df.sort_values("prediction",ascending=False)
        recommendations_str = "\n" + "*"*10 + " TOP %d RECOMMENDATIONS FOR USER %s "%(recommendation_num,str(user)) + "*"*10 + "\n"
        i = 1
        for _,item in df.iterrows():
            recommendations_str += (str(i) + ") Song: " + str(int(item['item'])) + ", Predicted rating: " +  str(item['prediction'])) + "\n"
            i += 1
            if i > recommendation_num:
                break
        
        return recommendations_str
        
        
class LightGCN_Implementation:
    def __init__(self,MAX_USERS) -> None:
        train_df = pd.read_csv("pipeline/train.txt",delimiter="\t",header=None,engine="python")
        train_df.columns = ["userID","itemID","rating"]

        test_df = pd.read_csv("pipeline/test.txt",delimiter="\t",header=None,engine="python")
        test_df.columns = ["userID","itemID","rating"]

        self.df = pd.concat([train_df,test_df],axis=0,ignore_index=True)
        self.TOP_K = 100
        self.MAX_USERS = MAX_USERS
        EPOCHS = 10
        BATCH_SIZE = 10
        self.train, self.test = python_stratified_split(self.df, ratio=0.75)
        SEED = DEFAULT_SEED
        self.data = ImplicitCF(train=self.train, test=self.test, seed=SEED)

        hparams = prepare_hparams("lightgcn.yaml",
                          n_layers=3,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          learning_rate=0.005,
                          eval_epoch=5,
                          top_k=self.TOP_K,
                         )
        self.model = LightGCN(hparams, self.data, seed=SEED)

    def trainGCN(self):
        with Timer() as train_time:
            self.model.fit()
        return train_time.interval

    def evaluate(self):
        topk_scores = self.model.recommend_k_items(self.test, top_k=self.TOP_K, remove_seen=True)
        precision = precision_at_k(self.test, topk_scores, k=self.TOP_K)
        recall = recall_at_k(self.test, topk_scores, k=self.TOP_K)
        RMSE = rmse(self.test, topk_scores)
        return (RMSE,precision,recall)
    
    def make_recommendations(self,user,number_of_recommendations):
        recommendations_str = "\n" + "*"*10 + " TOP %d RECOMMENDATIONS FOR USER %s "%(number_of_recommendations,str(user)) + "*"*10 + "\n"
        predictions = self.model.recommend_k_items(self.df,top_k=number_of_recommendations,remove_seen=True)
        u_df = predictions[predictions['userID'] == user]
        i = 0
        for _,item in u_df.iterrows():
            if(item['prediction'] > 5): item['prediction'] = 5
            recommendations_str += (str(i+1) + ") Song: " + str(int(item['itemID'])) + ", Predicted rating: " +  str(item['prediction'])) + "\n"
            i += 1
        return recommendations_str
