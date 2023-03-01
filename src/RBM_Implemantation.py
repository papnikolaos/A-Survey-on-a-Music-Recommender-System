import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from recommenders.models.rbm.rbm import RBM
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.evaluation.python_evaluation import rmse,mae, precision_at_k, recall_at_k
from time import time

class RBM_Implementation(object):
    def __init__(self,max_users):
        train_df = pd.read_csv("pipeline/train.txt",delimiter="\t",header=None,engine="python")
        train_df.columns = ["user_id","song_id","rating"]
        self.MAX_USERS = max_users
        test_df = pd.read_csv("pipeline/test.txt",delimiter="\t",header=None,engine="python")
        test_df.columns = ["user_id","song_id","rating"]

        data = pd.concat([train_df,test_df],axis=0,ignore_index=True)
        header = {
                "col_user": "user_id",
                "col_item": "song_id",
                "col_rating": "rating",
            }
        self.am = AffinityMatrix(df = data, **header)
        self.X, _, _ = self.am.gen_affinity_matrix()
        self.Xtr, self.Xtst = numpy_stratified_split(self.X)
        self.model = RBM(
            possible_ratings=np.setdiff1d(np.unique(self.Xtr), np.array([0])),
            visible_units=self.Xtr.shape[1],
            hidden_units=600,
            training_epoch=100,
            minibatch_size=60,
            keep_prob=0.9,
            with_metrics=True
        )
    def train(self):
        tic = time()
        self.model.fit(self.Xtr)
        toc = time()
        return float(toc-tic)

    def evaluate(self):
        K = 10
        top_k =  self.model.recommend_k_items(self.Xtst)
        top_k_df = self.am.map_back_sparse(top_k, kind = 'prediction')
        test_df = self.am.map_back_sparse(self.Xtst, kind = 'ratings')
        precision = precision_at_k(test_df, top_k_df, col_user="user_id", col_item="song_id", 
                               col_rating="rating", col_prediction="prediction", 
                               relevancy_method="top_k", k= K)

        recall = recall_at_k(test_df, top_k_df, col_user="user_id", col_item="song_id", 
                            col_rating="rating", col_prediction="prediction", 
                            relevancy_method="top_k", k= K)
        RMSE = rmse(test_df, top_k_df, col_user="user_id", col_item="song_id", 
                            col_rating="rating", col_prediction="prediction")
        return (RMSE,precision,recall)

    def make_recommendations(self,user,recommendation_num):
        top_k =  self.model.recommend_k_items(self.X,top_k=recommendation_num)
        df = self.am.map_back_sparse(top_k, kind = 'prediction')
        recommendations_str = "\n" + "*"*10 + " TOP %d RECOMMENDATIONS FOR USER %s "%(recommendation_num,str(user)) + "*"*10 + "\n"
        u_df = df[df['user_id'] == user].sort_values('prediction',ascending=False)
        i = 0
        for _,item in u_df.iterrows():
            recommendations_str += (str(i+1) + ") Song: " + str(int(item['song_id'])) + ", Predicted rating: " +  str(item['prediction'])) + "\n"
            i += 1
        return recommendations_str
        
