import surprise
from collections import defaultdict
from surprise import Reader,Dataset
from PrepareData import GetData
from time import time

class Baseline_Models:
    def __init__(self,models,train_file_path,test_file_path):

        maps = {
            1: surprise.prediction_algorithms.random_pred.NormalPredictor(),
            2: surprise.prediction_algorithms.baseline_only.BaselineOnly(verbose=False),
            3: surprise.prediction_algorithms.knns.KNNBasic(verbose=False,sim_options={"user_based":False}),
            4: surprise.prediction_algorithms.knns.KNNWithMeans(verbose=False,sim_options={"user_based":False}),
            5: surprise.prediction_algorithms.knns.KNNWithZScore(verbose=False,sim_options={"user_based":False}),
            6: surprise.prediction_algorithms.knns.KNNBaseline(verbose=False,sim_options={"user_based":False}),
            7: surprise.prediction_algorithms.matrix_factorization.SVD(verbose=False),
            8: surprise.prediction_algorithms.matrix_factorization.SVDpp(verbose=False),
            9: surprise.prediction_algorithms.matrix_factorization.NMF(verbose=False),
            10: surprise.prediction_algorithms.slope_one.SlopeOne(),
            11: surprise.prediction_algorithms.co_clustering.CoClustering(verbose=False)
        }
        
        self.models = [maps[model] for model in models]

        self.train_file_path = train_file_path
        train_reader = Reader(line_format="user item rating", sep="\t",rating_scale=(1, 5))
        train_data = Dataset.load_from_file(train_file_path, reader=train_reader)
        self.train_data = train_data.build_full_trainset()

        reader = Reader(line_format='user item rating', sep='\t',rating_scale=(1, 5))
        test_data = Dataset.load_from_file(test_file_path,reader=reader)
        test_data = test_data.build_full_trainset()
        self.test_data = test_data.build_testset()
        self.test_file_path = test_file_path

    def precision_recall_at_k(self,predictions, k=10, threshold=3.5):
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))
        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        
        precision = (sum(prec for prec in precisions.values()) / len(precisions))
        recall = (sum(rec for rec in recalls.values()) / len(recalls))
        return precision, recall
    
    def train_models(self):
        for model in (self.models):
            tic = time()
            model.fit(self.train_data)
            toc = time()
            yield self.calc_statistics(model,toc-tic)
            
    
    def calc_statistics(self,model,time):
        prediction = model.test(self.test_data)
        rmse = surprise.accuracy.rmse(prediction,verbose=False)
        mse = surprise.accuracy.mse(prediction,verbose=False)
        mae = surprise.accuracy.mae(prediction,verbose=False)
        pr,re = self.precision_recall_at_k(prediction)
        fcp = surprise.accuracy.fcp(prediction,verbose=False)
        return (rmse,mse,mae,pr,re,fcp,time)
    
    def make_recommendations(self,user,num_of_recommendations,model_idx):
        recommendation_lst = [[0,0]] * num_of_recommendations
        train_f = open(self.train_file_path,"r")
        lines = train_f.readlines()
        for line in lines:
            pred = self.models[model_idx].predict(str(user), line.split()[1], r_ui=1, verbose=False)
            if pred.est > recommendation_lst[num_of_recommendations-1][1] and line.split()[1] not in [i[0] for i in recommendation_lst]:
                recommendation_lst[num_of_recommendations-1] = [line.split()[1],pred.est]
                recommendation_lst.sort(key = lambda x: x[1], reverse=True)

        recommendation_string = "\n" + "*"*10 + " TOP %d RECOMMENDATIONS FOR USER %s "%(num_of_recommendations,str(user)) + "*"*10 + "\n"
        for i in range(num_of_recommendations):
            recommendation_string += (str(i+1) + ") Song: " + str(recommendation_lst[i][0]) + ", Predicted rating: " +  str(recommendation_lst[i][1])) + "\n"

        return recommendation_string


