# A-Survey-on-a-Music-Recommender-System

The aim of this project is to implement models which provide personalized song recommendations to different users taking into account their previous preferences. 

For its purposes we use the ”ydata-ymusic-user-song-rating-meta-v1_0” dataset from Yahoo. This
dataset is provided as part of the Yahoo! Research Alliance Webscope program for research purposes and it is available
in the link https://webscope.sandbox.yahoo.com/catalog.php?datatype=r after request. It supplies us with information
about the Yahoo! Music community’s preferences for various songs, as it contains over 717 million ratings of 136
thousand songs given by 2 million users of Yahoo! Music services. The data were collected between 2002 and 2006.
Additional information is given in the dataset concerning the artist,the album, and the genre for each song. These
attributes are represented by randomly assigned numeric ids.

The notebook contains the description and the implementation of our models. At the beginning we do some data analysis, both user-oriented and song-oriented in
order to better understand the distribution and the nature of the available data. Afterwards, we select
the subset of the data which we use for our model implementations and opt for the most suitable
metrics for the evaluation of the different algorithms. Then we experiment with the models which the
surprise Python library provides and compare them. Namely these models are:

*  Normal predictor
*  Baseline Estimate
*  Basic k-NN
*  k-NN with means
*  k-NN with z-score
*  k-NN with baseline rating
*  Matrix factorization with SVD
*  Matrix factorization with SVD++
*  Non-negative Matrix Factorization (NMF)
*  Slope One
*  Co-clustering algorithm

Additionally we attempt to deploy some neural net implementations and assess their performance. Namely these implementations are:
*  AutoRec
*  Neural Collaborative Filtering (NCF)
*  Light Graph Convolution Network (LightGCN)
*  Restricted Boltzmann Machine (RBM)

The src folder includes a user interface implementation of the above models using Tkinter.
