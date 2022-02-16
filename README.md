# Machine learning predictor of tennis matches

This is a predictor of results of tennis macthes based on Machine Learning. The algorithm is splited into 3 notebooks. First one handles the [data preprocessing](https://github.com/happypio/ML_tennis_predictor/blob/master/data_preprocessing.ipynb). In the second you can see [analysis of data](https://github.com/happypio/ML_tennis_predictor/blob/master/analysis.ipynb) and in the third one developing of [machine learning model](https://github.com/happypio/ML_tennis_predictor/blob/master/predictor.ipynb).

Data was collected from Jeff Sackman's repository. I have extracted 259 features from it, such as Player's height, age,  rank. There are also some sophisticated features such as number of wins in last semester/year on given surface, number of service points in last semester/year in different types of tournaments and so on. For more, you can check the [data preprocessing](https://github.com/happypio/ML_tennis_predictor/blob/master/data_preprocessing.ipynb) notebook.

This is how features are correlated:

![features_corr](https://github.com/happypio/ML_tennis_predictor/blob/master/feature_importance.png)

Features importance was checked:

![features_imp](https://github.com/happypio/ML_tennis_predictor/blob/master/feature_importance_2.png)

You can see that the distribution of data (in PCA with 2 components) is not ideal. That's why problem of predicting matches results is complicated.

![pca_analysis](https://github.com/happypio/ML_tennis_predictor/blob/master/pca_analysis.png)


Quite a lot models was trained to these features. I have performed special cross validation on them, because of chronological data. After this every hyperparameter was tuned. This is correlation of tuned models.

![model correlation](https://github.com/happypio/ML_tennis_predictor/blob/master/model_correlation.png)

It is important to tune also number of components in PCA. PCA can change (because of data rotation) accuracy of tree-based classifiers. Here is plot of how number of components affects accuracy of Logistic Regression:

![logistic_reg](https://github.com/happypio/ML_tennis_predictor/blob/master/log_reg_pca.png)

I stacked these models and trained meta classifier on their predictions of probabilities of given class. This is final result:

![fin_res](https://github.com/happypio/ML_tennis_predictor/blob/master/accuracy.png)

I fixed some threshold of how model is certain about given match (prob: 0.55,0.66,...) and checked accuracy on these threshold. It can have accuracy of 79% by predicing matches, where it can tell with 75% probability who will win. It would prune about 0.7 of data.

![prob_odds](https://github.com/happypio/ML_tennis_predictor/blob/master/prob_odds.png)
