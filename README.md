# StockMarketPredictor

This program uses a random forest classifier from the sklearn python library to predict future stock price increases. For use, the file should be run directly in a terminal.

When running the file, enter in the symbol of the stock you want the model to train on, query new data (or load saved data if already queried), and then choose whether to test the model performance or predict a future price increase.

To query stock market data, the program uses the Alpha Vantage API. You must enter in your Alpha Vantage API key to use the program. The api key can be obtained for free and can be used to do your own model testing, but if you want to get real time data and do real time predictions, you need the paid premium alpha vantage subscription (or modify the program and use your own favorite stock market API).

The model training set is currently set-up as 94 feature columns and 1 classification column. The classification is either 1 (price increases above a particular percent change threshold without decreasing by the same threshold within an hour) or the classification is 0 (price does not increase beyond that threshold or instead decreases by that threshold first).

The model uses 5 minute intraday closing price data over the last 4 years, corresponding to about 77,000 samples. The first 78 features used in the model are the time series closing price data for that sample over the last day (6.5 trading hours). The next 3 features are the simple moving averages over the last 50 minutes (10 periods), 100 minutes (20 periods), and 250 minutes (50 periods) respectively. The next feature is the relative strength index over the last 50 minutes (10 periods). The next 12 features are the time series volume data over the last hour.

Because we want the model to learn how past price fluctuations determine the probability of future price increases, we want the price and volume data in each sample to be a percent change from the previous 5 minute time stamp rather than the absolute value. So, except for the most recent price and volume data points in each sample, all price and volume data is converted into a percent change.

For the ADT stock, for a price increase threshold of 0.5%, I've optimized the model to 2000 estimators, max_depth of 18, min_samples_split of 5, and a 'balanced_subsample' class_weight.

The following is the performance results that the model can realistically achieve:

Upwards of 57% precision

Upwards of 80% accuracy

Upwards of 54% ROC-AUC

When the model predicts a price increase, precision describes how accurate that prediction is. While 57% precision and 54% ROC-AUC doesn't seem that good, it indicates that the model actually works and has accurate price increase predictions the majority of the time. Considering how noisy and random stock market data is and how hard it is to predict the future, this model has found some patterns in the data.

Consider running this for yourself on different stocks, doing your own optimizing, and changing any of the techniques I've mentioned above to attempt to improve the model performance. There is definitely still more performance that can be gained, especially adding more stock market indicators to the model.

To get a reliable precision number, make sure the number of predictions being made during testing is significant and not too low (a few hundred predictions at least). Too high of a prediction rate (20 to 30% or more) is nice, but it may result in lower precision. Try and find the optimal prediction rate and precision. Try adjusting the predict_span, threshold, max_depth, and min_samples_split in the Stock class as well as train_frac which determines how many samples to train on versus test on.
