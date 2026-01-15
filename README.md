# Stock Market Predictor

## Introduction and usage

This program uses a random forest classifier from the scikit-learn python library to predict future stock price increases. For use, market_predictor.py should be run directly in a terminal.

### When running the file:
- Enter the symbol of the stock you want the model to train on
- Query new data (or load saved data if already queried)
- Choose whether to test the model performance or predict a future price increase

To query stock market data, the program uses the Alpha Vantage API. You must enter in your Alpha Vantage API key to use the program. The api key can be obtained for free and can be used to do your own model testing, but if you want to get real time data and do real time predictions, you need the paid premium alpha vantage subscription (or modify the program and use your own favorite stock market API).

## Model description

My model uses 5 minute intraday closing price data over the last 4 years, corresponding to about 77,000 samples.

### Training set:

- ~77,000 rows
- 156 feature columns
  - First 78 columns are closing price data over the last day (6.5 trading hours)
  - Second 78 columns are volume data over the same period
- 1 Binary classification column [0,1]
  - Class = 1 means price increases above a particular percent change threshold without decreasing by the same threshold within an hour
  - Class = 0 means price does not increase beyond that threshold or instead decreases by that threshold first

The reason that only price and volume data is included in the model is because, based off of my testing, that's all you need for the model to learn. Adding in calculated stock signals or even time data seems to add unnecessary noise, resulting in decreased model performance.

Because we want the model to learn how past price fluctuations determine the probability of future price increases, we want the price and volume data in each sample to be a percent change rather than absolute values. Percent changes however are very dependent on initial values and don't add together well when representing change over multiple intraday periods. So instead, I use log change so that time series data percent gains and losses are properly symmetric. Except for the most recent price and volume data points, which are just absolute values, all data points are log changes between the 5 minute intraday periods.

## Results

### The following is the performance results that the model can realistically achieve:

- Upwards of 57% precision
- Upwards of 80% accuracy  
- Upwards of 54% ROC-AUC

When the model predicts a price increase, precision describes how accurate that prediction is. While 57% precision and 54% ROC-AUC doesn't seem that good, it indicates that the model actually works and has accurate price increase predictions the majority of the time. Considering how noisy and random stock market data is and how hard it is to predict the future, this model has found some patterns in the data.

## Do your own testing

Consider running this for yourself on different stocks, doing your own optimizing, and changing any of the parameters or general techniques I've mentioned above to attempt to improve the model performance.

To get a reliable precision number, make sure the number of predictions being made during testing is significant and not too low (a few hundred predictions at least). Too high of a prediction rate (20 to 30% or more) is nice, but it may result in lower precision. Try and find the optimal prediction rate and precision. Try adjusting the predict_span, threshold, max_depth, and min_samples_split in the Stock class as well as train_frac which determines how many samples to train on versus test on.
