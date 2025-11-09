import pandas as pd
import numpy as np
import requests
import json
import time
import matplotlib.pyplot as plt
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

# TODO enter alpha vantage api key
g_apikey = ""

# Static class for calculations
class Utility:
    ## returns change between data points as a percentage
    ## new: newest data point
    # old: previous data point
    def percent_change(new, old) -> float:
        return 100 * (new - old) / old
    
    def log_change(new, old) -> float:
        return 100 * np.log(new / old)

# Static class for user input
class UserInput:
    ## return simple user input
    # message: message prompt for user input
    def get_input(message):
        user_input = input(message)
        print()
        return user_input
    
    ## return user input from a number of choices
    # messages: list of choices the user can choose from
    def get_input_choice(messages) -> int:
        while True:
            print("Choose option")
            
            # displays choices
            for i, message in enumerate(messages):
                print(f"[{i + 1}] {message}")

            # User inputs an integer to choose choice
            try:
                choice = int(input("Enter choice number: "))
            except Exception as error:
                print()
                print("Invalid input:", error)
                print()
                continue

            # User can only choose out of the choices given
            if choice <= 0 or choice > len(messages):
                print()
                print("Invalid input: out of range")
                print()
                continue

            print()
            return choice

# Stock class which handles data acquisition, pre-processing,
# model training, and price increase predicting of a particular stock
class Stock:
    ## Initializes stock data and ML model
    # symbol:           stock symbol
    # price_data:       Time series price data of the stock
    # volume data:      Time series volume data of the stock
    # time_span:        Number of months to access past stock data to train
    #                       the model over
    # time_interval:    Length of interval in minutes to acquire intraday
    #                       data (1, 5, 15, 30, 60 minutes).
    # intervals:        Number of "time_interval" minute intervals in
    #                       "lookback_span" hours. Determines the number of
    #                       elements to input into the ML model to predict
    #                       future price increases.
    # predict_span:     How many "time_interval" intervals to look ahead
    #                       to classify a price increase for training
    # threshold:        The percent change threshold required to label
    #                       to label a price increase in the training data
    # rfc:              The random forest classifier used to train and
    #                       predict price increases.
    def __init__(self, symbol):
        self.symbol = symbol
        self.price_data = []
        self.volume_data = []
        self.time_data = []
        self.features = []
        self.labels = []

        self.time_span = 48 # months
        lookback_span = 6.5 # hours
        self.time_interval = 5 # minutes
        self.intervals = int(round(lookback_span * 60 / self.time_interval))
        self.predict_span = 12 # 12 5 minute intervals -> 1 hour
        self.threshold = 0.5

        estimators = 1800
        jobs = 100
        depth = 17
        samples_split = 5
        weight = 'balanced_subsample'
        self.rfc = RandomForestClassifier(
                n_estimators = estimators, n_jobs = jobs, max_depth = depth,
                min_samples_split = samples_split, class_weight = weight
            )
        
    ##  Querys and saves new stock data
    def query(self):
        print("Querying...")

        # Uses alpha vantage to query intraday data
        months = []
        today = str(date.today())[:7]
        current_year = int(today[:4])
        current_month = int(today[5:])
        time_span = self.time_span
        while time_span > 0:
            if current_month <= 0:
                current_month = 12
                current_year -= 1
            if current_month < 10:
                months.append(str(current_year) + "-0" + str(current_month))
            else:
                months.append(str(current_year) + "-" + str(current_month))
            current_month -= 1
            time_span -= 1
        request_params  = "function=TIME_SERIES_INTRADAY"
        request_params += f"&symbol={self.symbol}"
        request_params += f"&interval={self.time_interval}min"
        request_params += "&entitlement=realtime"
        request_params += "&extended_hours=false"
        request_params += "&outputsize=full"
        request_params += "&datatype=json"
        request_params += f"&apikey={g_apikey}"
        request_params += "&month=xxxx-xx"
        url = "https://www.alphavantage.co/query?" + request_params

        # Data is put into single stock_data dictionary
        stock_data = {}
        for i in range(len(months)):
            url = url[:len(url) - 7] + months[i]
            response = requests.get(url).json()
            for key in response[f"Time Series ({self.time_interval}min)"]:
                stock_data[key] = response[
                        f"Time Series ({self.time_interval}min)"][key]
        print("Done querying")

        # Save stock data as json file
        with open(self.symbol + "_data.txt", "w") as file:
            json.dump(stock_data, file)
        print("Saved data")
        print()

        # Use stock data to set price and volume data
        self.set_data(stock_data)

    ## Gets previously saved stock data
    def get_data(self):
        print("Loading data...")
        stock_data = None
        with open(self.symbol + "_data.txt", "r") as file:
            stock_data = json.load(file)
        print("Data loaded")
        print()

        # Sets volume and price data
        self.set_data(stock_data)

    ## Private method which sets price and volume data according to stock data
    def set_data(self, stock_data):
        self.price_data = []
        self.volume_data = []
        self.time_data = []
        for key in stock_data:
            self.price_data.append(float(stock_data[key]["4. close"]))
            self.volume_data.append(float(stock_data[key]["5. volume"]))
            month = int(str(key)[5:7])
            day = int(str(key)[8:10])
            time = int(str(key)[11:13]) + int(str(key)[14:16]) / 60
            self.time_data.append([month, day, time])

        # Time series data is backwards and must be reversed
        self.price_data.reverse()
        self.volume_data.reverse()
        self.time_data.reverse()

    ## Creates the training set from price and volume data
    def create_training_set(self):
        print("Creating training set...")
        training_set = []
        time_offset = 0
        offset_increment = 1

        # Creates data samples
        while (time_offset + self.intervals + self.predict_span - 1 <
               len(self.price_data)):
            sample = []

            # Append price data
            for t in range(self.intervals):
                sample.append(self.price_data[t + time_offset])

            # Append volume data
            volume_intervals = self.intervals
            for t in range(volume_intervals):
                sample.append(self.volume_data[self.intervals + time_offset - volume_intervals + t])

            # Append classification
            price_now = self.price_data[self.intervals - 1 + time_offset]
            sample.append(0)
            for i in range(self.predict_span):
                price = self.price_data[self.intervals + time_offset + i]
                if Utility.percent_change(price, price_now) <= -self.threshold:
                    break
                elif Utility.percent_change(price, price_now) >= self.threshold:
                    sample[len(sample) - 1] = 1
                    break
            
            # Append sample to the training set
            training_set.append(sample)
            time_offset += offset_increment

        # Change price and volume data in each sample to percent change
        # rather than absolute value
        training_set = pd.DataFrame(training_set, dtype = pd.Float64Dtype())
        for col in range(self.intervals - 1):
            initial = training_set.iloc[:, col]
            final = training_set.iloc[:, col + 1]
            training_set.iloc[:, col] = Utility.log_change(final, initial)
        for col in range(self.intervals, self.intervals + volume_intervals - 1):
            initial = training_set.iloc[:, col]
            final = training_set.iloc[:, col + 1]
            training_set.iloc[:, col] = Utility.log_change(final, initial)
        print("Training set:", training_set.shape)
        print("Class 1:", sum(training_set.iloc[:,training_set.shape[1] - 1]))
        print()

        # Separate features and labels
        self.labels = training_set.iloc[:, training_set.shape[1] - 1]
        self.features = training_set.drop(training_set.columns[training_set.shape[1] - 1], axis = 1)

    ## Test the model performance
    def test(self):
        # Determine index to split the samples into training and testing
        # sets based off of the train_fraction
        print("Testing model performance...")
        train_fraction = 0.80
        use_sample_weights = False
        split_index = int(round(self.features.shape[0] * train_fraction))
        print("Train:", split_index, "samples")
        print("Test:", self.features.shape[0] - split_index, "samples")
        print()
        sample_weights = []
        if use_sample_weights:
            for x in range(split_index):
                sample_weights.append(np.exp(x / (split_index / np.log(2))) - 1)
        else:
            sample_weights = None

        # Get training and testing sets
        X_train = self.features.iloc[:split_index]
        y_train = self.labels.iloc[:split_index]
        X_test = self.features.iloc[split_index:]
        y_test = self.labels.iloc[split_index:]

        # Fit on training data
        self.rfc.fit(X_train, y_train, sample_weight = sample_weights)

        # Predict on testing data
        y_pred = self.rfc.predict(X_test)

        # Display model performance
        print("Testing done")
        print("Number of positive predictions:", sum(y_pred), str(np.round(sum(y_pred) / len(y_pred) * 100, 2)) + "%")
        print("Precision: ", precision_score(y_test, y_pred))
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("ROC AUC: ", roc_auc_score(y_test, y_pred))
        print()

    ## Trains the model on the whole training set
    def train(self):
        print("Training model...")
        self.rfc.fit(self.features, self.labels)
        print("Training done")
        print()

    ## Predicts price increase using most recently acquired data
    def predict(self):
        # Create a sample from most recent data
        print("Predicting...")
        stock = []
        for time in range(self.intervals):
            stock.append(self.price_data[len(self.price_data) - self.intervals + time])
        for time in range(self.intervals):
            stock.append(self.volume_data[len(self.price_data) - self.intervals + time])
        for t in range(self.intervals - 1):
            final = stock[t + 1]
            initial = stock[t]
            stock[t] = Utility.log_change(final, initial)
        for t in range(self.intervals, self.intervals * 2 - 1):
            final = stock[t + 1]
            initial = stock[t]
            stock[t] = Utility.log_change(final, initial)
        stock = [stock]

        # Predict price increase
        prediction = self.rfc.predict_proba(stock)[0][1]
        print("Predicting done")
        print()
        return prediction
    
    ## Appends current real time data to price and volume data
    def update_data(self):
        price = self.get_price_realtime()[0]
        volume = self.get_price_realtime()[1]
        self.price_data.append(price)
        self.volume_data.append(volume)

    ## Gets current real time price and volume data from the stock
    def get_price_realtime(self):
        response = "<Response [500]>"
        while str(response)[11:14] != "200":
            request_params  = "function=REALTIME_BULK_QUOTES"
            request_params += "&symbol=" + self.symbol
            request_params += "&datatype=json"
            request_params += "&apikey=KU38C6LM4CTUYU9P"
            url = "https://www.alphavantage.co/query?" + request_params
            response = requests.get(url)
            time.sleep(1)
        response = response.json()
        return (float(response['data'][0]['close']), float(response['data'][0]["volume"]))
    
    ## Displays features importances of the fitted model in a bar graph
    def display_feature_importance(self):
        # Create a pandas Series for easier plotting and sorting
        importances = self.rfc.feature_importances_
        feature_importances = pd.Series(importances, index=self.features.columns)
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        feature_importances.plot(kind='bar')
        plt.title('Feature Importances from Random Forest')
        plt.ylabel('Importance Score')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.show()

def main(g_apikey):
    if g_apikey == "":
        g_apikey = UserInput.get_input("Enter api key: ")

    input1 = UserInput.get_input("Enter symbol: ")
    input2 = UserInput.get_input_choice(["Test", "Predict"])
    input3 = UserInput.get_input_choice(["Use saved data", "Query new data"])

    stock = Stock(input1)

    if input3 == 1:
        stock.get_data()
    elif input3 == 2:
        stock.query()

    stock.create_training_set()

    if input2 == 1:
        stock.test()
    elif input2 == 2:
        start = time.perf_counter()
        stock.train()
        prediction = stock.predict()
        end = time.perf_counter()
        print("Price increase probability:", prediction)
        print("Time elapsed:", end - start, "seconds")
        print()
    stock.display_feature_importance()

if __name__ == "__main__":
    main(g_apikey)
    exit(0)