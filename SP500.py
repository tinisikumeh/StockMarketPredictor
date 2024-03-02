
import yfinance as yf
#ticker class allows us to download data on the stock passed in
#once we have a ticker we can download the data on a stock
sp500 = yf.Ticker("^GSPC")
#history queries all data since the sp500 index was created
sp500 = sp500.history(period="max")
sp500 # gives us a pandas data frame

sp500.index # gives us a list with all the dates since the beginning of the stocks history

sp500.plot.line(y = "Close", use_index = True) # second param tells us to use the dates along the x- axis
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500["Tommorow"] = sp500["Close"].shift(-1)#makes the tmrw col = the opening price for the next day
sp500
sp500["Target"] = (sp500["Tommorow"] > sp500["Close"]).astype(int)
sp500
sp500 = sp500.loc["1990-01-01":].copy()# pandas loc allows us to take the rows were the index is at least the date pased in
sp500

#randomForestClassifier is a machine learning model that trains a bunch of decison trees w/ randmomized params
#it averages the result from these trees, making randForest moreresistant to overfitting
# when multiple decision trees form an ensemble in the random forest algorithm, they predict more accurate results
from sklearn.ensemble import RandomForestClassifier #can pickup on nonlinear relationships

#n_estimators - the # of decision trees in our forest that we want to train, the more trees the more accurate results up to a point
#min_sample_split - minimum number of samples required to split an internal node, helps us stop overfitting
#higher we set it, the less accurate it will be, but we will be more resistant to overfittong
#random_state - is set to 1, allows us to get the same result each time we run the model so we can make improvements

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)

train = sp500.iloc[:-100] #putting all the rows except the lat 100 into the training set
test = sp500.iloc[-100:] #putting the last 100 days data  into the test set

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"]) #builds forest of trees with data in order to predict the target

#precision score - when we said the market woule go up, did it go up
#what percentage of the time did the market actually go up
from sklearn.metrics import precision_score

preds = model.predict(test[predictors]) #predicted target

# preds gives us a numPy array that's hard to work with so im gonna make it a pandas series
import pandas as pd

preds = pd.Series(preds, index = test.index) # organizing the data in the series by the stock price date

preds
precision_score(test["Target"], preds)#calculating how correct we were by comparing the actual target to predicted target
#axis treats each of our inputs as a column in our data set
combined = pd.concat([test["Target"], preds], axis =1) #adding our predicted and actual values together in one df
#combined
combined.plot()
#Building a backtesting system

#def predict(train, test, predictors, model):
#    model.fit(train[predictors], train["Target"])
#    preds = model.predict(test[predictors])
#    preds = pd.Series(preds, index = test.index, name = "Predictions")
#    combined = pd.concat([test["Target"], preds], axis =1)
#    return combined

# every trading year has 250 days
#by having star be 2500, we're telling our first model to train 10 years of data
#we'll first take the first 10 years of data and use it to predict the 11th year
def backtest(data, model, predictors, start = 2500, step = 250):
    all_predictions = [] #stores dataframe with each each df having the predictions for a single year
    
    for i in range(start, data.shape[0], step): #function loops across data year by year
        train = data.iloc[0: i].copy()
        test = data.iloc[i: i + step].copy() #creating test set data
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions) #combines our list of dataframes into one whole dateframe
    

        
predictions = backtest(sp500, model, predictors)
predictions["Predictions"].value_counts()
#predicted the market would go down 3500 days and up 2602
precision_score(predictions["Target"], predictions["Predictions"])
#percentage of days where the market went up and down 22:00
predictions["Target"].value_counts()/predictions.shape[0]#num of rows
horizons = [2, 5, 60, 250, 1000]#last two days, trading week, 3 months, trading year, last 4 trading years

new_predictors=[]

#with this we're going to try to find the ration of the price today compared to the time length in our  horizons
for horizon in horizons :
    #A rolling window is a fixed-size interval or subset of data that moves sequentially through a larger dataset.
    rolling_averages = sp500.rolling(horizon).mean()#average of data over the horizon
    
    ratio_column = f"Close_ratio_{horizon}"
    sp500[ratio_column] = (sp500["Close"]/rolling_averages["Close"])#ratio between today's close and the average close over the horizon

    #trend over the past horizon days
    trend_column = f"Trend_{horizon}"
    
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]#sum of the number of days where the stock price went up 
    #aka the sum of the target

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()# getting rid of NaN  da
sp500
model = RandomForestClassifier(n_estimators = 200, min_samples_split=50, random_state =1)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]#returns the probability that the row will be a zero or 1
    preds[preds >= .6] = 1 #setting a custom threshold so that if the prob that the proce goes is greater than 60% we pick 1 for this spot
    preds[preds < .6] = 0 # these changes make our model more confident in its predictions
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis =1)
    return combined

predictions = backtest(sp500, model, new_predictors)
predictions["Predictions"].value_counts()
precision_score(predictions["Target"], predictions["Predictions"])
