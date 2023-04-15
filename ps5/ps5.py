# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import numpy as np
import scipy
import matplotlib.pyplot as plt
import re
import statistics as stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # TODO
    models = []
    for deg in degs:
        model = np.polyfit(x, y, deg)
        models.append(model)
    return models

def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    # TODO
    R_square = r2_score(y, estimated)
    return R_square

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # TODO
    for model in models:
        estimated = np.polyval(model, x)
        R_square = r_squared(y, estimated)
        plt.figure()
        plt.plot(x, y, 'bo', label='observed temperatures')
        
        if len(model) == 2:
            SE_over_slop = se_over_slope(x, y, estimated, model)
            plt.title(f'''Observed Temperature
                  R^2 = {R_square} degree = {len(model) - 1}
                  standard error of the slope = {SE_over_slop}''')
        else:
            plt.title(f'''Observed Temperature
                  R^2 = {R_square} degree = {len(model) - 1}''')
        plt.xlabel('Years')
        plt.ylabel('Temperature (Celsius)')
        
        plt.plot(x, estimated, 'r-', label=f'degree = {len(model) - 1}')
        plt.legend(loc='best')
        
def national_temp_1year(climate, multi_cities, year):
    multicities_temps = []
    for city in multi_cities:
        ave_temps = stats.mean(climate.get_yearly_temp(city, year))
        multicities_temps.append(ave_temps)
    national_ave = stats.mean(multicities_temps)
    return national_ave        

def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    # TODO
    national_annual_temp = []
    for year in years:
        national_ave = national_temp_1year(climate, multi_cities, year)
        national_annual_temp.append(national_ave)
    
    national_annual_temp = np.array(national_annual_temp)
    return national_annual_temp

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    # TODO
    mo_ave = []
    for i in range(len(y)):
        start_index = max(0, i - window_length+ 1)
        ave = stats.mean(y[start_index:i+1])
        mo_ave.append(ave)
        
    return mo_ave
def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    # TODO
    RMSE = mean_squared_error(y, estimated, squared=False)
    return RMSE

def std_yearly(climate, CITIES, year):
    yearly_temps = []
    for city in CITIES:
        yearly_temps.append(climate.get_yearly_temp(city, year))
    
    res = [0]*len(yearly_temps[0])
    for ele in yearly_temps:
        res += ele
    std = np.std(res)        
    return std    

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    # TODO
    national_std = []
    for year in years:
       national_std.append(std_yearly(climate, multi_cities, year))
        
    national_std = np.array(national_std)
    return national_std

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # TODO
    for model in models:
        estimated = np.polyval(model, x)
        RMSE = rmse(y, estimated)
        plt.figure()
        plt.plot(x, y, 'bo', label='observed temperatures')
        plt.title(f'''Observed Temperature
              RMSE = {RMSE} degree = {len(model) - 1}''')
        plt.xlabel('Years')
        plt.ylabel('Temperature (Celsius)')
        
        plt.plot(x, estimated, 'r-', label='Linear fit')
        plt.legend(loc='best')


def model_plot(x, y, degs):
    model = generate_models(x, y, degs)
    evaluate_models_on_training(x, y, model)
    return None


if __name__ == '__main__': 
    
    # Part A.4
    # TODO: replace this line with your code
    
# =============================================================================
#     rawdata = Climate('data.csv')
#     Jan_10 = []
#     for year in TRAINING_INTERVAL:
#         Jan_10.append(rawdata.get_daily_temp('NEW YORK', 1, 10, year))
# 
#     x = [i for i in TRAINING_INTERVAL]
#     x = np.array(x)
#     y = np.array(Jan_10)
#     model_plot(x, y, [1])
# =============================================================================
# =============================================================================
#     import statistics as stats
#     rawdata = Climate('data.csv')
#     annual = []
#     for year in TRAINING_INTERVAL:
#         ave_temp = stats.mean(rawdata.get_yearly_temp('NEW YORK', year))
#         annual.append(ave_temp)
#     
#     x = [i for i in TRAINING_INTERVAL]
#     x = np.array(x)
#     y = np.array(annual)
#     model_plot(x, y, [1])
# =============================================================================

    # Part B
    # TODO: replace this line with your code
# =============================================================================
#     climate = Climate('data.csv')
#     x = [i for i in TRAINING_INTERVAL]
#     x = np.array(x)
#     y = gen_cities_avg(climate, CITIES, x)
#     model_plot(x, y, [1])
# =============================================================================
       
    # Part C
    # TODO: replace this line with your code
# =============================================================================
#     climate = Climate('data.csv')
#     x = [i for i in TRAINING_INTERVAL]
#     x = np.array(x)
#     national_temps = gen_cities_avg(climate, CITIES, x)
#     y = moving_average(national_temps, 5)
#     model_plot(x, y, [1])
# =============================================================================
    # model = generate_models(x, y, [1])
    # evaluate_models_on_training(x, y, model)
    # Part D.2
    # TODO: replace this line with your code
    climate = Climate('data.csv')
    x = [i for i in TRAINING_INTERVAL]
    x = np.array(x)
    # national_temps = gen_cities_avg(climate, CITIES, x)
    # y = moving_average(national_temps, 5)
    # degs = [1, 2, 20]
    # # model_plot(x, y, degs)
   
    # models = generate_models(x, y, degs)
    # x_test = [i for i in TESTING_INTERVAL]
    # x_test = np.array(x_test)
    # National_temps = gen_cities_avg(climate, CITIES, x_test)
    # y_test = moving_average(National_temps, 5)
    # evaluate_models_on_testing(x_test, y_test, models)
    # evaluate_models_on_testing(x, y, models)
    
    # Part E
    # TODO: replace this line with your code
    stds = gen_std_devs(climate, CITIES, TRAINING_INTERVAL)
    std_moving = moving_average(stds, 5)
    model_plot(x, std_moving, [1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    