#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)
sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px  

def get_column_names(data):
    """ This function will be used to extract the column names for numerical and categorical variables
    info from the dataset
    input: dataframe containing all variables
    output: num_vars-> list of numerical columns
            cat_vars -> list of categorical columns"""
        
    num_var = data.select_dtypes(include=['int', 'float']).columns
    print()
    print('Numerical variables are:\n', num_var)
    print('-------------------------------------------------')

    categ_var = data.select_dtypes(include=['category', 'object']).columns
    print('Categorical variables are:\n', categ_var)
    print('-------------------------------------------------') 
    return num_var,categ_var
    
    
def percentage_nullValues(data):
    """
    Function that calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    """
    null_perc = round(data.isnull().sum() / data.shape[0],3) * 100.00
    null_perc = pd.DataFrame(null_perc, columns=['Percentage_NaN'])
    null_perc= null_perc.sort_values(by = ['Percentage_NaN'], ascending = False)
    return null_perc


# In[26]:


def select_threshold(data, thr):
    """
    Function that  calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    
    """
    null_perc = percentage_nullValues(data)
      
    col_keep = null_perc[null_perc['Percentage_NaN'] < thr]
    col_keep = list(col_keep.index)
    print('Columns to keep:',len(col_keep))
    print('Those columns have a percentage of NaN less than', str(thr), ':')
    print(col_keep)
    data_c= data[col_keep]
    
    return data_c


# In[33]:


def fill_na(data):
    """
    Function to fill NaN with mode (categorical variabls) and mean (numerical variables)
    input: data -> df
    """
    for column in data:
        if data[column].dtype != 'object':
            data[column] = data[column].fillna(data[column].mean())  
        else:
            data[column] = data[column].fillna(data[column].mode()[0]) 
    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data


# In[2]:

def OutLiersBox(df,nameOfFeature):
    """
    Function to create a BoxPlot and visualise:
    - All Points in the Variable
    - Suspected Outliers in the variable

    """
    trace0 = go.Box(
        y = df[nameOfFeature],
        name = "All Points",
        jitter = 0.3,
        pointpos = -1.8,
        boxpoints = 'all', #define that we want to plot all points
        marker = dict(
            color = 'rgb(7,40,89)'),
        line = dict(
            color = 'rgb(7,40,89)')
    )

    
    trace1 = go.Box(
        y = df[nameOfFeature],
        name = "Suspected Outliers",
        boxpoints = 'suspectedoutliers', # define the suspected Outliers
        marker = dict(
            color = 'rgba(219, 64, 82, 0.6)',
            #outliercolor = 'rgba(219, 64, 82, 0.6)',
            line = dict(
                outlierwidth = 2)),
        line = dict(
            color = 'rgb(8,81,156)')
    )


    data = [trace0,trace1]

    layout = go.Layout(
        title = "{} Outliers".format(nameOfFeature)
    )

    fig = go.Figure(data=data,layout=layout)
    fig.show()
    #fig.write_html("{}_file.html".format(nameOfFeature))

# In[3]:


def corrCoef(data):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    
    input: data->dataframe        
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    num_vars, categ_var = get_column_names(data)
    data_num = data[num_var]
    data_corr = data_num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(data_corr,
                xticklabels = data_corr.columns.values,
               yticklabels = data_corr.columns.values,
               annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7))


# In[4]:

def corrCoef_Threshold(df):
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Draw the heatmap
    sns.heatmap(df.corr(), annot=True, mask = mask, vmax=1,vmin=-1,
                cmap=sns.color_palette("RdBu_r", 7));


def outlier_treatment(df, colname):
    """
    Function that drops the Outliers based on the IQR upper and lower boundaries 
    input: df --> dataframe
           colname --> str, name of the column
    
    """
    
    # Calculate the percentiles and the IQR
    Q1,Q3 = np.percentile(df[colname], [25,75])
    IQR = Q3 - Q1
    
    # Calculate the upper and lower limit
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    
    # Drop the suspected outliers
    df_clean = df[(df[colname] > lower_limit) & (df[colname] < upper_limit)]
    
    print('Shape of the raw data:', df.shape)
    print('..................')
    print('Shape of the cleaned data:', df_clean.shape)
    return df_clean
       
    
def outliers_loop(df_num):
    """
    jsklfjfl
    
    """
    for item in np.arange(0,len(df_num.columns)):
        if item == 0:
            df_c = outlier_treatment(df_num, df_num.columns[item])
        else:
            df_c = outlier_treatment(df_c, df_num.columns[item]) 
    return df_c   


# pseudo-code for creating an own functions:
#1. parameters: `path1, path2, path3`, name of the variable,
#2. creating the paths: path1 = "xy1.csv", path2 = "xy2.csv" ...
#3. loading the dataframes: df1 = pd.read_csv(path1, sep=';') ...
#4. putting them all together in one dataframe: df_<name of the variable> = pd.concat([df1, df2, df3])
#5. check, if sum_rows of each df is the same as in df_<name of the variable>
#6. print out, that it is complete (or not)
#2. return: 1 dataframe, where all data is in one dataframe


def df_concat(path1, path2, seperator):
    """
    it loads the data that is in multiple csv-files
    and put it together in one file by giving the paths and
    the seperator as strings
    """
    path1 = path1
    path2 = path2

    df_1 = pd.read_csv(path1, sep=seperator)
    df_2 = pd.read_csv(path2, sep=seperator)

    df = pd.concat([df_1, df_2])
    
    sum_rows = df_1.shape[0] + df_2.shape[0]

    if df.shape[0] == sum_rows:
        print("dataframe for is completed. There are", sum_rows, "rows.");
    else:
        print("there's a mistake")
        print("Sum of the rows of each df:", sum_rows)
        print("Rows of the new dataframe:", df.shape[0])
    return df      

def df_concat_excel(path1, path2, path3):
    """
    it loads the data that is in multiple excel-files
    and put it together in one file by giving the paths
    """
    path1 = path1
    path2 = path2
    path3 = path3

    df_1 = pd.read_excel(path1)
    df_2 = pd.read_excel(path2)
    df_3 = pd.read_excel(path3)

    df = pd.concat([df_1, df_2, df_3])
    
    sum_rows = df_1.shape[0] + df_2.shape[0] + df_3.shape[0]

    if df.shape[0] == sum_rows:
        print("dataframe for is completed. There are", sum_rows, "rows.");
    else:
        print("there's a mistake")
        print("Sum of the rows of each df:", sum_rows)
        print("Rows of the new dataframe:", df.shape[0])
    return df  

