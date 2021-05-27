import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


def crop_pred(b_test):
    df = pd.read_csv('crop.csv')
    df['Yield'] = df['Production(Tonnes)']/df['Area(Hectare)']

    df = df.join(pd.get_dummies(df['District']))
    df = df.join(pd.get_dummies(df['Season']))
    df = df.join(pd.get_dummies(df['Crop']))
    df = df.join(pd.get_dummies(df['Year']))

    df=df.drop('District', axis=1)
    df = df.drop('Season',axis=1)
    df = df.drop('Crop',axis=1)
    df = df.drop('Year', axis=1)
    df = df.drop('Max Temp(C)', axis=1)
    df = df.drop('Min Temp(C)', axis=1)
    df = df.drop('Rainfall(in mm)', axis=1)
    df = df.drop('Rainfall Days', axis=1)
    df = df.drop('Avg Windspeed(kmph)', axis=1)


    from sklearn import preprocessing
    # Create x, where x the 'scores' column's values as floats
    x = df[['Area(Hectare)']].values.astype(float)
    x
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    #df_normalized = pd.DataFrame(x_scaled)
    x_scaled

    df['Area(Hectare)'] = x_scaled
    df.head

    df = df.fillna(df.mean())


    from sklearn.model_selection import train_test_split

    b = df['Yield(Tonnes/Hectare)']
    a = df.drop('Yield(Tonnes/Hectare)', axis = 1)

    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.3, random_state = 42)

    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a_train = sc.fit_transform(a_train)
    a_test = sc.transform(a_test)

    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(a_train, b_train)
    print('Intercept: \n', reg.intercept_)
    print('Coefficients: \n', reg.coef_)
    
    return reg.predict(b_test)





