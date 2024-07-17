import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import calculate_statiscal_descriptors, remove_outliers, chi2_test, anova_test, fill_null_with_mean

from sklearn.preprocessing import LabelEncoder

#from phik import phik_matrix


def data_loading(path):
    """
    Esta funcion carga un archivo csv y lo convierte en un dataframe de pandas.

    Args:
        path (str): Ruta del archivo csv.	
    
    Returns:
        pd.DataFrame: Dataframe de pandas con los datos del archivo csv.
    """
    data = pd.read_csv(path, encoding="utf-8")
    return data


def delete_cols(df):
    """
    Esta funcion elimina las columnas instance_id, artist_name, track_name y obtained_date de un dataframe.

    Args:
        df (pd.DataFrame): Dataframe de pandas.
    
    Returns:
        pd.DataFrame: Dataframe de pandas sin las columnas instance_id, artist_name, track_name y obtained_date.
    """
    df = df.drop(["code", "station_name","code", "id","Vegitation (High)","Vegitation (Low)"], axis=1)
    return df


def data_nulls_processing(df):
    """
    Esta funcion elimina las filas con valores nulos del dataframe, los reemplaza por la media y retorna el dataframe.

    Args:
        df (pd.DataFrame): Dataframe de pandas.
    
    Returns:
        pd.DataFrame: Dataframe de pandas procesado.
    """
    df = df.dropna(subset=["PM10", "PM2.5"])
    # saco las columnas con nulos, que son realmente necesarias
    columns_nulls = df.columns[df.isnull().any()]
    for col in columns_nulls:
        df = fill_null_with_mean(col, df)
    return df


def date_treatment(df):
    """
    Esta funcion elimina los outliers de las columnas numericas de un dataframe, reemplaza los valores np.nan de la columna tempo por la media de la columna.

    Args:
        df (pd.DataFrame): Dataframe de pandas.
    
    Returns:
        pd.DataFrame: Dataframe de pandas sin outliers.
    """
    df_time = pd.to_datetime(df["Date"])
    #df_time = pd.DataFrame(df_time)  // Estuve hueveando 3 horas en hacer esto un DF
    #df['year'] = df['timestamp'].dt.year (Asi quizas resultaba mas facil)
    df_time_hour = df_time.dt.hour
    df_day_of_week = df_time.dt.dayofweek
    df_year = df_time.dt.year
    df_month = df_time.dt.month
    df_season = df_time.dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Autumn

    time_df = pd.DataFrame({
        'year': df_year,
        'month': df_month,
        'day_of_week': df_day_of_week,
        'hour': df_time_hour,
        'season': df_season
        })
    df = pd.concat([df, time_df], axis=1) # me queda duda si debo ponerlo en utils.py
    return df

def drop_date(df):
    df.drop("Date", axis=1, inplace=True)
    return df

def outliers_treatment(df):
    """
    Esta funcion elimina los outliers de las columnas numericas de un dataframe.

    Args:
        df (pd.DataFrame): Dataframe de pandas.
    
    Returns:
        pd.DataFrame: Dataframe de pandas sin outliers.
    """
    num_cols = list(df.select_dtypes(exclude="object").columns)
    col_sd = {}
    for col in num_cols:
        col_sd[col] = calculate_statiscal_descriptors(df, col)   
    for col in num_cols:
        mean, std = col_sd[col]
        df[col] = df[col].apply(lambda x: remove_outliers(x, mean, std))
    return df, num_cols 

def feature_engineering(df):
    """df = df.query("duration_ms >=10")
    df["prom1"] = df[["danceability", "instrumentalness"]].mean(axis=1)
    df["prom2"] = df[["energy", "loudness"]].mean(axis=1)
    df["prom3"] = df[["energy", "liveness"]].mean(axis=1)
    encoders = {}
    cat_cols = ["key", "mode", "music_genre"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[[col]])
        encoders[col] = le
    return df"""
    df["prom_viento"] = df[["Wind-Speed U","Wind-Speed V"]].mean(axis=1)
    df["ratio_viento"] = df["Wind-Speed U"]/df["Wind-Speed V"]
    df["prom_temp"] = df[["Soil Temp", "Dewpoint Temp","Temp"]].mean(axis=1)
    df["ratio_NO2_O3"] = df["O3"]/df["NO2"]
    df["prom_NO2_O3"] = df["O3"]*df["NO2"]
    #df["prom_NO2_wind"] = df["O3"]*df["prom_viento"]
    df["Prec_Humedad"] = df["Total Percipitation"]*df["Relative Humidity"]
    df["location"] = df[["Latitude", "Longitude"]].mean(axis=1)
    df["Year_time"] = df[["month", "season"]].mean(axis=1)
    df["Year_period"] = df["month"]*df["season"]
    return df



def feature_selection(df, num_cols):
    #cat_cols = ["key", "mode"]
    useful_var = []
    # aplicamos anova
    for col in num_cols:
        p = anova_test(df, col, "PM10")
        if p < 0.05:
            useful_var.append(col)
    """# segundo voy a hacer el chi2
    for col in cat_cols:
        p = chi2_test(df, col, "music_genre")
        if p < 0.05:
            useful_var.append(col)"""
    df = df[useful_var + ["PM10"]]
    matrix_phik = df.phik_matrix()
    final_var = list(matrix_phik.query("PM10 > 0.22").index)
    df_final = df[final_var]
    return df_final
    