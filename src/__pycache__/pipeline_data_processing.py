from data_processing import data_loading, delete_cols, data_nulls_processing, date_treatment, drop_date, outliers_treatment, feature_engineering, feature_selection

def pipeline_datos_limpios(path):
    print("Procesando datos, pipeline inicia...")
    df = data_loading(path)
    print("Datos cargados...")
    df = delete_cols(df)
    print("Columnas eliminadas...")
    df = data_nulls_processing(df)
    print("Datos sin valores nulos...")
    df = date_treatment(df)
    print("Transforma fecha en valores útiles...")
    df = drop_date(df)
    print("Elimina la columna de fecha...")
    df = outliers_treatment(df)
    print("Datos sin outliers...")
    df = feature_engineering(df)
    print("Datos procesados con feature engineering...")
    df = feature_selection(df, num_cols)
    print("Datos procesados con feature selection...")
    print("Datos procesados, pipeline terminado.")
    df.to_csv("../../data/processed_data_lc.csv", index=False)

if __name__ == "__main__":
    pipeline_datos_limpios(path="../visual/data/athens_data.csv")