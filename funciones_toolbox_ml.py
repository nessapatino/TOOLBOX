import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.proportion import proportions_ztest


def tipifica_variables(df, umbral_categorica, umbral_continua):
    """
    Clasifica las variables de un DataFrame en función de sus características.

    Argumentos:
    - df: DataFrame que contiene las variables a clasificar.
    - umbral_categorica (int): Umbral para clasificar una variable como categórica.
    - umbral_continua (float): Umbral para clasificar una variable como continua.

    Retorna:
    DataFrame con dos columnas: 'nombre_variable' y 'tipo_sugerido'.
                      'nombre_variable': Nombre de la variable.
                      'tipo_sugerido': Tipo sugerido para la variable ('Categorica', 'Binaria', 'Numerica Discreta', 'Numerica Continua').
    """
    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame) or not isinstance(umbral_categorica, int) or not isinstance(umbral_continua, int):
        print("Error: df debe ser un DataFrame, 'umbral_categorica' un entero y 'umbral_continua' un entero.")
        return None
   
    df_tipificacion = pd.DataFrame([df.nunique(), df.nunique() / len(df) * 100, df.dtypes]).T.rename(columns={0: "Card", 1: "%_Card", 2: "Tipo"})
    df_tipificacion["Clasificada_como"] = "Categorica"
    df_tipificacion.loc[df_tipificacion.Card == 2, "Clasificada_como"] = "Binaria"
    df_tipificacion.loc[df_tipificacion["Card"] >= umbral_categorica, "Clasificada_como"] = "Numerica Discreta"
    df_tipificacion.loc[(df_tipificacion["%_Card"] >= umbral_continua) & (df_tipificacion["Card"] >= umbral_categorica), "Clasificada_como"] = "Numerica Continua"

    resultado = pd.DataFrame({
        "nombre_variable": df.columns,
        "tipo_sugerido": df_tipificacion["Clasificada_como"].values
    })
    return resultado

def describe_df(df):
    """
    Realiza un análisis descriptivo de un DataFrame.

    Argumentos:
    - df (DataFrame): El DataFrame a analizar.

    Retorna:
    DataFrame: Un DataFrame que contiene información sobre las características de cada columna del DataFrame de entrada.
        - Tipo: El tipo de datos de cada columna.
        - Porcentaje_Nulos: El porcentaje de valores nulos o missing en cada columna.
        - Valores_Unicos: El número de valores únicos en cada columna.
        - Porcentaje_Cardinalidad: El porcentaje de cardinalidad, es decir, la proporción de valores únicos respecto al total de valores, en cada columna.
    """
    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame")
        return None
    
    # Crea las variables
    tipos = df.dtypes
    nulos_porcentaje = df.isnull().mean() * 100
    unicos = df.nunique()
    cardinalidad_porcentaje = unicos / len(df) * 100

    # Crea el dataframe de salida
    df_describe = pd.DataFrame({
        "Tipo": tipos,
        "Porcentaje_Nulos": nulos_porcentaje,
        "Valores_Unicos": unicos,
        "Porcentaje_Cardinalidad": cardinalidad_porcentaje
    }).T

    return df_describe


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Esta función devuelve una lista con las columnas numéricas cuya correlación con 'target_col'
    es superior al valor absoluto de 'umbral_corr' y, si se proporciona 'pvalue', supera
    el test de hipótesis con significación mayor o igual a 1-pvalue.

    Argumentos:
    - df: DataFrame de pandas.
    - target_col: Nombre de la columna que se considerará el target del modelo de regresión.
    - umbral_corr: Valor umbral para la correlación (debe estar entre 0 y 1).
    - pvalue: Valor p para el test de hipótesis (si se proporciona, debe ser un número entre 0 y 1).

    Retorna:
    - Lista de columnas numéricas que cumplen con los criterios especificados.
    """

    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame) or not isinstance(target_col, str):
        print("Error: df debe ser un DataFrame y 'target_col' debe ser una cadena.")
        return None

    if target_col not in df.columns:
        print(f"Error: {target_col} no está presente en el DataFrame.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: 'pvalue' debe ser None o un número entre 0 y 1.")
        return None

    # Verifica que 'target_col' sea una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: 'target_col' debe ser una variable numérica continua.")
        return None

    # Comprueba valores nulos en el DataFrame
    if df.isnull().any().any():
        print("Error: El DataFrame contiene valores nulos. Imputa o elimina los valores nulos antes de continuar.")
        return None

    # Calcula la correlación y realiza las comprobaciones adicionales según los parámetros proporcionados
    correlated_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            correlation, p_value = pearsonr(df[col], df[target_col])

            if abs(correlation) > umbral_corr and (pvalue is None or p_value <= 1 - pvalue):
                correlated_columns.append(col)
    
    return correlated_columns

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Esta función pinta un pairplot del DataFrame considerando la columna designada por "target_col" y
    aquellas incluidas en "columns" que cumplen con ciertas condiciones de correlación.
    Además, devuelve una lista de las columnas que cumplen con estas condiciones.

    Argumentos:
    - df: DataFrame de pandas.
    - target_col: Nombre de la columna que se considerará el target del modelo de regresión.
    - columns: Lista de nombres de columnas numéricas a considerar (por defecto, la lista vacía).
    - umbral_corr: Valor umbral para la correlación (debe estar entre 0 y 1, por defecto 0).
    - pvalue: Valor p para el test de hipótesis (si se proporciona, debe ser un número entre 0 y 1).

    Retorna:
    - Lista de columnas numéricas que cumplen con las condiciones especificadas.
    """

    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame) or not isinstance(target_col, str):
        print("Error: 'df' debe ser un DataFrame y 'target_col' debe ser una cadena.")
        return None

    if target_col not in df.columns:
        print(f"Error: {target_col} no está presente en el DataFrame.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: 'pvalue' debe ser None o un número entre 0 y 1.")
        return None

    # Comprueba valores nulos en el DataFrame
    if df.isnull().any().any():
        print("Error: El DataFrame contiene valores nulos. Imputa o elimina los valores nulos antes de continuar.")
        return None

    # Verifica que 'target_col' sea una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: 'target_col' debe ser una variable numérica continua.")
        return None

    # Si 'columns' está vacío, utiliza todas las columnas numéricas del DataFrame
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filtra las columnas según las condiciones de correlación y p-value
    filtered_columns = []
    for col in columns:
        if col != target_col:
            correlation, p_value = pearsonr(df[col], df[target_col])

            if abs(correlation) > umbral_corr and (pvalue is None or p_value <= 1 - pvalue):
                filtered_columns.append(col)

    # Si la lista de columnas es grande, divide en varios pairplots
    num_plots = len(filtered_columns)
    num_subplots = min(num_plots, 5)

    for i in range(0, num_plots, num_subplots):
        cols_to_plot = filtered_columns[i:i + num_subplots] + [target_col]
        sns.pairplot(df[cols_to_plot])
        plt.show()


    return filtered_columns

def get_features_cat_regression(df, target_col, pvalue=0.05, cardinality_threshold=10):

    """
    Identifica columnas categóricas en un DataFrame que tienen una relación estadísticamente significativa
    con una columna objetivo numérica, utilizando T-Test para baja cardinalidad y Z-Test para alta cardinalidad.

    Argumentos:
    df (pd.DataFrame): El DataFrame que contiene los datos a analizar.
    target_col (str): El nombre de la columna objetivo numérica en el DataFrame.
    pvalue (float): El valor p umbral para determinar la significancia estadística.
    cardinality_threshold (int): El umbral de porcentaje para determinar alta cardinalidad en la columna objetivo.

    Retorna:
    - Lista de columnas categóricas que tienen una relación estadísticamente significativa con la columna objetivo.
    """
    
    if df.isnull().any().any():
        print("Advertencia: El DataFrame contiene valores NaN.")
        return None

    if target_col not in df.columns:
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna '{target_col}' debe ser numérica.")
        return None

    cardinality_percentage = df[target_col].nunique() / len(df) * 100
    if cardinality_percentage < cardinality_threshold:
        print(f"La columna '{target_col}' no tiene alta cardinalidad. Cardinalidad como porcentaje del total: {cardinality_percentage:.2f}%")
        return None

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    significant_cols = []
    overall_mean = df[target_col].mean()

    for col in categorical_cols:
        unique_values = df[col].nunique()

        if unique_values > 1 and unique_values <= 30:
            for unique_value in df[col].unique():
                group = df[df[col] == unique_value][target_col].dropna()
                if group.var() == 0:  
                    continue  
                t_stat, p_val = stats.ttest_1samp(group, overall_mean, nan_policy='omit')
                if p_val < pvalue:
                    significant_cols.append(col)
                    break

        elif unique_values > 30:
            for unique_value in df[col].unique():
                subgroup = df[df[col] == unique_value][target_col]
                success_count = np.sum(subgroup > overall_mean)
                nobs = len(subgroup)

                if nobs == 0:  
                    continue

                z_stat, p_val = proportions_ztest(success_count, nobs, prop=0.5)
                if p_val < pvalue:
                    significant_cols.append(col)
                    break

    return significant_cols

def plot_features_cat_regression(df, target_col, columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Genera histogramas agrupados de la variable target_col para cada uno de los valores de las variables
    categóricas en 'columns' que tienen una relación estadísticamente significativa con target_col,
    basado en un T-Test para baja cardinalidad y Z-Test para alta cardinalidad.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame que contiene los datos a analizar.
    target_col (str): El nombre de la columna objetivo numérica en el DataFrame.
    columns (list): Lista de nombres de columnas categóricas a analizar. Si está vacía, se utilizarán todas las columnas categóricas.
    pvalue (float): El valor p umbral para determinar la significancia estadística.
    with_individual_plot (bool): Si es True, genera un histograma para cada columna categórica significativa.
    
    Retorna:
    - Lista de nombres de columnas categóricas que tienen una relación estadísticamente significativa con la columna objetivo.
    """
    

    if df.isnull().any().any():
        print("Advertencia: El DataFrame contiene valores NaN.")
        return None

    if not target_col or target_col not in df.columns:
        print("La columna target_col no está en el DataFrame o no se especificó correctamente.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La columna target_col debe ser numérica.")
        return None

    if not columns:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    significant_cols = []

    for col in columns:
        if col not in df.columns:
            continue
        unique_values = df[col].nunique()

        if unique_values > 1:  
            for unique_value in df[col].unique():
                group = df[df[col] == unique_value][target_col].dropna()
                if group.var() == 0:  
                    continue
                if unique_values <= 30:  
                    t_stat, p_val = stats.ttest_1samp(group, df[target_col].mean(), nan_policy='omit')
                else:  
                    success_count = (group > df[target_col].mean()).sum()
                    nobs = len(group)
                    z_stat, p_val = stats.proportions_ztest(success_count, nobs, prop=0.5)

                if p_val < pvalue:
                    significant_cols.append(col)
                    break 
                
    if with_individual_plot and significant_cols:
        for col in significant_cols:
            if col in df.columns:
                df.groupby(col)[target_col].plot(kind='hist', alpha=0.5, legend=True, title=f"Histograma de '{target_col}' por '{col}'")
                plt.xlabel(target_col)
                plt.ylabel('Frecuencia')
                plt.legend(title=col)
                plt.show()

    return significant_cols