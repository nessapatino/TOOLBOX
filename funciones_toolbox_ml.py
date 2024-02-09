import pandas as pd


def tipifica_variables(df, umbral_categorica, umbral_continua):
    """
    Clasifica las variables de un DataFrame en función de sus características.

    Parametros:
    - df: DataFrame que contiene las variables a clasificar.
    - umbral_categorica (int): Umbral para clasificar una variable como categórica.
    - umbral_continua (float): Umbral para clasificar una variable como continua.

    Retorna:
    DataFrame con dos columnas: 'nombre_variable' y 'tipo_sugerido'.
                      'nombre_variable': Nombre de la variable.
                      'tipo_sugerido': Tipo sugerido para la variable ('Categorica', 'Binaria', 'Numerica Discreta', 'Numerica Continua').
    """



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
    tipos = df.dtypes
    nulos_porcentaje = df.isnull().mean() * 100
    unicos = df.nunique()
    cardinalidad_porcentaje = unicos / len(df) * 100

    df_describe = pd.DataFrame({
        "Tipo": tipos,
        "Porcentaje_Nulos": nulos_porcentaje,
        "Valores_Unicos": unicos,
        "Porcentaje_Cardinalidad": cardinalidad_porcentaje
    }).T



    return df_describe


