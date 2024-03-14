# TOOLBOX
# Práctica Grupal: Feature Selection Toolbox

Este proyecto consiste en un toolbox diseñado para facilitar el análisis y selección de features de manera flexible y rápida, específicamente para problemas de Machine Learning.

## Descripción

En el ámbito del Machine Learning, la selección de features es un paso crucial para mejorar la precisión y eficiencia de los modelos. Sin embargo, este proceso puede ser tedioso y requiere conocimientos especializados. Este toolbox proporciona una serie de funciones y herramientas que simplifican este proceso, permitiendo a los usuarios explorar y seleccionar features de manera eficiente.


### Características principales

- Flexibilidad: El toolbox ofrece una amplia gama de métodos de selección de features para adaptarse a diferentes tipos de datos y problemas de ML.

- Rapidez: Las funciones están optimizadas para ejecutarse de manera eficiente, permitiendo un análisis rápido incluso en conjuntos de datos grandes.

- Facilidad de uso: La interfaz de usuario intuitiva y la documentación detallada hacen que sea fácil para los usuarios utilizar las funciones sin necesidad de conocimientos especializados en feature selection.

  
## Funcionalidades

El toolbox ofrece una serie de funciones que abordan diferentes aspectos del proceso de selección de features:

- Análisis de datos: Proporciona herramientas para realizar un análisis exploratorio de los datos, incluyendo resúmenes descriptivos y visualizaciones.

- Selección de tipos de variables: Permite sugerir el tipo de variable para cada columna del dataframe, considerando la cardinalidad y otros criterios definidos por el usuario.

- Selección de features numéricas para modelos de regresión: Identifica las columnas numéricas del dataframe que tienen una correlación significativa con una columna designada como el objetivo de un modelo de regresión, con la opción de considerar un umbral de correlación y un valor de p-value para el test de hipótesis.

- Selección de features categóricas para modelos de regresión: Encuentra las columnas categóricas del dataframe que tienen una relación significativa con una columna designada como el objetivo de un modelo de regresión, con la opción de especificar un valor de p-value para el test de hipótesis.

- Visualización para modelos de regresión: Genera visualizaciones como pairplots y histogramas agrupados para explorar las relaciones entre features y el target en problemas de regresión.

- eval_model: Evalúa el rendimiento de modelos de Machine Learning. Para regresión, calcula y muestra RMSE, MAE, MAPE, y genera gráficos comparativos. Para clasificación, calcula precisión, recall, accuracy, y muestra informes de clasificación y matrices de confusión. Devuelve las métricas en una tupla.

- Selección de features númericas para modelos de clasificación: Identifica columnas numéricas en un dataframe que son estadísticamente significativas para un target de clasificación, basándose en un valor p de ANOVA.

- Selección de features categóricas para modelos de clasificación: Selecciona columnas categóricas en un dataframe que tengan una Mutual information significativa con un target de clasificación, normalizada o no.

- Visualización para modelos de clasificación: : Genera pairplots para columnas numéricas seleccionadas en un dataframe, basadas en su significancia estadística con respecto a un target y crea visualizaciones de la distribución de etiquetas de columnas categóricas seleccionadas en relación con un target de clasificación.


## Contribuidores
Este proyecto ha sido elaborado por:

  <li>Vanessa Patiño Del Hoyo
  <li>Alvaro Alonso Berenguer
  <li>Miguel Montuenga Díaz