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

- Selección de features numéricas para regresión: Identifica las columnas numéricas del dataframe que tienen una correlación significativa con una columna designada como el objetivo de un modelo de regresión, con la opción de considerar un umbral de correlación y un valor de p-value para el test de hipótesis.

- Selección de features categóricas para regresión: Encuentra las columnas categóricas del dataframe que tienen una relación significativa con una columna designada como el objetivo de un modelo de regresión, con la opción de especificar un valor de p-value para el test de hipótesis.

- Visualización de relaciones: Genera visualizaciones como pairplots y histogramas agrupados para explorar las relaciones entre features y el target en problemas de regresión.


## Contribuidores
Este proyecto ha sido elaborado por:

  <li>Vanessa Patiño Del Hoyo
  <li>Alvaro Alonso Berenguer
  <li>Miguel Montuenga Díaz