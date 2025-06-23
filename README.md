# T10_ML_Prediction_EduMendez

Proyecto de predicción del rendimiento del Treasury 10Y usando modelos de Machine Learning para el curso EAE3709 (PUC, 2025).

# ---------------------------------------------------------
# DESCARGA Y VISUALIZACIÓN DE LA TASA DEL TREASURY A 10 AÑOS (T10)
# ---------------------------------------------------------

En esta primera sección del proyecto, se realiza la descarga de la serie histórica del rendimiento del bono del Tesoro de EE.UU. a 10 años (ticker: DGS10) utilizando la librería fredapi, que accede a datos macroeconómicos provistos por la Reserva Federal (FRED).

Esta variable (T10_yield) será utilizada como variable objetivo (y) del modelo supervisado, dado que se trata de una tasa financiera clave en los mercados de deuda, inversión y política monetaria. Es observable, continua y tiene alta dependencia de factores macro, lo cual la convierte en un caso ideal para aplicar modelos de Machine Learning de regresión.

Además, se realiza una visualización básica para explorar su evolución temporal y confirmar que la serie está limpia y completa, eliminando observaciones nulas con .dropna().

# ---------------------------------------------------------
# DESCARGA DE VARIABLES MACROFINANCIERAS DIARIAS
# ---------------------------------------------------------

En este bloque se descargan desde FRED cuatro variables macrofinancieras diarias relevantes para explicar el comportamiento del rendimiento del Treasury a 10 años:
	•	VIX (VIXCLS): Índice de volatilidad implícita del S&P500, considerado un indicador de incertidumbre financiera global.
	•	DXY (DTWEXBGS): Índice del valor del dólar estadounidense frente a una canasta de monedas extranjeras. Influye en flujos de capital y tasas de interés.
	•	WTI (DCOILWTICO): Precio spot del petróleo West Texas Intermediate. Afecta expectativas inflacionarias y política monetaria.
	•	TED Spread (TEDRATE): Diferencial entre tasas LIBOR y bonos del Tesoro, usado como medida del riesgo sistémico en el sistema financiero.

Estas variables serán utilizadas como predictoras (X) en los modelos de Machine Learning, ya que tienen un fuerte impacto en la dinámica de tasas de interés de largo plazo. Además, se visualizan juntas para examinar su evolución histórica, detectar patrones o posibles faltantes antes del modelado.

# ---------------------------------------------------------
# DESCARGA DEL ÍNDICE S&P 500 DESDE YAHOO FINANCE (MULTIINDEX CORREGIDO)
# ---------------------------------------------------------

Se incorpora el índice bursátil S&P 500 (^GSPC en Yahoo Finance) como variable explicativa adicional del modelo.

El S&P 500 refleja las expectativas del mercado sobre crecimiento, tasas de interés y riesgo financiero, por lo que puede contener información útil para anticipar movimientos en el rendimiento de los bonos del Tesoro.

Se descarga la serie de precios ajustados diarios (Adj Close) desde el año 2000 hasta la fecha actual. Posteriormente, se limpia el índice, se renombran las columnas y se grafica para verificar visualmente su evolución.

# ---------------------------------------------------------
# UNIFICACIÓN DE VARIABLES Y EXPORTACIÓN DE LA SABANA FINAL
# ---------------------------------------------------------

En esta sección se realiza la unificación de todas las series macroeconómicas previamente descargadas —incluyendo la tasa del Treasury a 10 años (T10_yield), el índice de volatilidad (VIX), el tipo de cambio efectivo del dólar (DXY), el precio del petróleo (Oil_WTI), el TED Spread (TED_Spread) y el índice S&P 500 (SP500)— en un único DataFrame final.

La fusión se realiza mediante join iterativo con la opción how='inner', lo cual garantiza que solo se retengan aquellas fechas para las cuales hay observaciones disponibles en todas las variables.

Luego se realiza una revisión estructural (primeras filas, cantidad de datos, fechas disponibles) y se confirma que no existen valores nulos tras la unificación.

Finalmente, se exporta la sabana consolidada como archivo .csv, con el nombre sabana_macro_t10.csv, para que pueda ser utilizada en las siguientes etapas del proyecto (EDA y modelado supervisado).

# ---------------------------------------------------------
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ---------------------------------------------------------

En este trabajo se busca predecir el rendimiento del bono del Tesoro de Estados Unidos a 10 años (T10_yield) utilizando variables macrofinancieras diarias. Dado que se trata de una variable continua y que las relaciones entre las variables pueden ser tanto lineales como no lineales, se optó por comparar diversos modelos supervisados de regresión con diferentes niveles de complejidad e interpretabilidad. A continuación, se detallan los modelos seleccionados y la motivación de su uso:

 1. Regresión Lineal

Se utiliza como modelo baseline debido a su simplicidad y facilidad de interpretación. Permite establecer un punto de referencia inicial y detectar si existen relaciones lineales fuertes entre las variables predictoras y la variable objetivo.

 2. Árbol de Decisión

Modelo no paramétrico que captura relaciones no lineales y permite interpretar decisiones mediante reglas. Es útil para explorar divisiones simples del espacio de predicción y entender la lógica detrás de modelos más complejos.

 3. Random Forest

Extensión de los árboles de decisión mediante la técnica de bagging (bootstrap aggregation), que mejora la capacidad predictiva y reduce el sobreajuste. Además, proporciona medidas de importancia de variables que permiten interpretar el modelo.

 4. XGBoost

Modelo basado en gradient boosting que combina múltiples árboles secuenciales. Se justifica por su alto rendimiento en problemas estructuralmente complejos como los financieros, y su capacidad para capturar interacciones no lineales y efectos combinados.

 5. Red Neuronal (MLP Regressor)

Modelo no lineal flexible que permite aproximar funciones altamente complejas. Se incluye para evaluar si una arquitectura neuronal densa puede superar en desempeño a los modelos basados en árboles. Fue revisado en la ayudantía 11 del curso.

⸻

 Métricas de evaluación

Todos los modelos serán evaluados sobre un conjunto de test no visto (20% final de la serie) utilizando las siguientes métricas:
	•	R² (Coeficiente de Determinación): Proporción de la varianza explicada por el modelo.
	•	RMSE (Root Mean Squared Error): Error medio en las mismas unidades del T10_yield.

Esta batería de modelos permitirá evaluar tanto el rendimiento predictivo como la interpretabilidad de cada enfoque, siguiendo las directrices metodológicas del curso y alineado a los criterios de la rúbrica oficial.

# ---------------------------------------------------------
# Modelo 1: Regresión Lineal
# ---------------------------------------------------------

Como línea base, se entrenó una regresión lineal multivariada con el objetivo de predecir el rendimiento del bono del Tesoro estadounidense a 10 años (T10_yield) utilizando variables macroeconómicas como el índice VIX, el precio del petróleo WTI, el índice dólar DXY, el TED spread y el S&P 500.

Se aplicó una estandarización previa de las variables predictoras mediante StandardScaler, dado que este tipo de modelo es sensible a las escalas de los datos. Para evitar fugas de información propias de problemas temporales, se realizó un split cronológico del dataset, reservando el 20% final de las observaciones para evaluación.

Las métricas utilizadas fueron:
	•	R² (Coeficiente de determinación): mide cuánta varianza del T10_yield explica el modelo.
	•	RMSE (Root Mean Squared Error): representa el error medio de las predicciones en las mismas unidades que el target.

Este modelo, al ser simple y explicativo, actúa como benchmark contra el cual se compararán modelos más complejos. Sus resultados nos permiten evaluar si es necesario incorporar no linealidades o interacciones en la modelación.

# ---------------------------------------------------------
# MODELO 2: ÁRBOL DE DECISIÓN
# ---------------------------------------------------------

Se entrenó un modelo de Árbol de Decisión para regresión con el objetivo de capturar relaciones no lineales y efectos de umbral entre las variables macroeconómicas y el rendimiento del bono del Tesoro a 10 años (T10_yield). Este modelo es capaz de segmentar el espacio de predicción de forma jerárquica, dividiendo los datos en subconjuntos homogéneos.

Aunque no requiere escalado, se mantuvo la estandarización para mantener consistencia metodológica con los demás modelos. Además, se respetó la estructura temporal dividiendo el dataset en un 80% para entrenamiento y un 20% para evaluación, en orden cronológico.

Se utilizó una profundidad máxima (max_depth) de 5 para evitar sobreajuste, con posibilidad de ajustar este parámetro en futuras iteraciones.

Las métricas de desempeño fueron:
	•	R²: para evaluar la proporción de varianza explicada.
	•	RMSE: como medida directa del error en puntos porcentuales de tasa.

El árbol de decisión entrega interpretabilidad y estructura lógica simple, lo que permite comprender mejor cómo cada variable impacta en la predicción del T10_yield.


# ---------------------------------------------------------
# MODELO 3: RANDOM FOREST REGRESSOR
# ---------------------------------------------------------

El modelo de Random Forest Regressor se incorporó como extensión del árbol de decisión, mediante la técnica de bagging (bootstrap aggregation), con el objetivo de mejorar la capacidad predictiva y reducir el sobreajuste. Se utilizó un conjunto de 100 árboles independientes (n_estimators=100), entrenados sobre subconjuntos aleatorios del dataset.

Aunque Random Forest no requiere escalado de variables, se mantuvo la estandarización (StandardScaler) para conservar coherencia metodológica en la comparación de modelos.

La división entrenamiento/test se hizo de forma temporal (80/20), respetando el orden cronológico de las observaciones para prevenir fugas de información.

Las métricas utilizadas fueron:
	•	R²: Proporción de la varianza del T10_yield explicada por el modelo.
	•	RMSE: Error cuadrático medio de predicción.

Este modelo combina robustez, buena capacidad predictiva y una interpretación relativa de la importancia de cada variable, lo que lo convierte en una excelente referencia dentro del conjunto de modelos a evaluar.

# ---------------------------------------------------------
# MODELO 4: XGBOOST REGRESSOR
# ---------------------------------------------------------

En esta etapa se implementó XGBoost, una técnica de Gradient Boosting altamente eficiente, desarrollada para optimizar tanto el rendimiento predictivo como la velocidad de cómputo. Este modelo es particularmente útil en contextos financieros debido a su capacidad para capturar relaciones no lineales y efectos de interacción entre variables.

Aunque XGBoost no requiere normalización explícita, se optó por mantener la estandarización previa de las variables (StandardScaler) para asegurar consistencia metodológica a lo largo de todos los modelos probados. Asimismo, se respetó la división temporal del dataset (80/20) para prevenir fugas de información (data leakage) propias de las series de tiempo.

Los principales hiperparámetros seleccionados fueron:
	•	n_estimators = 100: cantidad de árboles de decisión a entrenar.
	•	learning_rate = 0.1: tasa de aprendizaje para la corrección de errores.
	•	max_depth = 4: profundidad máxima permitida en cada árbol.

Las métricas de evaluación utilizadas fueron:
	•	R² (Coeficiente de Determinación): permite evaluar cuánta varianza del T10_yield es explicada por el modelo.
	•	RMSE (Root Mean Squared Error): mide el error promedio de predicción en las mismas unidades del rendimiento.

XGBoost representa un modelo de última generación, optimizado para tareas de regresión complejas. En este proyecto, se incluye como una de las alternativas más robustas, capaz de adaptarse a patrones ocultos y relaciones complejas en los datos macroeconómicos.

# ---------------------------------------------------------
# MODELO 5: RED NEURONAL MULTICAPA (MLPREGRESSOR)
# ---------------------------------------------------------

Para capturar relaciones altamente no lineales y de múltiples interacciones entre variables, se entrenó una Red Neuronal Multicapa (MLPRegressor) con arquitectura (64, 32) en sus capas ocultas. Este modelo representa una aproximación flexible y no paramétrica, ideal para problemas con alta complejidad estructural como la predicción de tasas de interés.

A diferencia de los modelos basados en árboles, las redes neuronales requieren estandarización estricta de las variables de entrada. Se utilizó StandardScaler para garantizar que todas las variables tuvieran media cero y desviación estándar uno.

La división entrenamiento/test se realizó respetando la estructura temporal de los datos (80% de entrenamiento, 20% de test), evitando así cualquier posible fuga de información.

Los hiperparámetros considerados fueron:
	•	hidden_layer_sizes=(64, 32): dos capas ocultas con 64 y 32 neuronas respectivamente.
	•	max_iter=500: número máximo de iteraciones de entrenamiento.
	•	random_state=42: para asegurar reproducibilidad.

Las métricas de evaluación incluyeron:
	•	R²: proporción de varianza del T10_yield explicada por el modelo.
	•	RMSE: error cuadrático medio en la predicción.

Aunque las redes neuronales requieren mayor tiempo de entrenamiento, permiten modelar patrones no evidentes que otros modelos pueden pasar por alto, convirtiéndose en una herramienta potente para evaluación comparativa.
