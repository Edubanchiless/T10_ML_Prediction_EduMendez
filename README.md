# T10_ML_Prediction_EduMendez

Proyecto de predicción del rendimiento del Treasury 10Y usando modelos de Machine Learning para el curso EAE3709 (PUC, 2025).

# ---------------------------------------------------------
# MOTIVACIÓN DEL PROYECTO
# ---------------------------------------------------------

El rendimiento del bono del Tesoro estadounidense a 10 años (T10Y) es uno de los activos más influyentes a nivel global. Sirve como referencia para las tasas de interés de largo plazo, afecta directamente la valuación de activos financieros, la prima de riesgo y las decisiones de política monetaria de los bancos centrales. En economías desarrolladas y emergentes, su comportamiento tiene implicancias sistémicas para los portafolios institucionales y las condiciones financieras.

Modelar y predecir esta tasa representa un desafío debido a la naturaleza dinámica y multifactorial de los mercados financieros. Investigaciones recientes (Zhang et al., 2023; Goto & Xu, 2024) han demostrado que la aplicación de algoritmos de Machine Learning (ML) permite superar las limitaciones de los modelos lineales clásicos, capturando relaciones no lineales, interacción de variables, y efectos temporales rezagados sin requerir fuertes supuestos estructurales.

Este proyecto toma como referencia ese enfoque moderno para evaluar distintas técnicas de ML supervisado (Regresión Lineal, Árboles de Decisión, Random Forest, XGBoost y MLP) sobre un conjunto de datos macrofinancieros diarios. Se busca determinar cuáles modelos son más efectivos para anticipar variaciones en el T10Y, explorando el potencial predictivo de herramientas de aprendizaje automático en escenarios de alta volatilidad.

La motivación central es aportar evidencia empírica desde un enfoque reproducible y práctico, aprovechando datos públicos de alta frecuencia (FRED y Yahoo Finance), con el objetivo de construir una base analítica útil tanto para decisiones académicas como profesionales en el ámbito económico-financiero.

# ---------------------------------------------------------
# DATASET Y VARIABLES UTILIZADAS
# ---------------------------------------------------------

Para este estudio se construyó un dataset consolidado a partir de fuentes públicas de alta frecuencia: **FRED** (Federal Reserve Economic Data) y **Yahoo Finance**. La periodicidad diaria permite capturar variaciones de corto plazo en las condiciones macrofinancieras que podrían afectar el rendimiento del bono del Tesoro a 10 años (T10Y).

Las variables fueron seleccionadas con base en literatura especializada y criterios de relevancia económica. Cada una representa distintos canales de transmisión que pueden incidir en las tasas de interés de largo plazo:

| Variable         | Descripción                                      | Fuente         |
|------------------|--------------------------------------------------|----------------|
| `T10_yield`      | Rendimiento diario del bono del Tesoro a 10 años | FRED (DGS10)   |
| `VIX`            | Volatilidad implícita esperada del mercado       | FRED (VIXCLS)  |
| `DXY`            | Índice del dólar estadounidense                  | FRED (DTWEXBGS)|
| `Oil_WTI`        | Precio del petróleo crudo WTI                    | FRED (DCOILWTICO) |
| `TED_Spread`     | Indicador de riesgo de crédito interbancario     | FRED (TEDRATE) |
| `SP500`          | Índice bursátil S&P 500 (precio de cierre)       | Yahoo Finance (`^GSPC`) |

Las variables se unificaron mediante un `join` temporal tipo *inner*, eliminando fechas sin datos completos para asegurar consistencia. El resultado es una muestra limpia con observaciones en las que todas las variables están disponibles de forma simultánea.

Este conjunto de datos fue posteriormente exportado como `sabana_macro_t10.csv` para facilitar la reproducción del análisis y entrenamiento de modelos.

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
# ANÁLISIS AVANZADO (STYLIZED FACTS)
# ---------------------------------------------------------

Antes de aplicar modelos predictivos, se evaluó la estacionariedad de las variables mediante una estrategia robusta que combina los tests de Dickey-Fuller Aumentado (ADF) y KPSS. Esta doble prueba permite identificar de manera más confiable la presencia de raíces unitarias o estacionariedad en nivel, evitando errores de especificación comunes en series económicas.

El test ADF parte de la hipótesis nula de no-estacionariedad, mientras que el KPSS asume como nula la estacionariedad. Cuando ambos tests concuerdan, se puede tomar una decisión clara respecto a transformaciones necesarias. En caso de contradicción, se recurre a criterios económicos y visuales.

Complementariamente, se graficó la media móvil y la desviación estándar móvil de la tasa del Treasury a 10 años (T10Y), con el fin de analizar su comportamiento dinámico. Este enfoque permite detectar inestabilidades, cambios de régimen o períodos de alta volatilidad que pueden afectar la predicción. Este tipo de diagnóstico es común en la literatura macro-financiera aplicada y se utiliza para justificar transformaciones adicionales o modelos adaptativos.

# ---------------------------------------------------------
# TRANSFORMACIONES PARA ESTACIONARIEDAD
# ---------------------------------------------------------

Tras la evaluación de estacionariedad con ADF y KPSS, se aplicaron transformaciones a las series temporales para cumplir con los supuestos necesarios del modelamiento predictivo.

Cuando fue posible, se utilizó la diferencia logarítmica (`log-diff`), transformación estándar en economía y finanzas para capturar tasas de crecimiento. En variables con ceros o valores negativos, se aplicó la diferencia simple (`diff`) para evitar distorsiones matemáticas.

Este tratamiento garantiza que las series utilizadas en los modelos posteriores no presenten raíces unitarias ni tendencias espurias, mejorando la validez de los resultados. El conjunto transformado fue exportado como `sabana_transformada.csv` y será la base para el entrenamiento de modelos supervisados.

# ---------------------------------------------------------
# VALIDACIÓN POST-TRANSFORMACIÓN Y DICCIONARIO DE VARIABLES
# ---------------------------------------------------------

Una vez aplicadas las transformaciones de estacionariedad, se validó nuevamente cada serie temporal utilizando dos pruebas complementarias: **ADF (Augmented Dickey-Fuller)** y **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**. Esta doble validación refuerza la robustez metodológica al contrastar hipótesis opuestas (ADF: presencia de raíz unitaria / KPSS: estacionariedad).

Los resultados se sistematizaron en un cuadro resumen, clasificando las variables en tres categorías: estacionarias, conflictivas (ADF vs KPSS) y no estacionarias. Aquellas que no cumplieron los criterios fueron ajustadas con transformaciones adicionales, como diferenciación o log-diferenciación.

Junto con ello, se construyó un **diccionario de transformaciones**, detallando la variable original, la transformación aplicada y la justificación económica detrás de cada decisión. Esta documentación asegura la **trazabilidad y reproducibilidad** del trabajo, elementos fundamentales en investigación de alto estándar.

El resultado final fue el archivo `sabana_transformada.csv`, que se utilizará como base para los modelos de predicción del rendimiento del Treasury a 10 años.

# ---------------------------------------------------------
# CONSTRUCCIÓN DEL SET DE ENTRENAMIENTO SUPERVISADO
# ---------------------------------------------------------

Para modelar el rendimiento del Treasury a 10 años como un problema de regresión supervisada, se definió como **variable objetivo (`y`)** su versión transformada (`log-diff`). Esto permite modelar tasas de variación porcentual, lo cual es estándar en estudios macrofinancieros.

Las **variables explicativas (`X`)** corresponden al resto de las series macroeconómicas transformadas, incluyendo indicadores como el VIX, DXY, precio del petróleo (WTI), TED Spread e índice S&P 500.

La partición del dataset se realizó en un 80% para entrenamiento y un 20% para testeo, respetando el orden temporal de las observaciones para evitar look-ahead bias. Este conjunto se utilizará en las siguientes secciones para entrenar y evaluar distintos modelos de machine learning.

# ---------------------------------------------------------
# GENERACIÓN DE VARIABLES LAG (1 A 5 DÍAS)
# ---------------------------------------------------------

Como primer paso del Feature Engineering, se generaron rezagos (lags) de 1 a 5 días para todas las variables del dataset transformado.

Esta técnica permite capturar dinámicas temporales clave, como efectos retardados de shocks macroeconómicos o persistencia en las tasas de interés. Incorporar lags es especialmente útil en modelos predictivos donde se desea que el modelo aprenda patrones de memoria de corto plazo.

El nuevo dataset generado fue guardado como `sabana_transformada_lags.csv` y constituye la base para aplicar rolling statistics e interacciones complejas.

# ---------------------------------------------------------
# CONSTRUCCIÓN DEL SET DE ENTRENAMIENTO SUPERVISADO
# ---------------------------------------------------------

Se definió como variable objetivo (`y`) la tasa de crecimiento diaria (log-diferenciada) del rendimiento del Treasury a 10 años (`T10_yield_logdiff`). Esta decisión se basa en su relevancia financiera como referencia global del costo del dinero a largo plazo y su sensibilidad a variables macroeconómicas.

El conjunto de predictores (`X`) incluye rezagos (lags) de todas las variables transformadas, permitiendo capturar relaciones dinámicas de corto plazo entre factores económicos y la evolución del T10Y.

Este set supervisado se utilizará en las siguientes etapas para entrenar modelos de aprendizaje automático y evaluar su poder predictivo sobre el mercado de renta fija.

# ---------------------------------------------------------
# FEATURES DE ESTADÍSTICAS MÓVILES: ROLLING MEAN Y STD
# ---------------------------------------------------------

Como parte del enriquecimiento del set supervisado, se calcularon estadísticos móviles (media y desviación estándar) para cada predictor económico, utilizando ventanas de 20 y 60 días. Estas ventanas permiten capturar dinámicas de corto y mediano plazo, respectivamente.

La media móvil refleja tendencias persistentes, mientras que la desviación estándar móvil permite detectar cambios en la volatilidad de los indicadores.

Esta práctica es común en finanzas cuantitativas, ya que muchos modelos de predicción incorporan información sobre la dirección y la estabilidad reciente de las series.

El nuevo dataset con estas variables adicionales se exportó como `sabana_transformada_lags_rolling.csv`, y será utilizado en los modelos posteriores.

# ---------------------------------------------------------
# GENERACIÓN DE VARIABLES DE INTERACCIÓN ECONÓMICA
# ---------------------------------------------------------

Se incorporaron interacciones cruzadas entre variables macroeconómicas con fundamentos teóricos relevantes. Estas combinaciones permiten capturar efectos conjuntos entre shocks económicos, como volatilidad y tipo de cambio (VIX × SP500), o precios de commodities con tipo de cambio (Oil × DXY).

Al incluir estos términos cruzados, se enriquece el espacio de representación del modelo y se facilita la captura de no linealidades económicas relevantes. Esta técnica es comúnmente utilizada en modelos estructurales y machine learning financiero.

El dataset final con lags, estadísticas móviles y variables cruzadas fue exportado como `sabana_final_modelado.csv`.

# ---------------------------------------------------------
# ANÁLISIS DE MULTICOLINEALIDAD – HEATMAP Y VIF
# ---------------------------------------------------------

Antes de realizar selección de variables o entrenar modelos sensibles a redundancia, se evaluó la multicolinealidad entre predictores.

Primero se construyó un **heatmap de correlaciones altas** (|r| > 0.85), lo que permite identificar variables altamente correlacionadas que podrían distorsionar la interpretación o inducir overfitting.

Luego se calculó el **Variance Inflation Factor (VIF)** sobre un subconjunto representativo de las variables transformadas. Esta métrica cuantifica cuánta varianza de un predictor es explicada por otros, siendo útil para eliminar redundancias antes del modelamiento.

Estos análisis permiten depurar el espacio de predictores y construir modelos más robustos y estables.


# ---------------------------------------------------------
# ANÁLISIS DE MULTICOLINEALIDAD – CLASIFICACIÓN DE VARIABLES
# ---------------------------------------------------------

A partir del dataset sabana_final_modelado.csv, se clasificaron las variables en cuatro grupos funcionales: variables originales, rezagos (lags), estadísticas móviles (rolling features) y variables de interacción. Esta organización permite aplicar análisis dirigidos de multicolinealidad, evitando redundancia excesiva y mejorando la interpretación económica y estadística de los modelos.

La clasificación es clave para aplicar posteriormente técnicas como VIF, selección de variables o regularización, optimizando la parsimonia sin perder información relevante.

# ---------------------------------------------------------
# ANÁLISIS DE MULTICOLINEALIDAD – HEATMAP Y VIF
# ---------------------------------------------------------

Antes de realizar selección de variables o entrenar modelos sensibles a redundancia, se evaluó la multicolinealidad entre predictores.

Primero se construyó un heatmap de correlaciones altas (|r| > 0.85), lo que permite identificar variables altamente correlacionadas que podrían distorsionar la interpretación o inducir overfitting.

Luego se calculó el Variance Inflation Factor (VIF) sobre un subconjunto representativo de las variables transformadas. Esta métrica cuantifica cuánta varianza de un predictor es explicada por otros, siendo útil para eliminar redundancias antes del modelamiento.

Estos análisis permiten depurar el espacio de predictores y construir modelos más robustos y estables.

# ---------------------------------------------------------
# SELECCIÓN AUTOMÁTICA DE VARIABLES – LASSO
# ---------------------------------------------------------

Para reducir la dimensionalidad del set de entrenamiento sin perder capacidad predictiva, se aplicó una regresión Lasso con validación cruzada (LassoCV). Esta técnica impone una penalización L1 que fuerza coeficientes irrelevantes a cero, facilitando la selección de predictores significativos.

El Lasso fue entrenado sobre predictores estandarizados, y se seleccionaron solo aquellas variables con coeficientes distintos de cero. El conjunto resultante fue exportado como `sabana_lasso_selected.csv` y servirá como base para entrenar modelos más parsimoniosos y robustos.

Esta técnica es particularmente útil para evitar overfitting en datasets con alta dimensionalidad y correlaciones cruzadas.

# ---------------------------------------------------------
# SELECCIÓN DE VARIABLES – RFE CON RANDOM FOREST
# ---------------------------------------------------------

Para complementar la selección vía Lasso, se aplicó Recursive Feature Elimination (RFE) usando un Random Forest como modelo base. Esta técnica elimina recursivamente variables menos relevantes hasta encontrar un subconjunto óptimo de predictores.

Se utilizó validación cruzada temporal (TimeSeriesSplit) para respetar la estructura secuencial de los datos y evitar look-ahead bias. Se seleccionaron las 20 variables más relevantes, las cuales fueron exportadas como `sabana_rfe_selected.csv`.

Este paso asegura que las variables finales combinen robustez estadística y capacidad explicativa frente a relaciones no lineales.

# ---------------------------------------------------------
# SELECCIÓN DE VARIABLES – RFE CON RANDOM FOREST (OPTIMIZADO)
# ---------------------------------------------------------

Se aplicó el método **Recursive Feature Elimination (RFE)** utilizando un modelo base de Random Forest para seleccionar las variables más relevantes dentro del conjunto reducido por Lasso.

Debido al alto número de predictores (≈180) y la alta carga computacional que conlleva este tipo de modelos (entrenamiento iterativo de miles de árboles), se optó por una versión **computacionalmente optimizada** del algoritmo:

- Se redujo el número de árboles por bosque (`n_estimators=50`).
- Se aumentó el tamaño del paso (`step=5`) para eliminar múltiples variables por iteración.

Esta estrategia permite **preservar la robustez del análisis** sin comprometer excesivamente los recursos computacionales. Las 20 variables seleccionadas representan aquellas con mayor poder explicativo para predecir la variación de la tasa del Treasury a 10 años (`T10_yield_logdiff`).

El dataset final con las variables seleccionadas fue exportado como `sabana_rfe_selected.csv` y será la base para la construcción de modelos supervisados.



# ---------------------------------------------------------
# DEFINICIÓN FINAL DEL DATASET Y DIVISIÓN TEMPORAL
# ---------------------------------------------------------

Con las 20 variables seleccionadas mediante RFE, se definió la estructura del dataset supervisado para modelamiento de series temporales.

Se mantuvo como variable objetivo el crecimiento logarítmico de la tasa del Treasury a 10 años (`T10_yield_logdiff`), con un horizonte de predicción de t+1. Esto permite captar la dinámica diaria del rendimiento y evaluar shocks económicos de corto plazo.

La separación entre datos de entrenamiento y prueba respeta la secuencia temporal para evitar fuga de información, utilizando los últimos 20% de los datos como test set. Esta metodología simula un entorno realista de inversión, donde solo se dispone de información pasada para realizar proyecciones futuras.


# ---------------------------------------------------------
# VALIDACIÓN DE VARIABLES – FEATURE IMPORTANCE CON XGBOOST
# ---------------------------------------------------------

Para robustecer la selección de predictores, se utilizó un modelo de **XGBoost Regressor** para evaluar la importancia relativa de cada variable dentro del conjunto reducido. Esta técnica aprovecha la capacidad del algoritmo de ensamblado para capturar relaciones no lineales y jerarquizar variables según su contribución predictiva.

Se visualizaron las 20 variables más influyentes, permitiendo validar si las seleccionadas por RFE o Lasso coinciden con aquellas que realmente aportan al poder explicativo del modelo.

El resultado se exportó como `xgboost_feature_importance.csv` para su revisión y uso posterior.

# ---------------------------------------------------------
# INTERPRETABILIDAD – SHAP VALUES CON XGBOOST
# ---------------------------------------------------------

Para comprender cómo cada variable influye en las predicciones del modelo, se calcularon **SHAP values**, técnica que descompone cada predicción en contribuciones individuales de los predictores.

Se utilizó un modelo XGBoost entrenado con las 20 variables seleccionadas vía RFE y se aplicó `TreeExplainer`, método eficiente para modelos de árboles.

Los resultados se resumen en dos gráficos clave:
- **Gráfico de barras**: muestra las variables más influyentes en promedio.
- **Gráfico de dispersión**: revela cómo varía el impacto de cada variable según el valor observado.

Ambas visualizaciones permiten interpretar el modelo de forma transparente, facilitando insights económicos y respaldando la robustez del enfoque predictivo. Archivos exportados: `shap_summary_bar.png` y `shap_summary_dot.png`.

# ---------------------------------------------------------
# MODELADO PREDICTIVO – BENCHMARK Y MODELOS AVANZADOS
# ---------------------------------------------------------

Este bloque implementa distintos modelos de regresión para predecir el retorno logarítmico del Treasury a 10 años (`T10_yield_logdiff`) a partir de un conjunto de variables macroeconómicas seleccionadas por RFE. 

Se incluyen tanto modelos lineales tradicionales como algoritmos no paramétricos, con el objetivo de comparar desempeño predictivo y estabilidad en series temporales:

- OLS (regresión lineal clásica)
- Ridge y Lasso (penalización para evitar sobreajuste)
- Random Forest (modelo de árbol con bagging)
- XGBoost (modelo de boosting optimizado)

Todos los modelos se evalúan con una validación temporal consistente con la naturaleza de los datos financieros, asegurando la no filtración de información futura.

# ---------------------------------------------------------
# MODELADO – REGRESIÓN LINEAL BASELINE (BENCHMARK)
# ---------------------------------------------------------

Se entrena un modelo de regresión lineal como punto de partida base para la predicción del rendimiento del Treasury a 10 años (`T10_yield_logdiff`). Esta etapa cumple el rol de **benchmark interpretativo**, permitiendo comparar la ganancia marginal de modelos no lineales posteriores.

- Se realiza una validación temporal (**train-test split secuencial**) para respetar la estructura temporal del problema.
- Se calculan métricas estándar: MAE, RMSE y R² para evaluar desempeño.
- Se visualiza la comparación entre valores reales y predichos en el conjunto de test.

Esta etapa es fundamental para establecer un piso comparativo y validar si los modelos avanzados efectivamente entregan mejoras significativas.

Archivos exportados: `linear_regression_summary.png`

# ---------------------------------------------------------
# MODELADO – XGBOOST CON TUNING Y MÉTRICAS EXTENDIDAS
# ---------------------------------------------------------

Se entrena un modelo `XGBoost` sobre el conjunto de datos reducido por RFE, buscando capturar no linealidades y relaciones complejas en la evolución del rendimiento del Treasury a 10 años (`T10_yield_logdiff`).

Se utiliza una configuración optimizada del modelo con `learning_rate` moderado, profundidad controlada y validación temporal. Se reportan métricas estándar (MAE, RMSE, R²) tanto en entrenamiento como en testeo, junto con un gráfico de comparación visual entre predicciones y valores reales.

Esta etapa permite evaluar la ganancia predictiva al pasar de un modelo lineal a uno no lineal robusto, incorporando además resultados listos para interpretación y exportación.

# ---------------------------------------------------------
# MODELADO – CLASIFICACIÓN BINARIA: ↑ / ↓ T10Y
# ---------------------------------------------------------

Se reformuló el problema como una tarea de clasificación binaria: predecir si el rendimiento del Treasury a 10 años (`T10_yield_logdiff`) subirá (`1`) o bajará (`0`) en el siguiente período.

Este enfoque es útil para estrategias de dirección de mercado o decisiones de cobertura. Se entrenó un modelo `XGBoostClassifier` sobre las 20 variables seleccionadas por RFE, y se reportan métricas de precisión, matriz de confusión y curva ROC.

Este tipo de modelamiento complementa la regresión tradicional, permitiendo entender mejor los factores que anticipan cambios de dirección en tasas de interés de largo plazo.

# ---------------------------------------------------------
# MODELADO – REGRESIÓN RIDGE
# ---------------------------------------------------------

Se aplicó una regresión lineal con regularización L2 (Ridge) sobre las variables seleccionadas por RFE. Este tipo de modelo es útil cuando se trabaja con muchas variables correlacionadas, ya que penaliza los coeficientes grandes y ayuda a reducir el overfitting.

El modelo fue evaluado con métricas estándar (MAE, RMSE, R²) y se comparó su desempeño en conjunto de entrenamiento y testeo temporal.

Esta aproximación sirve como línea base interpretativa y permite contrastar con modelos no lineales como XGBoost.

# ---------------------------------------------------------
# MODELADO – REGRESIÓN XGBOOST FINAL
# ---------------------------------------------------------

Se entrena un modelo `XGBoost Regressor` utilizando las variables seleccionadas por RFE. Este algoritmo de boosting de árboles es potente para capturar relaciones no lineales y efectos de interacción entre predictores macroeconómicos.

El modelo se ajustó con hiperparámetros conservadores y se evaluó con métricas de desempeño (MAE, RMSE, R²) sobre conjuntos de entrenamiento y testeo temporal. Este modelo representa el baseline no lineal robusto para predecir el retorno del Treasury a 10 años (`T10_yield_logdiff`).

Las predicciones generadas pueden ser utilizadas para trading signals, análisis de sensibilidad o como input en sistemas de asset allocation.

# ---------------------------------------------------------
# MODELADO – RIDGE REGRESSION MULTIVARIADA
# ---------------------------------------------------------

Se implementa una regresión Ridge sobre el conjunto reducido por RFE, ideal para series macroeconómicas con muchas variables correlacionadas. Este modelo penaliza grandes coeficientes, lo que estabiliza el ajuste y reduce el riesgo de overfitting.

La regresión se entrena utilizando validación temporal (train/test split) y se evalúan sus predicciones con métricas estándar (MAE, RMSE, R²). Esto proporciona una línea base lineal robusta contra la cual comparar modelos más complejos como XGBoost.

Los coeficientes obtenidos ofrecen una interpretación directa del impacto marginal de cada variable sobre el rendimiento del Treasury (`T10_yield_logdiff`).

# ---------------------------------------------------------
# MODELADO – LASSO REGRESSION MULTIVARIADA
# ---------------------------------------------------------

Se implementa una regresión Lasso como modelo base adicional. A diferencia de Ridge, Lasso realiza **selección automática de variables** al penalizar los coeficientes, reduciendo algunos a cero.

Esto permite identificar predictores clave de forma parsimoniosa, manteniendo solo aquellos con mayor poder explicativo sobre la evolución de `T10_yield_logdiff`.

El modelo se evalúa con métricas clásicas (MAE, RMSE, R²) bajo una división temporal 80/20, y se exportan las predicciones para análisis comparativo posterior.

# ---------------------------------------------------------
# MODELADO – RANDOM FOREST REGRESSOR
# ---------------------------------------------------------

Se entrena un modelo de Random Forest para predecir la variación logarítmica de la tasa del Treasury a 10 años (`T10_yield_logdiff`). Este algoritmo es robusto a no linealidades y a variables irrelevantes, por lo que se espera que capte interacciones y relaciones complejas entre predictores económicos.

Se utiliza un split temporal 80/20, y se evalúan métricas clave de desempeño. Las predicciones sobre el conjunto de test se exportan para futura comparación con otros modelos.

# ---------------------------------------------------------
# MODELADO – ELASTICNET REGRESSION
# ---------------------------------------------------------

Se entrena un modelo de regresión ElasticNet sobre el dataset con variables seleccionadas por RFE. Esta técnica combina penalizaciones L1 y L2, lo que la hace útil frente a predictores correlacionados y modelos con regularización balanceada.

El ajuste de hiperparámetros (`alpha`, `l1_ratio`) se realiza con validación cruzada sobre un esquema temporal. Se comparan métricas MAE, RMSE y R², y se exportan las predicciones para benchmarking con otros modelos.

# ---------------------------------------------------------
# MODELADO – LIGHTGBM REGRESSOR
# ---------------------------------------------------------

Se entrena un modelo de gradiente boosting con LightGBM, reconocido por su eficiencia computacional y buen desempeño en datasets con alto número de variables, como en nuestro caso tras la selección con RFE.

Se aplicó normalización estándar, división temporal y evaluación con métricas MAE, RMSE y R². LightGBM es especialmente útil en contextos financieros por su capacidad de manejar relaciones no lineales y su robustez ante colinealidad moderada.

Este modelo se añade al pipeline como alternativa base al enfoque XGBoost, permitiendo luego una comparación cuantitativa entre ambos. Se exportan predicciones y métricas para consolidar la evaluación final.

# ---------------------------------------------------------
# MODELADO – CATBOOST REGRESSOR
# ---------------------------------------------------------

Se incorpora un modelo basado en boosting con CatBoost, optimizado para evitar codificación explícita de variables categóricas (aunque en este caso todas las variables son numéricas). Es altamente eficiente y robusto ante overfitting, especialmente útil en datasets con colinealidad y tamaños medianos.

El entrenamiento sigue la misma lógica que XGBoost y LightGBM: división temporal, normalización y evaluación con MAE, RMSE y R². Esto permite mantener consistencia en la comparación entre modelos.

CatBoost destaca por su estabilidad en predicciones y buen desempeño en entornos financieros con ruido moderado.

# ---------------------------------------------------------
# COMPARACIÓN DE MODELOS – XGBOOST VS LIGHTGBM VS CATBOOST
# ---------------------------------------------------------

Se compararon los tres modelos entrenados (XGBoost, LightGBM y CatBoost) mediante sus principales métricas de desempeño sobre el conjunto de test:

- **MAE (Mean Absolute Error)**: error absoluto promedio.
- **RMSE (Root Mean Squared Error)**: raíz del error cuadrático medio.
- **R² (Coeficiente de Determinación)**: varianza explicada por el modelo.

Cada métrica se graficó por separado para facilitar su interpretación y detectar diferencias finas en precisión y robustez. Esta evaluación es crucial para seleccionar el modelo con mejor desempeño económico y técnico.

# ---------------------------------------------------------
# CONCLUSIONES FINALES Y HALLAZGOS ECONÓMICOS
# ---------------------------------------------------------

A partir del modelado multivariado del rendimiento del Treasury a 10 años (`T10_yield_logdiff`), se obtuvieron los siguientes insights clave:

## Variables más relevantes según múltiples enfoques

- **Persistencia temporal**: Los lags del rendimiento (`T10_yield_logdiff_lag1`, `lag2`, etc.) fueron los predictores más consistentes y robustos.
- **Factores de riesgo global**: Variables como `DXY_logdiff`, `SP500_logdiff`, `VIX_logdiff` y `Oil_WTI_diff` mostraron relación significativa, revelando sensibilidad del T10 a shocks externos.
- **Interacción cruzada relevante**: La variable `VIX_x_SP500_lag1` capturó efectos no lineales asociados a incertidumbre bursátil y su relación con la política monetaria.

## Evaluación de modelos

| Modelo      | MAE     | RMSE    | R²     | Comentario                                  |
|-------------|---------|---------|--------|---------------------------------------------|
| **XGBoost** | ✅ Bajo | ✅ Bajo | ✅ Alto | Mejor balance general entre error y robustez |
| **LightGBM**| Muy bueno | Ligeramente inferior | Bueno | Rápido y eficiente, gran alternativa        |
| **CatBoost**| Similar | Similar | Bueno  | Estable, ideal si hubiera variables categóricas |

> **XGBoost resultó el mejor modelo en precisión y generalización**, validado además por la interpretación SHAP.

## Interpretabilidad con SHAP

- Las variables seleccionadas por RFE fueron confirmadas como relevantes por SHAP.
- El gráfico de dependencia muestra cómo, por ejemplo, un aumento en `T10_yield_logdiff_lag1` tiende a incrementar la predicción actual, reflejando inercia estructural en los rendimientos.

## Conclusión económica

- La tasa del Treasury a 10 años presenta **dinámica autoregresiva fuerte**, pero también es sensible a shocks externos reflejados en volatilidad (`VIX`), expectativas bursátiles (`SP500`), condiciones monetarias globales (`DXY`) y riesgo sistémico (`TED Spread`).
- Modelos como **XGBoost**, combinados con técnicas de selección y explicación robustas (**Lasso**, **RFE**, **SHAP**), permiten **anticipar movimientos marginales** del T10 con precisión competitiva.
- Estos insights son clave para decisiones de **cobertura de duration**, construcción de portafolios de renta fija y análisis macroeconómico predictivo.

# ---------------------------------------------------------
# BIBLIOGRAFÍA
# ---------------------------------------------------------

- Bursac, Z., Gauss, C. H., Williams, D. K., & Hosmer, D. W. (2008). Purposeful selection of variables in logistic regression. *Source Code for Biology and Medicine, 3*(1), 17. https://doi.org/10.1186/1751-0473-3-17

- Buraschi, A., & Jiltsov, A. (2005). Inflation risk premia and the expectations hypothesis. *Journal of Financial Economics, 75*(2), 429–490. https://doi.org/10.1016/j.jfineco.2004.01.004

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software, 33*(1), 1–22. https://doi.org/10.18637/jss.v033.i01

- Goto, S., & Xu, Y. (2024). Yield curve forecasting using macroeconomic variables and machine learning. *Journal of International Money and Finance, 141*, 102024. https://doi.org/10.1016/j.jimonfin.2024.102024

- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems, 30*. https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Proceedings of the 31st International Conference on Neural Information Processing Systems* (pp. 4765–4774). https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems, 31*. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

- Zhang, Z., Chen, Z., & Liu, Y. (2023). A machine learning-based model for forecasting the US Treasury yield curve. *Applied Sciences, 15*(13), 6903. https://doi.org/10.3390/app15136903