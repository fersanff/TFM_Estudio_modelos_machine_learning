# TFM Estudio modelos machine learning
Repositorio con el código del TFM de "Estudio de modelos de machine learning  para clasificación", este repositorio contiene todo el codigo que se utilizo para el desarrollo del TFM de "Estudio de modelos de machine learning para clasificación", en este punto se explicaran todos los puntos necesarios para entender la estructura de este proyecto, una nota importante es que en este fichero no se explicaran conceptos tecnicos de los modelos estos estan explicados en el pdf por lo que se anima a su lectura.
## Requerimientos
- MATLAB (R2021b or later)
- Libreria CVX
## Estructura de ficheros 
### Ejemplos
Este fichero contiene una prueba para verificar que la instalacion de CVX en Matlab ha sido correcta, se tiene que ejecutar y verificar.
### Modelos Hinge
Esta carpeta contiene todas las implimentaciones de los modelos Hinge, tanto las del SVM como las del PSVM. El codigo utilizado para el modelo SVM es **svm_dual_quadprog_hinge.m** y el codigo utilizado para los modelos PSVM es **psvm_dual_quadprog_hinge.m**.
### Modelos Pinball
Esta carpeta contiene todas las implimentaciones de los modelos Pinball, tanto las del SVM como las del PSVM. El codigo utilizado para el modelo SVM es **svm_soft_margin_quadprog.m** y el codigo utilizado para los modelos PSVM es **psvm_dual_quadprog_pinball.m**.
### Pruebas graficas
Incluye ficheros con pruebas graficas de los distintos modelos simplemente hay que ejecutarlas y saldran graficas con los resultados de cada modelo.
### Validaciones del modelo
Contiene todos los archivos utilizados para la validación de los modelos primero contiene la validacion cruzada de los datos sin ruido, todos los ficheros donde ponga "no noise" y luego estan los datos con ruido, donde pone "no noise". Estos ficheros dan unas matrices con los mejores resultados que se han almacenado en los ficheros con el nombre de cada modelo cada fila, columna y pagina corresponde con el rango de los hiperparametros de los modelos, hay luego otros ficheros que calculan la robustividad de cada modelo. Por ultimo hay funciones auxiliares para entre otras cosas sacar los mejores valores con sus hiperparamentros de los resultados de las matrices antes mencionadas.
