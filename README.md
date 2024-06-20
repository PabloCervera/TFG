# Trabajo Fin de Grado

Explainable Boosting Machine in the prediction of Alzheimer’s disease conversion from Mild Cognitive Impairment using longitudinal data

## Descripción

Este proyecto tiene como objetivo predecir la conversión de pacientes con Deterioro Cognitivo Leve (MCI) a Enfermedad de Alzheimer (AD) utilizando el modelo Explainable Boosting Machine (EBM). El proyecto utiliza datos de la base de datos ADNI y se enfoca en la importancia de la explicabilidad en los modelos de inteligencia artificial en el ámbito médico.

## Contenido del Repositorio
.
├── data/ # Carpeta con los conjuntos de datos utilizados
├── README.md # Este archivo
├── imputer.py # Archivo de preprocesado de datos
└── train.py # Archivo de entrenamiento del modelo

## Requisitos previos

- Python 3.8

## Instalación de dependencias 

### Fichero imputer.py

pip install pandas scikit-learn imbalanced-learn missingpy

### Fichero train.py

pip install pandas seaborn matplotlib scikit-learn interpret 

## Uso

### Fichero imputer.py

py -3.8 imputer.py

### Fichero train.py
py -3.x imputer.py    #x = cualquier versión de Python a partir de la 3.8 incluida
