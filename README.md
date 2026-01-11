# TFG – Modelado de la Confianza Humana mediante Neurodispositivos

**Autora:** Lucía Rebolledo Romillo  
**Trabajo:** Trabajo Fin de Grado  
**Grado:** Grado en Ciencia de Datos e Inteligencia Artificial
**Universidad:** Universidad Politecnica de Madrid
**Curso académico:** 2025–2026



Este repositorio contiene el código y los análisis desarrollados en el Trabajo Fin de Grado titulado **“Modelado de la Confianza Humana mediante Neurodispositivos”**.  
El objetivo principal del trabajo es analizar la relación entre señales electroencefalográficas (EEG) y los niveles de confianza del usuario, empleando técnicas de análisis estadístico, aprendizaje automático y métodos de explicabilidad de modelos.

El repositorio está organizado por bloques de análisis, siguiendo la misma estructura conceptual que la memoria del TFG.



## Estructura del repositorio

### Análisis Estadístico
Contiene un notebook dedicado al análisis descriptivo y estadístico de las señales EEG y de las variables derivadas.  
Incluye estudios exploratorios y comparaciones entre condiciones experimentales.

---

### Análisis Temporal
Incluye el estudio de las señales EEG en el dominio temporal, analizando su evolución a lo largo del tiempo bajo las distintas condiciones experimentales (revisar)

Este análisis sirve como apoyo al estudio posterior en el dominio frecuencial y espacial.

---

### Análisis Topográfico
Contiene el código para la generación y análisis de mapas topográficos EEG.  
Se representan distribuciones espaciales de potencia cerebral por bandas de frecuencia y condiciones experimentales, tanto a nivel individual como grupal.

---

### Aprendizaje No Supervisado
Incluye los experimentos de aprendizaje no supervisado aplicados a las características extraídas de las señales EEG.  
El objetivo es explorar la posible existencia de patrones cerebrales relacionados con la confianza sin utilizar etiquetas predefinidas.

---

### Aprendizaje Supervisado
Contiene los modelos de aprendizaje automático supervisado desarrollados para la clasificación de la confianza del usuario a partir de señales EEG.  
Se incluye la preparación de datos, la codificación de la variable Trust, el entrenamiento de los modelos y la evaluación de su rendimiento.

Este bloque constituye el núcleo del análisis predictivo del trabajo.

---

### SHAP
Incluye el análisis de explicabilidad de los modelos supervisados mediante valores SHAP.  
Se estudia la contribución de las distintas características (canales, regiones cerebrales y bandas de frecuencia) en la toma de decisiones de los modelos.

---

## Requisitos y entorno
El código ha sido desarrollado principalmente en Python, utilizando notebooks de Jupyter y archivos .py.  
Las principales librerías empleadas incluyen:
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- SHAP  

Los datos EEG en bruto no se incluyen en el repositorio por motivos de tamaño y privacidad.

---

## Uso del repositorio
Cada carpeta contiene su propio `readME` con una breve explicación del contenido.(revisar)

Aunque los análisis pueden ejecutarse de forma independiente, se recomienda seguir el flujo general del trabajo:
1. Análisis estadístico y temporal  
2. Análisis topográfico  
3. Aprendizaje no supervisado  
4. Aprendizaje supervisado  
5. Análisis de explicabilidad (SHAP)

## Licencia
Este repositorio se proporciona con fines académicos y de investigación.
