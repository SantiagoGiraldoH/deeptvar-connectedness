# DeepTVAR-Connectedness

Implementación del modelo **DeepTVAR** integrado con medidas de
**conectividad direccional generalizada** para el análisis de riesgo
sistémico en redes comerciales de gestión patrimonial.

Este repositorio contiene el código, los datos sintéticos de validación y
los scripts de análisis asociados al trabajo de tesis:

> **Aplicación de modelos de aprendizaje profundo en series de vectores
> autorregresivos (DeepTVAR) para estudiar la interacción entre flujos de
> inversión gestionados ante choques en el mercado.**
> Santiago Giraldo Henao — Maestría en Estadística, Universidad Nacional
> de Colombia, 2026.

---

## Resumen

El proyecto desarrolla y aplica un modelo de vectores autorregresivos con
parámetros variantes en el tiempo (TVP-VAR), estimado mediante una red
neuronal recurrente LSTM siguiendo el enfoque de Li y Yuan (2024). Sobre
los parámetros estimados se construyen medidas de conectividad direccional
basadas en la descomposición generalizada de la varianza del error de
pronóstico (GFEVD), siguiendo a Diebold y Yilmaz (2014) y Antonakakis et
al. (2017).

El aporte central del trabajo consiste en integrar el modelo DeepTVAR con
el marco de conectividad direccional, lo que permite analizar de forma
dinámica los canales de transmisión de volatilidad en una red de oficinas
comerciales de una firma de gestión patrimonial. La transformación de
Ansley-Kohn garantiza que los coeficientes generados por la red satisfagan
la condición de causalidad en cada instante, preservando la
interpretabilidad estadística necesaria para el análisis estructural.
