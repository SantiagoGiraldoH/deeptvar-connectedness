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


## Requisitos

### Python (entrenamiento del modelo)

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy, Pandas
- Optuna (búsqueda de hiperparámetros)

Instalación rápida:

```bash
pip install -r requirements.txt
```

### R (análisis de conectividad)

- R ≥ 4.2
- Paquetes: `reticulate`, `tvReg`, `shrinkTVPVAR`, `dplyr`, `ggplot2`,
  `kableExtra`

Instalación:

```r
install.packages(c("reticulate", "tvReg", "shrinkTVPVAR",
                   "dplyr", "ggplot2", "kableExtra"))
```

---

## Reproducción de los resultados

### 1. Validación con datos simulados

```bash
python training/train.py --config configs/simulation.yaml
Rscript connectedness/gfevd.R --input outputs/sim_params.npy
```

Reproduce la figura comparativa de conectividad direccional por pares
entre el proceso generador real, DeepTVAR y los métodos tradicionales.

### 2. Entrenamiento sobre los datos empíricos

> **Nota:** Los datos empíricos no se incluyen en el repositorio por
> razones de confidencialidad. El script asume un archivo `data/empirical/flows.csv`
> con cuatro columnas (flujos diarios por oficina) y una columna de fecha.

```bash
python training/hyperparams.py    # búsqueda con TPE (Optuna)
python training/train.py --config configs/empirical.yaml
```

### 3. Análisis de conectividad

```r
source("connectedness/gfevd.R")
source("connectedness/thresholds.R")
```

Genera el TCI, las medidas de conectividad direccional neta, las
matrices de conectividad por pares y los umbrales operativos.

---

## Datos

Los flujos de inversión utilizados en el análisis empírico fueron
proporcionados por **SURA Investments** en el marco del convenio
institucional que apoya el desarrollo de este trabajo. Por razones de
confidencialidad de la información comercial y de los clientes, **los
datos empíricos no están incluidos en el repositorio**.

Se incluyen, en cambio, los datos sintéticos generados a partir de un
proceso TVP-VAR(2) conocido, utilizados para la validación del modelo
(sección 7.2.1 del documento).

---

## Referencias principales

- Li, X. y Yuan, J. (2024). DeepTVAR: Deep learning for a time-varying
  VAR model with extension to integrated VAR. *International Journal of
  Forecasting*, 40, 1123–1133.
- Diebold, F. y Yilmaz, K. (2014). On the network topology of variance
  decompositions: Measuring the connectedness of financial firms.
  *Journal of Econometrics*, 182, 119–134.
- Antonakakis, N., Chatziantoniou, I. y Gabauer, D. (2017). Refined
  measures of dynamic connectedness based on time-varying parameter
  vector autoregressions. *Journal of Risk and Financial Management*,
  13(4), 84.
- Ansley, C. F. y Kohn, R. (1986). A note on reparameterizing a vector
  autoregressive moving average model to enforce stationarity. *Journal
  of Statistical Computation and Simulation*, 24, 99–106.

---

## Citación

Si utilizas este código o sus resultados, por favor cita:

```bibtex
@mastersthesis{giraldo2026deeptvar,
  author  = {Giraldo Henao, Santiago},
  title   = {Aplicación de modelos de aprendizaje profundo en series de
             vectores autorregresivos (DeepTVAR) para estudiar la
             interacción entre flujos de inversión gestionados ante
             choques en el mercado},
  school  = {Universidad Nacional de Colombia},
  year    = {2026},
  address = {Medellín, Colombia},
  type    = {Trabajo de grado de Maestría en Estadística}
}
```

---

## Licencia

Distribuido bajo licencia MIT. Consulta `LICENSE` para más detalles.

---

## Contacto

**Santiago Giraldo Henao**
Departamento de Estadística — Universidad Nacional de Colombia
[sgiraldoh@unal.edu.co](mailto:sgiraldoh@unal.edu.co)
