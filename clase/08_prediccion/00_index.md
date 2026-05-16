---
title: "Predicción: Taxonomía y Marco de 5 Dimensiones"
---

# Predicción: Taxonomía y Marco de 5 Dimensiones

Predecir es un problema fundamental en IA, pero no lo es todo — es una herramienta que ayuda a los agentes a entender el mundo. *¿De dónde viene el conocimiento? ¿Qué significa la incertidumbre? ¿Qué queremos realmente del modelo?*

Este módulo presenta un marco de **5 dimensiones ortogonales** para categorizar cualquier método de predicción. En lugar de memorizar una lista de algoritmos, aprenderás a ubicar cada método como un punto en un espacio 5-dimensional — y a elegir el punto correcto para tu problema.

## Contenido

| Sección | Tema | Idea clave |
|:------:|------|-----------|
| 8.1 | [El problema fundamental](01_el_problema_fundamental.md) | Qué es predecir, los 7 objetivos matemáticos, jerarquía y restricción |
| 8.2 | [Fuente del conocimiento (D1)](02_fuente_del_conocimiento.md) | Deductivo, inductivo, híbrido — el marco de 5 dimensiones |
| 8.3 | [Incertidumbre y objetivo (D2 + D3)](03_incertidumbre_y_objetivo.md) | Frequentist vs Bayesian + qué cantidad estimar |
| 8.4 | [Arquitectura y supuestos (D4 + D5)](04_arquitectura_y_supuestos.md) | Estructura entre variables + sesgo inductivo + Z vs L |
| 8.5 | [Atlas de métodos](05_atlas_de_metodos.md) | Matrices de métodos: supervisados, no supervisados, self-supervised, temporales, IA generativa |
| 8.6 | [Mapa y heurísticas](06_mapa_y_heuristicas.md) | Mapa visual, heurísticas de elección, 8 casos de estudio, reflexión final |

## Cómo correr el lab (para imágenes)

```bash
cd clase/08_prediccion
python lab_prediccion.py
```

## Objetivos de aprendizaje

Al terminar este módulo podrás:

1. **Definir** qué es predicción y distinguir los 7 objetivos matemáticos ($P(Y)$, $E[Y]$, $P(Y \mid X)$, $E[Y \mid X]$, $P(X)$, $\phi(X)$, $P(Y \mid do(X))$).
2. **Explicar** por qué los datos nunca son suficientes y por qué toda predicción requiere restricciones.
3. **Describir** las 5 dimensiones ortogonales que caracterizan cualquier método de predicción.
4. **Distinguir** entre enfoques deductivo, inductivo e híbrido, y cuándo usar cada uno.
5. **Comparar** las interpretaciones frequentist, bayesiana y de propensión de la probabilidad.
6. **Clasificar** métodos concretos (regresión, random forest, GP, VAE, GPT, DSGE, etc.) en el espacio 5D.
7. **Aplicar** heurísticas por dimensión para elegir un enfoque adecuado a un problema nuevo.
8. **Analizar** casos de estudio reales ubicando cada decisión en el marco taxonómico.

---

**Siguiente:** [El problema fundamental →](01_el_problema_fundamental.md)
