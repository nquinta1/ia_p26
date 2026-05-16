---
title: "02 · On-policy vs Off-policy"
summary: "La tabla Q como objeto concreto, el error TD, las dos ecuaciones de Bellman y la bifurcación SARSA / Q-learning"
---

## La tabla $Q$ como objeto concreto

Antes de hablar de algoritmos, construyamos el objeto central: la **tabla $Q$**.

Es una matriz con un estado por fila y una acción por columna.
Cada celda responde: *"si estoy en $s$ y tomo $a$, ¿qué retorno total espero acumular hasta la meta?"*

Al principio no sabemos nada, así que inicializamos todo en cero:

![Tabla Q vacía]({{ '/23_reinforcement_learning/images/03_q_table_empty.png' | url }})

Para la escalera: 5 filas (estados 0–4; el estado 5 es terminal y no necesita acciones) y 2 columnas ($+1$ y $+2$).
La celda $(s=4, +2)$ está marcada como no disponible porque desde $s=4$ la única acción posible es $+1$.

Con cada episodio que juega el agente, algunas celdas se actualizan.
Después de suficientes episodios, la tabla converge a $Q^∗$.

**Dato importante:** tanto SARSA como Q-learning usan exactamente esta misma tabla como única estructura de datos.
No hay ninguna diferencia en cómo se almacena o se lee — ambos comienzan con todos los valores en cero y actualizan celda a celda tras cada paso.
La única diferencia entre los dos algoritmos está en **cómo calculan el target** que empuja cada celda hacia su valor correcto.

---

## El esqueleto de la actualización TD

La idea central de TD es simple: **compara lo que esperabas con lo que realmente pasó, y ajusta**.

Formalmente, la actualización tiene esta estructura:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \bigl[ r + \gamma \cdot \textbf{?} - Q(s,a) \bigr]$$

El término entre corchetes es el **error TD** $\delta_t$:

Piezas de la fórmula:

| Símbolo | Nombre | Intuición |
|---------|--------|-----------|
| $Q(s,a)$ | Estimación actual | "Antes de ejecutar $a$, creía que valía $Q(s,a)$" |
| $r$ | Recompensa observada | Lo que el ambiente devolvió al hacer $a$ en $s$ |
| $\gamma \cdot \textbf{?}$ | Estimación del futuro | Cuánto más se puede ganar desde $s'$ en adelante |
| $r + \gamma \cdot \textbf{?}$ | Target TD | Nueva estimación del retorno total — con un paso real |
| $\alpha$ | Tasa de aprendizaje | Cuánto actualizamos: 0 = no aprendes, 1 = reemplazas todo |
| $\delta_t$ | Error TD | Diferencia entre lo nuevo y lo viejo: la "sorpresa" |

**`?`** es la pieza que falta: estimación del valor en $s'$. **Esta elección distingue SARSA de Q-learning.**

### El error TD $\delta_t$: intuición

$$\boxed{\delta_t = (r + \gamma \cdot \textbf{?}) - Q(s,a)}$$

- $r + \gamma \cdot \textbf{?}$ → **nueva estimación** (target TD): lo que el agente cree que valdrá el futuro tras este paso real
- $Q(s,a)$ → **estimación anterior**: lo que el agente creía antes de ejecutar $a$

Piénsalo como un sistema de aprendizaje por retroalimentación:

| Caso | Lectura | Efecto |
|------|---------|--------|
| $\delta_t > 0$ | "La acción fue **mejor** de lo que pensaba" — el target supera a $Q(s,a)$ | Subimos $Q(s,a)$ |
| $\delta_t < 0$ | "La acción fue **peor** de lo que pensaba" — el target está por debajo de $Q(s,a)$ | Bajamos $Q(s,a)$ |
| $\delta_t = 0$ | "Exactamente lo que esperaba" — no hay nueva información | $Q(s,a)$ no cambia |

El algoritmo empuja $Q(s,a)$ hacia el target `r + γ·?` en pequeños pasos de tamaño $\alpha$.
Con suficientes episodios, $Q$ converge al punto donde $\delta_t = 0$ para todos los pares $(s,a)$ — que es exactamente la ecuación de Bellman.

La pregunta que queda abierta: ¿qué ponemos en el `?`?

---

## Las dos ecuaciones de Bellman

Del módulo 21 y de la página anterior, recordamos:

$$Q^\pi(s,a) = \mathbb{E}_{s' \sim T, a' \sim \pi}\bigl[r + \gamma Q^\pi(s', a')\bigr] \tag{Eq. 1}$$

$$Q^∗(s,a) = \mathbb{E}_{s' \sim T}\bigl[r + \gamma \max_b Q^∗(s', b)\bigr] \tag{Eq. 2}$$

($b$ es la variable muda de optimización — equivalente a $a'$ en la notación estándar)

Colocadas juntas, la diferencia salta a la vista:

| | Eq. 1 — valor de $\pi$ | Eq. 2 — valor óptimo |
|--|------------------------|----------------------|
| Valor futuro en $s'$ | $Q^\pi(s', a')$ con $a' \sim \pi$ | $\max_b Q^∗(s', b)$ |
| Pregunta que responde | ¿Cuánto vale seguir mi política actual? | ¿Cuánto vale hacer lo mejor posible? |
| Converge a | $Q^\pi$ — depende de $\pi$ | $Q^∗$ — óptimo global |

Ambas ecuaciones son **exactas** cuando $Q$ es la función correcta.
Las actualizaciones TD son versiones *aproximadas* que usan una muestra $(s, a, r, s')$ en lugar de la esperanza completa.

---

## La bifurcación: ¿qué va en el `?`?

Dos opciones naturales para rellenar el `?`:

**Opción A — usa la acción que *realmente* tomaremos en $s'$** (muestreada de $\pi$):

$$\textbf{?} = Q(s', a') \quad \text{donde } a' \sim \pi \qquad \Longrightarrow \quad \text{SARSA}$$

Apunta a la ecuación (1): aprende el valor de la política que se está ejecutando.

**Opción B — usa el máximo sobre todas las acciones de $s'$**:

$$\textbf{?} = \max_b Q(s', b) \qquad \Longrightarrow \quad \text{Q-learning}$$

Apunta a la ecuación (2): aprende el valor óptimo, sin importar qué acción ejecuta el agente.

Esto da lugar a una forma de pensar en los dos algoritmos que es muy útil:

| | Pregunta que responde $Q(s,a)$ | Converge a |
|--|-------------------------------|-----------|
| **SARSA** | *"¿Cuánto vale $a$ si después sigo actuando como ahora — con exploración incluida?"* | $Q^{\pi_\varepsilon}$ |
| **Q-learning** | *"¿Cuánto vale $a$ si después actúo de la mejor manera posible?"* | $Q^∗$ |

SARSA evalúa el valor de las acciones que *realmente tomas* (incluyendo las exploratorias).
Q-learning evalúa el valor de la *mejor acción posible* desde cada estado, independientemente de lo que el agente haga en la práctica.
En una frase: **SARSA aprende el valor de lo que haces; Q-learning el valor de lo que podrías hacer.**

```
Actualización TD: Q(s,a) ← Q(s,a) + α [r + γ · ? − Q(s,a)]
                                              |
                  ┌───────────────────────────┴─────────────────────────┐
                  │                                                     │
         ? = Q(s', a'),  a'∼π                           ? = max Q(s', a')
                                                               a'
         ┌───────────────────┐                       ┌───────────────────┐
         │      SARSA        │                       │    Q-learning     │
         │   (on-policy)     │                       │  (off-policy)     │
         └───────────────────┘                       └───────────────────┘
```

---

## Política de comportamiento y política objetivo

:::exercise{title="Definiciones — on-policy vs off-policy"}

**Política de comportamiento** $\mu$: la política que usa el agente para **explorar** el ambiente — las acciones que *realmente* ejecuta.

**Política objetivo** $\pi$: la política que el agente está **aprendiendo** — hacia la cual converge $Q$.

**On-policy** ($\mu = \pi$): el agente aprende el valor de la política que está ejecutando.

**Off-policy** ($\mu \neq \pi$): el agente puede explorar con una política diferente a la que está aprendiendo.

:::

En ambos algoritmos usamos $\varepsilon$-greedy como política de comportamiento $\mu$ para garantizar exploración.
La diferencia está en $\pi$:

| | SARSA | Q-learning |
|--|-------|------------|
| $\mu$ (comportamiento) | $\varepsilon$-greedy | $\varepsilon$-greedy |
| $\pi$ (objetivo) | $\varepsilon$-greedy ($\mu = \pi$) | greedy pura ($\mu \neq \pi$) |
| Clasificación | **on-policy** | **off-policy** |

---

## ¿A qué converge cada uno?

:::exercise{title="Garantías de convergencia"}

**SARSA** con $\varepsilon$ fijo: converge a $Q^{\pi_\varepsilon}$ — el valor de la política $\varepsilon$-greedy, no el óptimo exacto.

**SARSA** decrementando $\varepsilon \to 0$ (con condiciones de convergencia estándar): converge a $Q^∗$.

**Q-learning**: converge a $Q^∗$ *independientemente de la política de comportamiento*, siempre que cada par $(s,a)$ sea visitado un número suficiente de veces.

:::

La diferencia práctica: Q-learning siempre aprende la política óptima aunque explore agresivamente; SARSA aprende el valor de la política con la que realmente opera.

---

## Un momento de divergencia concreto

Veamos exactamente cuándo y por qué los dos algoritmos producen resultados distintos.

### Situación de partida

Después de dos episodios ($\alpha=0.5$, $\gamma=1$, $\varepsilon=0.4$), ambos algoritmos tienen **la misma tabla $Q$**:

| Estado | $+1$ | $+2$ |
|--------|------|------|
| $s=0$ | $-1.0$ | $-2.5$ |
| $s=1$ | $0$ | $-5.0$ |
| $s=2$ | $0$ | $-0.5$ |
| $s=3$ | $0$ | $0$ |
| $s=4$ | $0$ | — |

### Episodio 3, paso 1 — el momento de la bifurcación

**Contexto:** el agente está en $s=0$.

La acción greedy sería $+1$ (porque $Q(0,+1)=-1.0 > Q(0,+2)=-2.5$),
pero $\varepsilon$-greedy **explora** y elige $a=+2$.

El ambiente responde: $s'=2$, $r=-5$.

**Valores disponibles en $s'=2$:**

$$Q(2,+1) = 0 \qquad Q(2,+2) = -0.5$$

---

### Q-learning: usa siempre el valor máximo en $s'$

No importa qué acción se ejecutará realmente — toma el mejor caso posible:

$$\textbf{?} = \max_b Q(2, b) = \max(0,\ -0.5) = 0$$

Error TD:

$$\delta = r + \gamma \cdot \textbf{?} - Q(0,+2) = -5 + 1 \cdot 0 - (-2.5) = \mathbf{-2.5}$$

Nueva celda:

$$Q(0,+2) \leftarrow -2.5 + 0.5 \cdot (-2.5) = \mathbf{-3.75}$$

---

### SARSA: usa la acción que *realmente* tomará en $s'$

La política $\varepsilon$-greedy en $s'=2$ elige (greedily) $+1$ con prob. $0.6$, pero en este paso **explora** y elige $a'=+2$:

$$\textbf{?} = Q(2,\ a'=+2) = -0.5$$

Error TD:

$$\delta = r + \gamma \cdot \textbf{?} - Q(0,+2) = -5 + 1 \cdot (-0.5) - (-2.5) = \mathbf{-3.0}$$

Nueva celda:

$$Q(0,+2) \leftarrow -2.5 + 0.5 \cdot (-3.0) = \mathbf{-4.00}$$

---

### Resultado: tablas distintas a partir de este paso

| Algoritmo | Error $\delta$ | Nuevo $Q(0,+2)$ |
|-----------|---------------|-----------------|
| Q-learning | $-2.5$ | $-3.75$ |
| SARSA | $-3.0$ | $-4.00$ |

**¿Por qué difieren?**
SARSA incorporó la penalización de su propia exploración: eligió $+2$ (valor $-0.5$), que es peor que el greedy $+1$ (valor $0$).
Q-learning ignoró esa exploración y asumió el mejor caso posible en $s'$.

Ese es el precio de ser **on-policy**: si tu política explora acciones subóptimas, aprendes que son subóptimas.
Si eres **off-policy**, aprendes el valor óptimo aunque explores.

---

## Evolución de la tabla $Q$ en los dos primeros episodios

Para ver cómo se van llenando las celdas antes del punto de divergencia:

![Evolución de la tabla Q]({{ '/23_reinforcement_learning/images/04_q_table_evolution.png' | url }})

Los episodios 1 y 2 producen la misma tabla para SARSA y Q-learning porque en esos pasos el greedy coincide con lo que ε-greedy elige (o las celdas consultadas son cero de todas formas).
El divergence empieza cuando la tabla tiene valores suficientemente distintos de cero para que la diferencia entre $a' \sim \pi$ y $\max_b$ importe.
