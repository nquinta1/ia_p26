# Reflexión sobre Cadenas de Markov y Modelos Ocultos de Markov
Después de ver los videos sobre Cadenas de Markov y Modelos Ocultos de Markov, me quedó
mucho más claro cómo funcionan los modelos probabilísticos para representar sistemas que
cambian con el tiempo. Algo que me llamó mucho la atención es lo simple pero poderoso que es
el supuesto fundamental de las cadenas de Markov: el futuro depende únicamente del estado
actual y no de toda la historia pasada. Esto, aunque parece una simplificación fuerte, permite
modelar una gran cantidad de fenómenos reales de forma eficiente.
En el primer video entendí que una cadena de Markov se basa en estados y probabilidades de
transición. Es decir, si estoy en un estado A, existe cierta probabilidad de pasar a B, C, etc. Lo
interesante es que estas probabilidades se pueden representar en matrices, lo que conecta
directamente con herramientas matemáticas que ya hemos visto en el curso, como álgebra lineal.
También me sorprendió cómo estas cadenas pueden converger a distribuciones estables, lo que
significa que, después de muchos pasos, el sistema alcanza un comportamiento predecible
independientemente del estado inicial.
En el segundo video, sobre Modelos Ocultos de Markov (HMM), el concepto se vuelve aún más
interesante. Aquí ya no observamos directamente los estados, sino solo evidencias o “señales”
que dependen de esos estados ocultos. Esto me pareció muy poderoso porque refleja mejor la
realidad: en muchos problemas reales no tenemos acceso directo a lo que realmente está
pasando, sino solo a datos observables. Por ejemplo, en reconocimiento de voz, no vemos
directamente las palabras “reales”, sino señales de audio.
Lo que más me sorprendió fue cómo estos modelos pueden inferir información oculta a partir de
observaciones, usando probabilidades. Es decir, pueden “adivinar” el estado más probable que
generó una secuencia de datos. Esto conecta directamente con temas del curso como redes
bayesianas y razonamiento probabilístico, donde se busca inferir información incompleta.
Además, estos modelos tienen aplicaciones muy claras en ciencia de datos, como predicción de
comportamiento, procesamiento de lenguaje natural y sistemas de recomendación. También los
relaciono con búsqueda y toma de decisiones, porque en muchos casos el agente no tiene
información completa del entorno, lo cual es muy similar a lo que ocurre en los HMM.
En conclusión, estos temas me ayudaron a entender mejor cómo la probabilidad puede utilizarse
para modelar incertidumbre y sistemas dinámicos. Más allá de la teoría, veo que son
herramientas fundamentales en inteligencia artificial moderna, especialmente cuando se trabaja
con datos secuenciales o incompletos.
