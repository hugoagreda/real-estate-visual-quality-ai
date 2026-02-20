# Estrategia de Modelado y Aprendizaje del Sistema

## 1. Enfoque general

El sistema propuesto no se basa en un modelo de clasificación simple (por ejemplo, “imagen buena / imagen mala”), sino en un enfoque más avanzado y explicable que combina:

- Regresión multivariable
- Aprendizaje incremental con feedback humano (human-in-the-loop)
- Análisis de contribución de métricas (ablation study)

Este enfoque permite evaluar la calidad visual de una imagen de forma continua, interpretable y progresivamente mejorable.

---

## 2. Representación del problema

Cada imagen es transformada en un **vector de características visuales**, extraídas mediante técnicas de visión artificial.

Ejemplo de características:
- score de iluminación
- score de nitidez
- score de composición
- score de balance de color
- score de complejidad visual

Estas características numéricas representan distintos aspectos técnicos de la calidad visual de la imagen.

---

## 3. Modelo principal: Regresión multivariable

### 3.1 Justificación del uso de regresión

Se utiliza un modelo de **regresión multivariable** en lugar de clasificación por las siguientes razones:

- El objetivo del sistema es generar un **score continuo (0–100)**, no una etiqueta discreta.
- La regresión permite modelar relaciones graduales entre las métricas visuales y la calidad percibida.
- Facilita la interpretación del peso e importancia de cada métrica.
- Permite comparar imágenes entre sí de forma más precisa.

Este enfoque es más adecuado para evaluar calidad visual que una clasificación binaria o multiclase.

---

### 3.2 Tipo de modelos considerados

El sistema puede emplear modelos de regresión como:

- Regresión lineal multivariable
- Regresión Ridge o Lasso (para regularización)
- Modelos de boosting para regresión (opcional, en fases avanzadas)

En la primera versión del proyecto se priorizan modelos interpretables y estables.

---

## 4. Aprendizaje con feedback humano (Human-in-the-loop)

### 4.1 Motivación

La calidad visual es un concepto parcialmente subjetivo.  
Por ello, el sistema incorpora feedback humano para ajustar progresivamente su comportamiento.

---

### 4.2 Tipo de feedback

El usuario puede proporcionar feedback de forma sencilla, por ejemplo:

- aceptar o rechazar una mejora propuesta
- indicar si la evaluación ha sido útil o no
- comparar resultados antes y después de una corrección

Este feedback se almacena junto con las métricas visuales asociadas a la imagen.

---

### 4.3 Uso del feedback en el aprendizaje

El feedback se utiliza para:
- recalibrar el modelo de regresión
- ajustar los pesos de las métricas
- mejorar la coherencia entre score técnico y percepción del usuario

Este proceso permite una mejora progresiva del sistema sin necesidad de reentrenamientos masivos.

---

## 5. Análisis de contribución de métricas (Ablation Study)

### 5.1 Objetivo del análisis

El análisis de ablación permite estudiar:
- qué métricas aportan más valor al score global
- cómo afecta la eliminación de una métrica al resultado final
- si existen métricas redundantes o poco relevantes

Este análisis aporta rigor metodológico y validación del diseño del sistema.

---

### 5.2 Metodología

El análisis se realiza mediante los siguientes pasos:

1. Definir un modelo base con todas las métricas activas.
2. Eliminar una métrica individualmente.
3. Recalcular el score global.
4. Comparar resultados con el modelo base.
5. Analizar la variación en precisión, estabilidad y coherencia.

Este proceso se repite para cada métrica del sistema.

---

### 5.3 Resultados esperados

El análisis permite:
- identificar métricas críticas (por ejemplo, iluminación)
- justificar los pesos asignados en el score global
- respaldar decisiones de diseño con evidencias cuantitativas

---

## 6. Ventajas del enfoque propuesto

El enfoque de regresión + feedback + análisis aporta:

- Mayor complejidad que un sistema de clasificación simple
- Alta explicabilidad del modelo
- Mejora progresiva con el uso
- Base sólida para validación académica
- Posibilidad de extensión futura

Este diseño equilibra complejidad técnica y viabilidad práctica dentro del contexto de un máster en Inteligencia Artificial.

---

## 7. Alcance del proyecto

El proyecto se centra en:
- evaluación técnica de calidad visual
- aprendizaje incremental supervisado por el usuario
- análisis metodológico del sistema

No se aborda:
- valoración económica del inmueble
- generación de imágenes artificiales
- evaluación estética subjetiva

---

## 8. Conclusión

El sistema propuesto representa una aplicación realista y avanzada de técnicas de inteligencia artificial, combinando visión artificial, aprendizaje supervisado y análisis explicable.

Este enfoque permite demostrar tanto conocimientos técnicos como criterio metodológico, alineándose con los objetivos formativos del máster.
