# Informe Técnico: Fase 1 - Validación del Modelo y Selección de Arquitectura
**Fecha:** Diciembre de 2025

**Proyecto:** CattleCounter (Monitoreo Aéreo de Ganado)

**Autor:** Carlos Luis Noriega - ML Researcher

### 1. Resumen Ejecutivo
El objetivo de la Fase 1 fue evaluar la viabilidad de utilizar Aprendizaje Profundo (Deep Learning) para contar ganado a partir de imágenes aéreas de drones (vista cenital) sin un entrenamiento específico previo (Zero-Shot). Comparamos enfoques arquitectónicos y establecimos un rendimiento de referencia utilizando un Detection Transformer (DETR) pre-entrenado.

**Resultado Clave:** El modelo de referencia logró una precisión de conteo (accuracy) del 84% (37/44 sujetos) en un entorno de prueba controlado, demostrando una alta viabilidad para la implementación en producción después del ajuste fino (fine-tuning).

### 2. Decisión Arquitectónica: CNN vs. Transformers
Evaluamos dos enfoques de vanguardia para el motor de detección de objetos.

##### Opción A: CNNs (ej., YOLOv8)

*   **Pros:** Inferencia extremadamente rápida (Tiempo real), ligero.

*   **Contras:** Dificultad con la oclusión y multitudes de alta densidad. El "campo receptivo" es local, lo que dificulta la distinción de animales individuales en agrupaciones.

*   **Veredicto:** Descartado para el pipeline de procesamiento por lotes, reservado para futuras implementaciones solo en el borde (edge-only).

##### Opción B: Transformers (DETR - Detection Transformer)

*   **Pros:** Utiliza mecanismos de Auto-Atención (Self-Attention). Esto permite que el modelo comprenda el **contexto global de la imagen**. Modela eficazmente las relaciones entre partes del objeto, lo que lo hace superior para manejar la oclusión (p. ej., separar dos vacas caminando lado a lado).

*   **Contras:** Mayor costo computacional (inferencia más lenta).

*   **Veredicto:** SELECCIONADO. Dado que los requisitos permiten el procesamiento por lotes (informes asincrónicos) en lugar de una retroalimentación en tiempo real de milisegundos, la precisión se prioriza sobre la velocidad pura.

### 3. El Desafío de la "Vista Cenital"
Estandarizamos una vista de 90° de Arriba-a-Abajo (Cenital) para la adquisición de datos.

*   **Ventaja de Ingeniería:** Elimina casi por completo la oclusión. Cada animal ocupa un área de cuadro delimitador (bounding box) distinta en el suelo.

*   **Desafío de IA (El "Sesgo COCO"):** Los conjuntos de datos estándar (COCO) contienen imágenes de vacas vistas de lado. Desde arriba, una vaca aparece como una mancha texturizada rectangular.

*   **Observaciones:** El modelo DETR pre-entrenado inicialmente clasificó erróneamente las vacas como pájaro, oveja u oso debido a las similitudes geométricas desde la perspectiva de arriba hacia abajo.

*   **Solución:** Implementamos un pipeline heurístico de post-procesamiento:
    1.  **Mapeo de Clases:** Remapeamos clases detectadas específicas (pájaro, oveja, oso) a "Vaca".
    2.  **Filtrado de Área:** Se descartaron las detecciones por debajo de 4000px² (ruido) y por encima de 150000px².

### 4. Resultados Experimentales
**Banco de Pruebas:**

*   **Entrada:** Video 1080p @ 60 FPS (Vuelo de dron).
*   **Verdad Terreno (Ground Truth):** 44 Vacas (Conteo Manual).
*   **Predicción del Modelo:** 37 Vacas.
*   **Métricas:**
    *   **Precisión (Accuracy):** ~84%.
    *   **FPS (Inferencia):** ~15 FPS en CPU (Mac M1/MPS).

**Análisis de Errores:** El 16% faltante (7 vacas) se atribuyó a:
1.  **Casos Extremos (Edge Cases):** Las vacas que entraban/salían del cuadro se filtraron debido a la baja confianza.
2.  **Similitud Visual:** Las vacas oscuras sobre suelo oscuro fueron omitidas por los pesos pre-entrenados.
3.  **Agrupación:** El algoritmo ByteTrack ocasionalmente intercambió IDs cuando las vacas se movieron rápidamente en direcciones opuestas.

### 5. Próximos Pasos (Fase 2)
Para cerrar la brecha del 84% a >98% de precisión:

1.  **Recolección de Datos:** Adquirir un conjunto de datos diverso de más de 200 imágenes cenitales (etiquetadas manualmente a partir de los vídeos de campo).
2.  **Ajuste Fino (Fine-Tuning):** Reentrenar DETR específicamente con el nuevo conjunto de datos para corregir el error de clasificación "Pájaro/Oso" y mejorar la confianza en objetivos oscuros.
3.  **Despliegue:** Contenerizar el pipeline para su ejecución en la Nube (Cloud execution) (Azure).