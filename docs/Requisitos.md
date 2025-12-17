# Requisitos de Adquisición de Datos de Campo
**Para:** Equipo de Operaciones de Drones

**De:** Equipo de Ingeniería de IA

**Asunto:** Especificaciones para la Recolección de Datos (Dataset Collection) (Contador de Ganado)

Para asegurar el éxito del entrenamiento del modelo de IA (*Fine-Tuning* o Ajuste Fino), por favor, adhiérase a las siguientes especificaciones para la recolección de video/imágenes. La calidad del modelo depende enteramente de la calidad de estas entradas.

### 1. Parámetros de Vuelo

*   **Ángulo de la Cámara (Gimbal):** Estrictamente 90° (Nadir/Cenital). La cámara debe apuntar directamente hacia abajo. No utilice ángulos oblicuos (45°), ya que introducen oclusión.
*   **Altitud:** Mantenga una altitud constante entre 15m y 30m.
    *   *Objetivo: Las vacas deben ser claramente visibles, pero no deben llenar toda la pantalla. Necesitamos ver la separación entre los animales.*
*   **Velocidad:** Lenta y constante (Modo Cinemático). Evite giros (guiñada) repentinos o aceleración rápida para prevenir el desenfoque de movimiento (*motion blur*).
*   **Ruta de Vuelo:** Vuele estrictamente sobre el rebaño o a lo largo del camino. Se prefieren barridos lineales en lugar de órbitas circulares.

### 2. Especificaciones de Video

*   **Resolución:** Mínimo 720p (1280x720px). Se prefiere 1080p (1920x1080px) si el almacenamiento lo permite.
*   **Tasa de Cuadros (Frame Rate):** 30 FPS o 60 FPS.
*   **Formato:** .mp4 o .mov.
*   **Cantidad:** Necesitamos 5 a 10 clips cortos (de 15 a 30 segundos cada uno).

### 3. Condiciones Ambientales (Diversidad)
Es necesario que la IA aprenda a ver vacas en diferentes situaciones. Por favor, intente capturar:

*   **Iluminación:**
    *   **Ideal:** Días nublados/encapotados (sombras suaves).
    *   **Aceptable:** Temprano en la mañana o tarde en la tarde.
    *   **Evitar:** Pleno mediodía (12:00 PM) con sol intenso, ya que las sombras fuertes pueden ser confundidas con una segunda vaca por la IA.
*   **Fondos:**
    *   Pasto verde.
    *   Tierra seca/Caminos de tierra.
    *   Concreto/Corrales de engorde (*Feedlots*) (si están disponibles).
*   **Sujetos:**
    *   Intente capturar diferentes colores de rebaño (Negro, Blanco, Marrón/Rojo) si es posible.
    *   Capture diferentes densidades (vacas dispersas vs. vacas agrupadas).

### 4. Muestras "Vacías" (Opcional, pero Útil)
Si es posible, grabe 5-10 segundos del campo sin vacas (solo pasto, rocas o árboles). Esto *nos ayuda a enseñar a la IA qué no es una vaca para reducir los falsos positivos*.

**Lista de Verificación Resumen para el Operador:**

* Ángulo a 90° hacia abajo.
* Altitud constante (misma que el vídeo cenital de referencia).
* Sin desenfoque de movimiento (*motion blur*).
* Variedad de fondos grabados.