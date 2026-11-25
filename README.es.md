# Mole Poblano

**De foto a cuarto 3D.** Convierte fotos o un video corto de un cuarto en un **plano con medidas** y una **escena 3D que puedes recorrer**. Todo corre en tu propia tarjeta de video, con ComfyUI y Blender.

[![Mira el trailer de 1 minuto en YouTube](docs/img/trailer_thumb.jpg)](https://youtu.be/cJ_rc77GipY)

![Habitación de hotel reconstruida en 3D desde un video del celular](docs/img/hotel_orbit.gif)

**Qué hace:**

1. Encuentra el piso, las paredes y el techo, y calcula la profundidad de cada punto de la foto.
2. Recorta, nombra y mide cada mueble, y calcula hacia dónde mira.
3. Dibuja el plano visto desde arriba, con medidas y distancias. Con una medida real se corrige la escala de todo el plano.
4. Arma la escena 3D en Blender, con la misma cámara que la foto.
5. Junta varias fotos o cuadros de un video en un solo cuarto.

**Qué entra y qué sale:** cada cuarto de ejemplo empezó como una grabación rápida con el celular. Las grabaciones originales están en `examples/`, comprimidas y sin audio:
- **Hotel:** dos videos de unos 30 s (`examples/source_videos/hotel_part1.mp4` y `hotel_part2.mp4`).
- **Comedor y cocina:** cuatro clips cortos juntados en `examples/source_videos/dining_kitchen.mp4`.
- **Cuarto con vestidor:** ocho fotos sacadas de un video (`examples/source_photos_bedroom/`).

**Qué trae:**

- **`pipeline/`:** los scripts.
- **`viewer/`:** el visor, en español e inglés, con tres cuartos de ejemplo. Ábrelo con `serve.bat`.
- **`demo/`:** el demo paso a paso.
- **`examples/`:** una foto de ejemplo para probar.

**Los pesos de los modelos no vienen incluidos.** Cada quien los descarga de su página oficial.

## Cómo se hicieron los cuartos de ejemplo

El pipeline entrega el plano, las medidas y un modelo 3D de cada objeto. Los tres cuartos del visor y el hotel del GIF tuvieron además un acabado hecho cuarto por cuarto, con scripts extra y decisiones a mano que no vienen en este repositorio:
- limpiar el plano;
- corregir muebles encimados o que atravesaban paredes;
- elegir materiales;
- hornear la luz en Blender.

El pipeline te da un buen borrador, no el acabado de los ejemplos. Automatizar ese acabado es el siguiente paso.

## Pulir un borrador con tu propio agente de IA

Todo el cuarto vive en un solo archivo, `out/<cuarto>/map.json`, y `pipeline/CONVENTIONS.md` documenta cada campo. Si usas un agente de código con tu propia cuenta (Claude Code, Cursor, Codex, Antigravity…), abre la carpeta de Mole Poblano con él y pídele cambios en lenguaje natural. Por ejemplo: *"en out/hotel_room, mueve el sillón 20 cm a la izquierda, deja el escritorio en 75 cm de alto y regenera la escena"*.

El agente encuentra solo las instrucciones en [AGENTS.md](AGENTS.md): cómo son las coordenadas, qué cambios respeta la escena 3D y el comando para regenerar todo en unos segundos (`pipeline/rebuild.py`). Este repositorio no incluye claves ni suscripciones. El agente y su costo son tuyos, y todo lo demás sigue corriendo en tu equipo.

## Cómo se construyó

Yo diseñé el proyecto, elegí los modelos, lo probé contra medidas reales y lo usé con clientes reales. Escribí el código con **Claude Code** como asistente de programación con IA; por eso algunos commits dicen `Co-Authored-By: Claude`.

## Licencias

- **El código** es **GPL-3.0**.
- **Los cuartos de ejemplo** (modelos 3D, planos, fotos y renders) son **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es)**: puedes usarlos para lo que quieras, incluso comercialmente, dando crédito a "BySCM – Mole Poblano".
- **Las licencias de cada modelo** están en [MODELS.md](MODELS.md); algunas no permiten uso comercial.

Las instrucciones completas, la configuración y las limitaciones están en el [README en inglés](README.md).
