# Qwen3.8-27B-FP8 en Cloudera AI Workbench SP2/SP3

Esta guía documenta el perfil probado de `Qwen/Qwen3.8-27B-FP8` como **Workbench Model** en una NVIDIA A100 PCIe de 80 GB. El endpoint usa la función `predict(args)` de Cloudera y devuelve una respuesta JSON similar a Chat Completions.

## Estado validado

La configuración quedó validada el 19 de septiembre de 2026 en un entorno on-premises SP2 con:

| Componente | Valor observado |
|---|---|
| GPU | NVIDIA A100 80GB PCIe, 79.3 GiB visibles, SM80 |
| Runtime | Nvidia GPU Edition, Python 3.10 |
| vLLM | `0.29.0+cu129` |
| PyTorch | `2.13.0+cu129` |
| CUDA de PyTorch | `12.9` |
| Transformers | `5.15.0` |
| NumPy del worker | `2.2.6` |
| Pesos | FP8 block-wise, ejecutados como weight-only FP8 mediante Marlin/W8A16 |
| Memoria de pesos observada | 28.9 GiB |
| Caché KV | BF16, 40.31 GiB disponibles |
| Capacidad de caché observada | 647,288 tokens |
| Longitud configurada | 262,144 tokens |
| Concurrencia calculada por vLLM | 2.47 solicitudes de 262,144 tokens; el wrapper limita a una |
| Backend de atención | `TRITON_ATTN` |
| Sampler | Nativo de vLLM/PyTorch; FlashInfer sampler deshabilitado |
| Límite de salida configurado en SP2 | `QWEN_MAX_OUTPUT_TOKENS=8192` |
| Salida predeterminada si la petición omite `max_tokens` | 128 tokens |
| Resultado | Modelo cargado y endpoint operativo |

Estas cifras son una referencia de diagnóstico, no una reserva contractual. Pueden variar con el driver, Runtime, revisión del checkpoint, procesos residentes y memoria realmente asignada. Este perfil exige una A100 de 80 GB completa y rechaza menos de 70 GiB visibles para detectar MIG, vGPU o una A100 de 40 GB antes de cargar los pesos.

## Configuración mínima que funciona

En **Deploy model from code** configure:

| Campo | Valor |
|---|---|
| Root Directory / Model Root Directory | vacío o `.` |
| Build Script Path | `cdsw-build.sh` |
| Build variable | `MODEL_FAMILY=qwen3_8` |
| Build variable | `GPU_TYPE=a100` |
| File | `qwen3_8/model_a100.py` |
| Function | `predict` |
| Example Input | contenido de `examples/qwen3_8_input.json` |
| Runtime | Nvidia GPU Edition, Python 3.10 o 3.11, Ubuntu 24.04 |
| GPU | 1 × A100 80 GB completa, sin MIG |
| CPU | 8 vCPU como mínimo; 16 recomendados |
| RAM | 64 GiB como mínimo; 128 GiB recomendados |
| Réplicas iniciales | 1 |

El arranque funciona sin variables de ejecución porque el código contiene defaults seguros. En el despliegue SP2 validado se añadió `QWEN_MAX_OUTPUT_TOKENS=8192` para permitir respuestas largas. Esta variable es un **techo**, no obliga al modelo a generar 8192 tokens: cada petición continúa controlándose con `max_tokens`, cuyo default es 128. Si la interfaz separa las variables de build de las variables de ejecución, `MODEL_FAMILY` y `GPU_TYPE` deben estar disponibles durante el **build**.

## Qué hace cada fichero

| Fichero | Responsabilidad |
|---|---|
| `../cdsw-build.sh` | Selecciona el instalador usando `MODEL_FAMILY` y `GPU_TYPE` |
| `install_a100.sh` | Crea `.venv`, instala la rueda vLLM cu129 y valida versiones/imports |
| `requirements.txt` | Fija Transformers para builds reproducibles |
| `model_a100.py` | Punto de entrada PBJ; importa Cloudera y actúa como proxy ligero |
| `worker_a100.py` | Proceso aislado que importa vLLM, carga el modelo y genera respuestas |
| `../examples/qwen3_8_input.json` | Entrada lista para pegar en el formulario |
| `../examples/qwen3_8_output.json` | Forma orientativa de la respuesta |

## Por qué existe un virtualenv aislado

El Runtime SP2 ya contiene librerías utilizadas por Cloudera, por ejemplo paquetes compilados para NumPy 1.x y restricciones como `protobuf==4.25.3`. vLLM 0.29 instala un stack más reciente, incluido NumPy 2. Mezclar ambos conjuntos en el mismo intérprete produjo incompatibilidades ABI como:

```text
numpy.dtype size changed, may indicate binary incompatibility
```

El instalador crea `qwen3_8/.venv`. El diseño no activa ese entorno dentro del proceso PBJ ni inserta sus `site-packages` en `sys.path`:

```text
Cloudera/PBJ con Python base
        |
        | predict(args), JSON por socket local
        v
model_a100.py  ----------------->  worker_a100.py con .venv/bin/python
  cml.models_v1                    torch + numpy 2 + vLLM + transformers
```

El proxy inicia un único worker persistente. El worker carga los pesos una sola vez, responde `ready` después de inicializar la caché y los kernels, y procesa las llamadas posteriores. Si falla durante el arranque, envía el traceback completo al proxy para que la causa aparezca en el log de Cloudera.

No configure manualmente `PYTHONPATH`, `VIRTUAL_ENV`, `PIP_USER` ni otro intérprete. `install_a100.sh` neutraliza `PIP_USER=true` únicamente durante el build porque pip no permite instalaciones `--user` dentro de un venv.

## Por qué se usa la rueda cu129

La distribución estándar de vLLM 0.29.0 publicada en PyPI puede resolver una variante CUDA distinta de la soportada por el driver del Runtime. El instalador descarga explícitamente:

```text
vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_<arquitectura>.whl
```

Después ejecuta `pip check` dentro del venv y valida:

- versión de vLLM;
- versión exacta de Transformers;
- CUDA contra la que fue construido PyTorch;
- importación de `Qwen3_5ForConditionalGeneration`.

El modelo se publica como Qwen3.8, mientras que la clase interna resuelta por Transformers/vLLM es `Qwen3_5ForConditionalGeneration`. Eso es normal para este checkpoint y no implica que se esté cargando otro modelo.

## Precisión, memoria y contexto

Hay que distinguir los pesos de la caché KV:

- **Pesos FP8:** válidos en A100 mediante el kernel Marlin weight-only FP8/W8A16. La A100 no tiene cálculo FP8 nativo y puede rendir menos que Hopper, pero el modelo carga correctamente.
- **Caché KV FP8:** no válida con Triton en A100 SM80. El backend exige SM89 o posterior.
- **Caché KV BF16:** configuración validada. El código la usa por defecto y transforma automáticamente una configuración heredada `fp8` en `bfloat16` cuando detecta una GPU anterior a SM89.
- **Contexto:** `262144` tokens quedó validado con 40.31 GiB de caché y capacidad calculada de 647,288 tokens.
- **Prefill:** `8192` limita el bloque procesado simultáneamente; no reduce la longitud total del chat.
- **Concurrencia:** el endpoint serializa las llamadas y usa `max_num_seqs=1` para priorizar contexto y estabilidad.

Si otro Runtime deja menos VRAM libre, reduzca `QWEN_MAX_MODEL_LEN` en este orden: `131072`, `65536`, `32768`. No aumente `QWEN_GPU_MEMORY_UTILIZATION` por encima de `0.95`.

## Triton, FlashInfer y ninja

La configuración separa tres caminos de ejecución:

1. Marlin ejecuta las capas lineales con pesos FP8 sobre Ampere.
2. `TRITON_ATTN` ejecuta la atención y admite la caché BF16.
3. El sampler nativo de vLLM/PyTorch realiza top-k/top-p.

FlashInfer detectó la GPU e intentó compilar su sampler durante el warmup. El Runtime no incluía el ejecutable `ninja`:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

Instalar solo `ninja` no garantiza éxito porque una compilación JIT también puede necesitar un toolkit NVCC visible. Por eso el worker fija antes de importar vLLM:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

Esto solo cambia el muestreo top-k/top-p. No desactiva Triton, no convierte los pesos a BF16 y no reduce la longitud de contexto.

## Variables de build

| Variable | Obligatoria | Default | Uso |
|---|---:|---|---|
| `MODEL_FAMILY` | Sí | `nemotron` en el dispatcher global | Debe ser `qwen3_8` |
| `GPU_TYPE` | Sí | `l40s` en el dispatcher global | Debe ser `a100` |
| `PYTHON_BIN` | No | `python3` | Python utilizado para crear el venv |
| `VLLM_VERSION` | No | `0.29.0` | Versión de la rueda oficial |
| `VLLM_CUDA_VARIANT` | No | `129` | Sufijo CUDA de la rueda |
| `VLLM_WHEEL_URL` | No | GitHub Releases oficial | Permite usar un mirror interno |
| `PYTORCH_INDEX_URL` | No | índice PyTorch cu129 | Permite usar un mirror interno |

No copie estas variables al entorno de ejecución salvo que una política de Cloudera utilice el mismo conjunto para ambas fases. Cambiar una variable de build requiere crear un build nuevo.

## Variables de ejecución: mapa completo

Hay tres cifras diferentes que no deben confundirse:

- **Default del código:** valor utilizado si la variable no existe.
- **Valor SP2 validado:** valor que se empleó en el despliegue que funcionó.
- **Rango aceptado:** límites de validación del wrapper, no una promesa de que todos los valores sean adecuados para una A100.

| Variable | Default del código | SP2 validado | Rango aceptado | Recomendación |
|---|---:|---:|---:|---|
| `QWEN_MODEL_ID` | `Qwen/Qwen3.8-27B-FP8` | igual | ID o ruta | Mantener o usar snapshot inmutable |
| `QWEN_SERVED_MODEL_NAME` | `qwen3.8-27b-fp8` | igual | texto | Solo cambia metadatos de salida |
| `QWEN_TENSOR_PARALLEL_SIZE` | `1` | `1` | `1-8` | Mantener `1` con una sola A100 |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.90` | `0.90` | `0.50-0.95` | Mantener `0.90`; probar `0.92` solo si hace falta KV |
| `QWEN_MAX_MODEL_LEN` | `262144` | `262144` | `2048-262144` | Mantener el máximo nativo; reducir si falta memoria |
| `QWEN_MAX_NUM_SEQS` | `1` | `1` | `1-64` | Mantener `1`; el proxy actual serializa |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | `8192` | `8192` | `2048-131072` | SP3: probar `16384` para prefill largo |
| `QWEN_KV_CACHE_DTYPE` | `bfloat16` | `bfloat16` | cadena aceptada por vLLM | En A100 debe permanecer BF16 |
| `QWEN_ENFORCE_EAGER` | `true` | `true` | booleano | SP3: probar `false` de forma aislada |
| `QWEN_ATTENTION_BACKEND` | `TRITON_ATTN` | igual | backend vLLM | Mantener Triton en este perfil |
| `QWEN_CPU_OFFLOAD_GB` | `0` | `0` | `0-128` | Mantener `0`; los pesos caben en GPU |
| `QWEN_MAX_OUTPUT_TOKENS` | `512` | **`8192`** | `1-8192` | Mantener `8192` como techo si los timeouts lo permiten |
| `QWEN_MAX_IMAGES_PER_PROMPT` | `1` | `1` | `0-4` | Mantener `1`; probar `2` solo con mediciones |
| `QWEN_STARTUP_TIMEOUT_SECONDS` | `1800` | `1800` | `60-7200` | No afecta a inferencia, solo al arranque |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | `0` | `0/1` | Mantener `0` mientras falten `ninja`/NVCC validados |
| `VLLM_ALLOWED_MEDIA_DOMAINS` | vacío | vacío | lista CSV | En producción usar una allowlist explícita |
| `HF_TOKEN` | vacío | según entorno | secreto | Token de solo lectura, nunca en Git |
| `HF_HOME` | default HF | según entorno | ruta | Usar volumen persistente con espacio suficiente |

Los antiguos nombres `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_KV_CACHE_DTYPE`, `VLLM_ENFORCE_EAGER`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_TENSOR_PARALLEL_SIZE` y `VLLM_CPU_OFFLOAD_GB` se aceptan como alias. El worker los captura y elimina antes de importar vLLM porque no son variables oficiales de vLLM 0.29 y, si permanecen, generan `Unknown vLLM environment variable`. Use los nombres `QWEN_*` en despliegues nuevos.

## Explicación didáctica de cada parámetro

### Identidad y procedencia del modelo

#### `QWEN_MODEL_ID`

Indica de dónde se descargan configuración, tokenizer y pesos. El valor actual apunta a Hugging Face:

```text
Qwen/Qwen3.8-27B-FP8
```

En producción es preferible usar un snapshot fijado o una copia interna. Si se deja una rama móvil como `main`, dos builds realizados en fechas diferentes podrían descargar revisiones distintas. Cambiar esta variable por otro modelo no garantiza compatibilidad: el código, la cuantización, la arquitectura y la VRAM se validaron únicamente con este checkpoint.

#### `QWEN_SERVED_MODEL_NAME`

Es una etiqueta lógica. Aparece en el campo `model` de la respuesta, pero no selecciona los pesos. Puede adaptarse al catálogo corporativo sin impacto en memoria o rendimiento.

### Distribución del modelo sobre GPU

#### `QWEN_TENSOR_PARALLEL_SIZE`

Define cuántas GPU colaboran para ejecutar una copia del modelo. Nuestro pod recibe una sola A100, por eso el único valor coherente es `1`.

Aunque el validador acepta hasta `8`, poner `2` con una sola GPU no duplica rendimiento: vLLM intentará crear dos particiones y fallará. Para usar dos GPU habría que asignarlas al mismo pod, comprobar NVLink/PCIe y volver a validar memoria, NCCL y latencia.

#### `QWEN_CPU_OFFLOAD_GB`

Permite trasladar parte de los pesos a RAM y copiarlos por PCIe cuando se necesitan. Es una solución para modelos que no caben en GPU, no una optimización gratuita. Aquí los pesos ocupan aproximadamente 28.9 GiB y caben; `0` evita latencia y tráfico PCIe.

### Presupuesto de VRAM y contexto

#### `QWEN_GPU_MEMORY_UTILIZATION`

Es la fracción de VRAM que vLLM intenta utilizar para pesos, activaciones, kernels y caché KV. No significa que PyTorch reserve exactamente ese porcentaje en todo momento.

- `0.90`: valor seguro y validado.
- `0.92`: experimento razonable si se necesitara más caché.
- `0.95`: límite superior admitido por nuestro wrapper; deja poco margen a procesos del nodo y picos de memoria.
- Un valor menor, como `0.85`, reduce riesgo de OOM pero también la caché disponible.

En SP2 ya se observaron 40.31 GiB de KV y capacidad para 647,288 tokens, muy por encima del contexto configurado. Subir esta variable no hará que una respuesta corta sea más rápida ni permitirá superar el máximo nativo del modelo.

#### `QWEN_MAX_MODEL_LEN`

Es el presupuesto total de tokens de una secuencia:

```text
tokens de instrucciones
+ historial del chat
+ texto recuperado por RAG
+ tokens visuales
+ respuesta generada
```

El valor `262144` es el máximo nativo declarado. No significa que todas las peticiones deban rellenarlo. Un chat RAG normal debería trabajar con presupuestos menores para reducir tiempo de prefill y coste computacional.

Si un Runtime deja menos VRAM libre, reduzca en escalones: `131072`, `65536`, `32768`. No use RoPE scaling para superar 262K sin una evaluación específica de calidad.

#### `QWEN_KV_CACHE_DTYPE`

La caché KV almacena información de atención de los tokens ya procesados. Crece con la longitud y la concurrencia, pero no es lo mismo que la precisión de los pesos.

La A100 es SM80: puede ejecutar los pesos FP8 mediante Marlin, pero Triton no admite en ella KV FP8 nativo. Por eso `bfloat16` no es un conservadurismo opcional sino el valor compatible comprobado. SP3 no cambia la generación de la GPU; una A100 continúa siendo SM80.

### Prefill, batching y concurrencia

#### `QWEN_MAX_NUM_BATCHED_TOKENS`

Es el máximo de tokens que el scheduler procesa juntos en un paso de prefill. Una analogía útil: el contexto completo es un libro y este parámetro es el número de páginas que se llevan a la mesa cada vez.

- Un bloque mayor puede acelerar prompts largos al reducir el número de tandas.
- También eleva el pico de activaciones y puede perjudicar la latencia de otras peticiones.
- No amplía el contexto total.

`8192` quedó validado en SP2. En SP3 proponemos probar `16384`; `32768` solo después de medir memoria, tiempo hasta el primer token y estabilidad.

#### `QWEN_MAX_NUM_SEQS`

Limita cuántas secuencias mantiene vLLM activas simultáneamente. Aunque el log calculó capacidad teórica para 2.47 secuencias máximas, el proxy tiene un lock en `predict` y otro en el worker. Por ello `2` no aumenta el throughput actual: las llamadas siguen entrando de una en una.

Para aprovechar `2` habría que rediseñar IPC, cancelación y concurrencia. Hasta entonces `1` alinea el scheduler con el comportamiento real y prioriza contexto.

#### `QWEN_ENFORCE_EAGER`

Con `true`, vLLM ejecuta las operaciones inmediatamente y evita CUDA Graphs. Es más conservador, arranca con menos compilación y facilitó estabilizar Ampere/SP2.

Con `false`, vLLM puede capturar y reutilizar grafos CUDA:

- posible mejora de latencia y tokens/s;
- warmup más largo;
- consumo adicional de VRAM;
- posibilidad de bloqueos o incompatibilidades de captura.

Es el experimento de rendimiento más interesante para SP3, pero debe probarse solo, sin cambiar a la vez el prefill o la memoria.

### Backends de ejecución

#### `QWEN_ATTENTION_BACKEND`

Selecciona la implementación de atención. `TRITON_ATTN` es la ruta que funcionó con A100 y KV BF16. Cambiarla puede alterar compatibilidad, memoria y kernels; no se recomienda escoger un backend solo porque tenga un nombre más rápido en otra GPU.

#### `VLLM_USE_FLASHINFER_SAMPLER`

Controla únicamente el sampler top-k/top-p de FlashInfer, no la atención. Con el default interno de vLLM, FlashInfer intentó compilar durante el warmup y falló porque no existía `ninja`. Una compilación completa también puede necesitar NVCC y headers CUDA.

Mantenga `0`. En SP3 solo tendría sentido probar `1` después de verificar `ninja`, `nvcc`, toolkit y compatibilidad del kernel. Para una única secuencia, la ganancia esperable es secundaria frente al riesgo.

### Longitud de respuesta

#### `QWEN_MAX_OUTPUT_TOKENS`

Es el techo de seguridad que el endpoint acepta en el campo `max_tokens`; no es la longitud que genera automáticamente.

En el despliegue SP2 funcional se configuró:

```text
QWEN_MAX_OUTPUT_TOKENS=8192
```

Por ejemplo:

- Si la petición omite `max_tokens`, genera como máximo 128 porque ese es el default de petición.
- Si envía `"max_tokens": 1024`, el techo 8192 lo permite.
- Si envía `"max_tokens": 9000`, el wrapper lo rechaza.

Permitir 8192 es seguro desde el punto de vista de validación y contexto, pero una generación tan larga puede superar el timeout del cliente porque Workbench `predict` no hace streaming.

### Imágenes, red y seguridad

#### `QWEN_MAX_IMAGES_PER_PROMPT`

Limita imágenes por conversación. `1` reduce tokens visuales, activaciones y tiempo de prefill. El wrapper acepta hasta `4`, pero esos valores no han sido validados. Vídeo está fijado a cero en el código y no se habilita con esta variable.

#### `VLLM_ALLOWED_MEDIA_DOMAINS`

Es una lista separada por comas de dominios desde los que vLLM puede descargar imágenes remotas. En producción debe actuar como allowlist para evitar que el endpoint se use para consultar direcciones internas. También deben aplicarse controles de tamaño, tipo MIME y tiempo de descarga fuera del modelo.

### Operación del contenedor

#### `QWEN_STARTUP_TIMEOUT_SECONDS`

Es el tiempo que el proxy espera a que el worker descargue/cargue pesos, cree la caché y caliente kernels. No es el timeout de una inferencia y no mejora rendimiento. `1800` segundos da margen al primer arranque; una caché persistente reduce cargas posteriores.

#### `HF_TOKEN` y `HF_HOME`

`HF_TOKEN` evita límites anónimos y permite acceder a repositorios autorizados. Debe almacenarse como secreto de solo lectura. `HF_HOME` decide dónde se guardan pesos y metadatos; conviene apuntarlo a almacenamiento persistente para no descargar el modelo en cada réplica.

## Qué no puede arreglar una variable

Algunos límites pertenecen a capas distintas:

| Límite | Capa que lo impone | Consecuencia |
|---|---|---|
| KV FP8 no disponible | A100 SM80 + backend Triton | Mantener KV BF16 incluso en SP3 |
| FP8 no nativo | Hardware Ampere | Pesos FP8 funcionan mediante Marlin/W8A16, con menor rendimiento que Hopper |
| Una sola GPU | Perfil de recursos | `tensor_parallel_size=1` |
| FlashInfer JIT falla | Runtime sin toolchain `ninja`/NVCC validada | Sampler nativo |
| CUDA 12.9 | Driver y rueda disponible | Rueda vLLM `cu129`, no PyPI CUDA 13 |
| Conflictos NumPy/protobuf | Librerías base de Cloudera | Worker en venv y proceso separado |
| Sin streaming | Contrato Workbench `predict`/PBJ | Una respuesta JSON al finalizar |
| Sin API OpenAI | Tipo de endpoint Workbench | RAG Studio necesita AI Inference o gateway |
| Sin vídeo | Límite explícito del worker | Requiere código, decodificación y pruebas; no una variable actual |
| Concurrencia real igual a uno | Locks del proxy y worker | Subir `MAX_NUM_SEQS` por sí solo no ayuda |

## Streaming: por qué no es un parámetro

vLLM puede producir tokens incrementalmente, pero este endpoint usa la interfaz síncrona `LLM.chat()`. Después el worker construye un JSON completo, lo manda por el socket IPC y `predict(args)` devuelve un único objeto a PBJ. Ninguno de esos tres pasos expone Server-Sent Events o WebSocket al cliente.

Cambiar el worker para enviar tokens parciales no bastaría: la función `@cml_model` seguiría esperando un resultado serializable. Las alternativas son:

1. Desplegar el modelo en **Cloudera AI Inference service**, que expone API OpenAI y `stream=true` cuando el checkpoint esté validado.
2. Crear una **Workbench Application** con FastAPI y SSE/WebSocket, usando un motor asíncrono.
3. Implementar un patrón de trabajo asíncrono más polling; no sería streaming real.

AI Inference SP3 usa vLLM 0.20 y declara `Qwen3_5ForConditionalGeneration`, la arquitectura interna resuelta por el checkpoint. Eso justifica una prueba, pero no equivale a certificación explícita del artefacto Qwen3.8 FP8.

## Plan de mejora para SP3

La regla principal es cambiar **una sola variable cada vez**. Si se modifican eager, prefill y memoria simultáneamente, un fallo no permite saber qué cambio lo causó.

### Fase 0: reproducir la línea base

Desplegar en SP3 exactamente el perfil SP2:

```text
QWEN_MAX_OUTPUT_TOKENS=8192
QWEN_MAX_MODEL_LEN=262144
QWEN_MAX_NUM_BATCHED_TOKENS=8192
QWEN_MAX_NUM_SEQS=1
QWEN_GPU_MEMORY_UTILIZATION=0.90
QWEN_KV_CACHE_DTYPE=bfloat16
QWEN_ENFORCE_EAGER=true
QWEN_ATTENTION_BACKEND=TRITON_ATTN
VLLM_USE_FLASHINFER_SAMPLER=0
```

No se considera válida la comparación hasta que esta base pase texto, contexto largo e imagen.

### Fase 1: bloques de prefill mayores

Cambiar únicamente:

```text
QWEN_MAX_NUM_BATCHED_TOKENS=16384
```

Objetivo: reducir tiempo hasta el primer token con prompts RAG largos. Revertir a `8192` si sube demasiado el pico de memoria, empeora la latencia corta o el motor no inicia.

### Fase 2: CUDA Graphs

Volver a una base conocida y cambiar únicamente:

```text
QWEN_ENFORCE_EAGER=false
```

Objetivo: mejorar latencia de decode y tokens/s. Revertir a `true` ante errores de graph capture, arranque excesivo, OOM o inestabilidad. Si funciona, repetir la prueba combinándolo después con prefill `16384`.

### Fase 3: imágenes

Solo si existe un caso de uso real de comparación visual:

```text
QWEN_MAX_IMAGES_PER_PROMPT=2
```

Medir memoria, tokens visuales y tiempo hasta el primer token con la resolución objetivo. No saltar directamente a cuatro imágenes.

### Fase 4: memoria GPU

Probar `0.92` únicamente si una fase anterior reduce la caché por debajo del contexto requerido:

```text
QWEN_GPU_MEMORY_UTILIZATION=0.92
```

No aporta velocidad por sí mismo. No usar `0.95` sin observar la GPU completa y los procesos residentes.

### Fase 5: evaluar AI Inference

Registrar el checkpoint y comprobar si el servidor gestionado vLLM 0.20 lo acepta. Si funciona, evaluar:

- API `/v1/chat/completions`;
- streaming `stream=true`;
- integración directa con RAG Studio;
- autoscaling y autenticación Knox;
- tool calling y parser de razonamiento;
- imágenes;
- equivalencia de calidad y contexto con Workbench.

Si el servidor no reconoce el checkpoint o la cuantización, conservar Workbench 0.29 o crear una imagen/servidor OpenAI propio; no degradar silenciosamente librerías del entorno funcional.

### Fase 6: concurrencia, que requiere código

Solo después de estabilizar el motor se puede rediseñar el proxy para varias peticiones. Entonces tendría sentido probar `QWEN_MAX_NUM_SEQS=2`. Con el código actual, cambiar esa variable no genera paralelismo útil.

## Cómo medir las mejoras

Para cada fase guarde exactamente el mismo conjunto de pruebas:

| Métrica | Qué enseña |
|---|---|
| Tiempo de arranque | Coste de compilación, descarga y warmup |
| Tiempo hasta primer token | Rendimiento del prefill; con `predict` se estima mediante pruebas controladas |
| Tokens de salida por segundo | Rendimiento de decode |
| Latencia total | Experiencia real del consumidor |
| Pico de VRAM | Margen frente a OOM |
| Capacidad KV informada | Si sigue cabiendo el contexto objetivo |
| Éxito con 262K | Conservación de la capacidad máxima |
| Texto corto y largo | Evita optimizar un único extremo |
| Una imagen real | Comprueba el procesador visual |
| Tres reinicios consecutivos | Detecta fallos no deterministas de warmup |

Una mejora se acepta si aumenta rendimiento sin cambiar la respuesta funcional, sin reducir el contexto requerido y sin introducir fallos en tres arranques. Documente también Runtime, driver, GPU visible, wheel, Torch, CUDA y commit para que la comparación sea reproducible.

## Contrato de `predict`

La función seleccionada en Cloudera debe ser exactamente:

```text
predict
```

La entrada debe ser un objeto JSON. Se admite un chat:

```json
{
  "messages": [
    {"role": "system", "content": "Responde de forma breve."},
    {"role": "user", "content": "¿Qué es Cloudera AI?"}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.8,
  "enable_thinking": false,
  "reasoning_effort": "low"
}
```

También se admite la forma corta:

```json
{"prompt": "Explica RAG en dos frases", "max_tokens": 128}
```

### Parámetros de petición

| Campo | Default | Valores admitidos |
|---|---|---|
| `messages` | ninguno | Array no vacío; roles `system`, `user`, `assistant`, `tool` |
| `prompt` | ninguno | Alternativa de texto a `messages` |
| `max_tokens` | `128` | `1` hasta `QWEN_MAX_OUTPUT_TOKENS` |
| `temperature` | `0.7` sin thinking; `1.0` con thinking | `0.0-2.0` |
| `top_p` | `0.80` sin thinking; `0.95` con thinking | `0.01-1.0` |
| `top_k` | `20` | `-1-1000` |
| `min_p` | `0.0` | `0.0-1.0` |
| `presence_penalty` | `1.5` sin thinking; `0.0` con thinking | `-2.0-2.0` |
| `repetition_penalty` | `1.0` | `0.01-2.0` |
| `seed` | `0` | `0-2147483647` |
| `stop` | ninguno | Valor aceptado por `SamplingParams` |
| `enable_thinking` | `false` | Booleano |
| `preserve_thinking` | `true` | Booleano enviado al chat template |
| `reasoning_effort` | `low` | `low`, `medium`, `xhigh` |

La respuesta tiene forma tipo OpenAI, pero se entrega mediante la API propia de Workbench Models:

```json
{
  "id": "chatcmpl-example",
  "object": "chat.completion",
  "created": 0,
  "model": "qwen3.8-27b-fp8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Cloudera AI es una plataforma empresarial de IA.",
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 12,
    "total_tokens": 32
  }
}
```

Si el modelo emite un bloque `<think>...</think>`, el wrapper coloca ese contenido en `reasoning_content` y deja la respuesta visible en `content`.

## Multimodalidad

El chat template acepta `content` como texto o lista multimodal. El motor limita cada prompt a una imagen y cero vídeos. Si se permiten imágenes por URL, configure `VLLM_ALLOWED_MEDIA_DOMAINS` con una allowlist explícita; no deje acceso abierto a hosts internos. Valide además el tamaño de las imágenes y los timeouts antes de exponer el endpoint.

## Cómo reconocer un arranque correcto

Las líneas relevantes deben aparecer en este orden aproximado:

```text
SHM_DIAGNOSTIC ...
GPU_DIAGNOSTIC name=NVIDIA A100 80GB PCIe ...
VLLM_CONFIG ... max_model_len=262144 ... kv_cache_dtype=bfloat16 ...
Selected MarlinFP8ScaledMMLinearKernel ...
Model loading took ...
Available KV cache memory ...
GPU KV cache size: ... tokens
FlashInfer top-p/top-k sampling disabled ...
```

El warning que indica que la A100 no tiene FP8 nativo es esperado: confirma que se usa compresión weight-only mediante Marlin. Los avisos de deprecación de `cuda.nvrtc`, `cuda.cudart` o `use_fast` no bloquearon el arranque validado.

## Historial de fallos y solución

| Síntoma | Causa | Solución actual |
|---|---|---|
| Build falla rápidamente o usa CUDA 13 | Rueda PyPI no adecuada al driver | Rueda oficial `0.29.0+cu129` explícita |
| Conflictos con Pandas, matplotlib, RAZ o protobuf | vLLM instalado en Python global | `.venv` aislado |
| `Can not perform a '--user' install` | Runtime exporta `PIP_USER=true` | El instalador fija `PIP_USER=false` dentro del build |
| `__file__ is not defined` | PBJ ejecuta el fichero como celda | Búsqueda compatible mediante cwd |
| Inspección de arquitectura falla | vLLM ejecutado mezclando intérpretes | Arquitectura y motor enteramente dentro del worker |
| `numpy.dtype size changed` | NumPy 2 inyectado en proceso con extensiones NumPy 1 | Separación proxy/worker por proceso |
| `FP8 KV cache is not supported ... A100` | A100 SM80 no admite KV FP8 con Triton | KV BF16 y corrección automática de aliases FP8 |
| `No such file or directory: 'ninja'` | Sampler FlashInfer intenta compilación JIT | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Warning NCCL al apagarse | Consecuencia de abortar EngineCore tras otro error | Buscar siempre el primer traceback anterior |
| Descarga anónima de Hugging Face | Falta `HF_TOKEN` | Configurar token/caché persistente |

## Operación y seguridad

- Use un `HF_TOKEN` de solo lectura y guárdelo como secreto, no en Git.
- Para producción, fije `QWEN_MODEL_ID` a un snapshot inmutable o replique el modelo internamente.
- Dimensione el disco para el snapshot, cachés y artefactos temporales; deje al menos 40 GB libres como punto de partida.
- No arranque muchas réplicas simultáneamente contra una caché vacía. Precargue el snapshot.
- Mantenga una réplica durante la validación. Cada réplica necesita su propia A100 completa.
- `predict` no ofrece streaming. Limite `max_tokens` para respetar el timeout de Workbench.
- El socket IPC es local al contenedor, usa framing de 8 bytes y limita cada mensaje a 128 MiB.
- El endpoint serializa la inferencia mediante un lock. Para más throughput, escale réplicas con una GPU por réplica en lugar de aumentar concurrencia sin pruebas.

## Workbench frente a RAG Studio

Este código implementa un endpoint clásico de Workbench Models con `predict`. Aunque la respuesta se parece a OpenAI, no expone directamente `/v1/chat/completions`, `/v1/models`, streaming SSE ni autenticación OpenAI. No debe registrarse en RAG Studio como un endpoint OpenAI sin un adaptador/gateway que traduzca la API de Workbench.

En SP3, AI Inference service puede aportar una API gestionada y compatibilidad OpenAI para los modelos incluidos en su matriz. Este perfil Qwen usa vLLM 0.29 dentro de Workbench porque la versión gestionada debe soportar explícitamente la arquitectura concreta. Verifique la matriz de la versión instalada antes de migrarlo.

## Checklist final

- [ ] Runtime Nvidia GPU Edition con Python 3.10/3.11.
- [ ] A100 80 GB completa, sin MIG.
- [ ] Root Directory vacío o `.`.
- [ ] Build Script Path `cdsw-build.sh`.
- [ ] Build variables `MODEL_FAMILY=qwen3_8` y `GPU_TYPE=a100`.
- [ ] File `qwen3_8/model_a100.py`.
- [ ] Function `predict`.
- [ ] Entrada de ejemplo válida.
- [ ] Salida de red a las fuentes de dependencias durante build.
- [ ] Acceso al modelo o snapshot durante el primer arranque.
- [ ] `VLLM_USE_FLASHINFER_SAMPLER` ausente o igual a `0`.
- [ ] `QWEN_KV_CACHE_DTYPE` ausente o igual a `bfloat16`.
- [ ] Sin modificaciones manuales de `PYTHONPATH` o `VIRTUAL_ENV`.
- [ ] Log confirma Marlin, KV BF16 y capacidad de caché suficiente.

## Fuentes oficiales de referencia

- [Ficha de Qwen3.8-27B-FP8 en Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Repositorio oficial de Qwen3.8](https://github.com/QwenLM/Qwen3.8)
- [Novedades de Cloudera AI on premises 1.5.5 SP3](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-whats-new-1-5-5-sp3.html)
- [Inventario de servidores y arquitecturas de AI Inference SP3](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-model-server-inventory-sp3.html)
- [Argumentos vLLM soportados por AI Inference SP3](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-caii-supported-vllm-command-line-arguments-sp3.html)
- [Interacción y streaming con Model Endpoints de AI Inference](https://docs.cloudera.com/machine-learning/1.5.5/ai-inference/ml-ai-inference.pdf)
