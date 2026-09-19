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

No hace falta declarar variables de ejecución en la primera prueba. El código contiene los valores seguros validados. Si la interfaz separa las variables de build de las variables de ejecución, `MODEL_FAMILY` y `GPU_TYPE` deben estar disponibles durante el **build**.

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

## Variables de ejecución

Para la configuración validada pueden omitirse todas. Solo defina una variable cuando quiera cambiar conscientemente el default.

| Variable | Default | Rango/uso |
|---|---|---|
| `QWEN_MODEL_ID` | `Qwen/Qwen3.8-27B-FP8` | ID de Hugging Face o snapshot local |
| `QWEN_SERVED_MODEL_NAME` | `qwen3.8-27b-fp8` | Nombre devuelto en el JSON |
| `QWEN_TENSOR_PARALLEL_SIZE` | `1` | `1-8`; este perfil solo está validado con una GPU |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.90` | `0.50-0.95` |
| `QWEN_MAX_MODEL_LEN` | `262144` | `2048-262144` |
| `QWEN_MAX_NUM_SEQS` | `1` | `1-64`; el proxy serializa las peticiones |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | `8192` | `2048-131072`; tamaño del bloque de prefill |
| `QWEN_KV_CACHE_DTYPE` | `bfloat16` | En A100 debe terminar siendo BF16 |
| `QWEN_ENFORCE_EAGER` | `true` | Evita CUDA graphs en este perfil conservador |
| `QWEN_ATTENTION_BACKEND` | `TRITON_ATTN` | Backend validado en A100 |
| `QWEN_CPU_OFFLOAD_GB` | `0` | `0-128`; no recomendado si cabe en GPU |
| `QWEN_MAX_OUTPUT_TOKENS` | `512` | `1-8192`; techo admitido por `predict` |
| `QWEN_MAX_IMAGES_PER_PROMPT` | `1` | `0-4`; vídeo permanece deshabilitado |
| `QWEN_STARTUP_TIMEOUT_SECONDS` | `1800` | `60-7200`; espera del proxy al worker |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Mantener `0` en el Runtime validado |
| `VLLM_ALLOWED_MEDIA_DOMAINS` | vacío | Dominios de imágenes remotas separados por comas |
| `HF_TOKEN` | vacío | Recomendado para evitar límites anónimos de Hugging Face |
| `HF_HOME` | default de Hugging Face | Caché persistente con espacio suficiente |

Los antiguos nombres `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_KV_CACHE_DTYPE`, `VLLM_ENFORCE_EAGER`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_TENSOR_PARALLEL_SIZE` y `VLLM_CPU_OFFLOAD_GB` se aceptan como alias. El worker los captura y elimina antes de importar vLLM porque no son variables oficiales de vLLM 0.29 y, si permanecen, generan `Unknown vLLM environment variable`. Use los nombres `QWEN_*` en despliegues nuevos.

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
