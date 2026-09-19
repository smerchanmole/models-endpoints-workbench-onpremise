# Model endpoints en Cloudera AI Workbench on premises

Proyecto para desplegar Model Endpoints desde Hugging Face en **Cloudera AI on premises 1.5.5 SP2 o SP3**:

- LLM: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`.
- LLM multimodal: `Qwen/Qwen3.8-27B-FP8` con vLLM 0.29.0+cu129 en A100 80 GB.
- Embeddings: `BAAI/bge-m3` (568 M parámetros, 1024 dimensiones, más de 100 idiomas y contexto de 8192 tokens).

Los endpoints usan la función desplegable `predict(args)` y el decorador `@cml_model`. Se conserva `api_wrapper` como alias retrocompatible, pero en nuevos Model Deployments debe seleccionarse siempre **Function = `predict`**.

## Despliegue rápido: embedding BGE-M3 en A100 (SP2)

El repositorio incluye un perfil específico para **NVIDIA A100 de 40 u 80 GB** en Cloudera AI on premises SP2.

| Campo del build/deployment | Valor |
|---|---|
| ML Runtime | Nvidia GPU, Python 3.10 |
| Build variable | `MODEL_FAMILY=embedding` |
| Build variable | `GPU_TYPE=a100` |
| Instalador ejecutado | `embedding/install_a100.sh` |
| File | `embedding/model_a100.py` |
| Function | `predict` |
| Example Input | contenido de `examples/embedding_input.json` |
| GPU | 1 × A100 40/80 GB |
| Variables de ejecución | ninguna obligatoria |

El instalador A100 fija `torch==2.9.1` con CUDA 12.8 para evitar el error observado en SP2 con el driver CUDA 12.9. Es necesario crear un **build nuevo**; reiniciar un build anterior no reemplaza sus dependencias.

> Nemotron dispone actualmente de perfiles L40S y H100. El perfil A100 añadido en este repositorio corresponde al modelo de embeddings BGE-M3.

## Despliegue rápido: Qwen3.8-27B-FP8 en A100 80 GB

Este perfil fija vLLM 0.29.0, una versión actual compatible con Qwen3.8. El instalador no usa la rueda normal de PyPI, porque esa variante está compilada para CUDA 13.0: descarga explícitamente la rueda oficial `0.29.0+cu129`. El perfil está preparado para una **A100 completa de 80 GB**; el código rechaza una GPU con menos de 70 GiB visibles para evitar un arranque que terminaría en OOM.

Las dependencias se instalan en `qwen3_8/.venv`, aisladas del Python global de Cloudera. Esto evita los conflictos de SP2 entre `numpy<2`/`protobuf==4.25.3` del Runtime y las versiones que necesita vLLM 0.29. `model_a100.py` activa automáticamente esos paquetes antes de importar vLLM; no hay que seleccionar otro intérprete ni añadir una variable al Model Deployment.

En el formulario **Deploy model from code**, use rutas desde la raíz del proyecto y no configure un Model Root Directory personalizado:

| Campo del build/deployment | Valor |
|---|---|
| Root Directory / Model Root Directory | vacío o `.` |
| Build Script Path | `cdsw-build.sh` |
| Build variable | `MODEL_FAMILY=qwen3_8` |
| Build variable | `GPU_TYPE=a100` |
| File | `qwen3_8/model_a100.py` |
| Function | `predict` |
| Example Input | contenido de `examples/qwen3_8_input.json` |
| Runtime | Nvidia GPU Edition, Python 3.10 o 3.11, Ubuntu 24.04 |
| GPU | 1 × A100 80 GB, sin MIG |
| CPU / memoria | mínimo 8 vCPU y 64 GiB; recomendado 128 GiB |
| Réplicas | 1 |

El build requiere salida HTTPS a GitHub Releases, PyTorch y PyPI. El primer arranque requiere salida a Hugging Face y unos 40 GB de disco libre para pesos y metadatos. Si el entorno no tiene Internet, publique la rueda y el snapshot en repositorios internos y use `VLLM_WHEEL_URL`, `PYTORCH_INDEX_URL` y `QWEN_MODEL_ID`.

## Elección de precisión y memoria

| Endpoint | GPU | Checkpoint / precisión | Configuración inicial | Motivo |
|---|---|---|---|---|
| Nemotron | L40S 48 GB | FP8, descarga aproximada 32.7 GB | 262K contexto, 1 secuencia, prefill por bloques de 4096 | BF16 no cabe con runtime y cachés |
| Nemotron | H100 80 GB | BF16, pesos aproximados 60 GB | 262K contexto, 1 secuencia, prefill por bloques de 8192 | Máxima calidad dentro de una H100 completa |
| BGE-M3 | L40S | FP16 | 8K contexto, lote 8, máximo 32 | Embedding denso multilingüe para RAG |
| BGE-M3 | A100 40/80 GB | BF16 | 8K contexto, lote 8, máximo 32 | Perfil compatible con SP2 y CUDA 12.9 |
| BGE-M3 | H100 | BF16 | 8K contexto, lote 16, máximo 64 | Mayor lote y rango numérico |
| Qwen3.8 | A100 80 GB | FP8 block-wise; cálculo W8A16/Marlin en Ampere | 262K contexto, 1 secuencia, prefill 8192, KV FP8, eager | El checkpoint ocupa unos 31 GB; el perfil fija vLLM 0.29 |

Las cifras presuponen una GPU **completa**, sin MIG ni una vGPU con menos VRAM. Nemotron declara un máximo de 262144 tokens. Su arquitectura solo tiene 6 capas de atención y 2 cabezas KV, por lo que su caché KV crece mucho menos que la de un Transformer denso de 30B. El prefill por bloques evita procesar los 262K tokens de una vez. Aun así, 262K es un perfil de capacidad, no de baja latencia: debe validarse con el Runtime y driver reales.

En la H100 de 80 GB, BF16 deja poco margen; si el runtime concreto consume más memoria, use el checkpoint FP8 también en H100 cambiando `LLM_MODEL_ID` y `LLM_DTYPE=auto`, `LLM_KV_CACHE_DTYPE=fp8`. Si una L40S no inicia con 262K, reduzca primero a 131072 y después a 65536.

BGE-M3 ocupa aproximadamente 2.27 GB en FP32 y continúa siendo pequeño frente a estas GPU. Se elige frente a E5-small porque amplía el contexto de embedding de 512 a 8192 tokens y ofrece mejor encaje para documentos largos. No requiere prefijos diferentes para consultas y documentos; `input_type` se conserva en la API para mantener explícita la intención. No mezcle vectores creados con modelos, dimensiones o políticas de normalización distintas en el mismo índice: al cambiar desde E5 hay que reconstruir la colección vectorial.

Qwen3.8 declara 262144 tokens nativos y admite texto, imágenes y vídeo. Este perfil deshabilita vídeo y limita cada petición a una imagen para contener memoria. En una A100 Ampere los pesos FP8 usan un kernel compatible W8A16, no el camino FP8 nativo de Hopper. `VLLM_ENFORCE_EAGER=true` y `TRITON_ATTN` son valores conservadores para evitar bloqueos de CUDA graphs y compilación JIT de FlashInfer. Si 262K no deja KV cache suficiente, reduzca `VLLM_MAX_MODEL_LEN` primero a 131072 y luego a 65536.

## Ficheros

```text
.
├── cdsw-build.sh
├── nemotron/
│   ├── install_l40s.sh
│   ├── install_h100.sh
│   ├── model_l40s.py
│   └── model_h100.py
├── embedding/
│   ├── install_l40s.sh
│   ├── install_a100.sh
│   ├── install_h100.sh
│   ├── model_l40s.py
│   ├── model_a100.py
│   └── model_h100.py
├── qwen3_8/
│   ├── install_a100.sh
│   ├── model_a100.py
│   └── requirements.txt
├── examples/
│   ├── nemotron_input.json
│   ├── nemotron_output.json
│   ├── embedding_input.json
│   ├── embedding_output.json
│   ├── qwen3_8_input.json
│   └── qwen3_8_output.json
└── rag_studio/
    ├── embedding_bge_m3.args
    ├── nemotron_h100.args
    └── nemotron_l40s.args
```

Cloudera ejecuta `cdsw-build.sh` en un build limpio; las librerías instaladas manualmente en una sesión no se transfieren al modelo. El dispatcher selecciona el instalador mediante `MODEL_FAMILY` y `GPU_TYPE`.

## Requisitos previos del administrador

1. Nodo x86_64 con L40S 48 GB, A100 80 GB o H100 80 GB, drivers NVIDIA visibles desde Kubernetes y perfil de recursos de 1 GPU completa.
2. ML Runtime **Nvidia GPU Edition**. En SP2 se recomienda Python 3.10; también se admite una versión posterior si existen ruedas compatibles.
3. Salida HTTPS a Hugging Face y PyPI durante el build/arranque, o un mirror interno con los mismos artefactos.
4. Al menos 80 GB libres de disco/caché para L40S y 140 GB para H100 BF16. No incluya los pesos en Git ni en el snapshot del proyecto.
5. Para instalaciones aisladas, precargue las ruedas y el snapshot del modelo en almacenamiento compartido e indique su ruta local mediante `LLM_MODEL_ID`/`EMBEDDING_MODEL_ID`.
6. Acepte la licencia NVIDIA Nemotron aplicable. Configure `HF_TOKEN` como secreto si Hugging Face lo exige o para evitar límites de descarga.

Antes del primer build, el administrador debería comprobar dentro del mismo ML Runtime:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Crear los Model Deployments

Construya **un modelo de Cloudera distinto por endpoint/GPU**. En cada build seleccione la misma Nvidia GPU Runtime que utilizará al desplegar.

### Nemotron en L40S

- Build variables: `MODEL_FAMILY=nemotron`, `GPU_TYPE=l40s`.
- Fichero: `nemotron/model_l40s.py`.
- Función: `predict`.
- Example Input: contenido de `examples/nemotron_input.json`.
- Recursos recomendados: 1 L40S 48 GB, 8 vCPU como mínimo, 64 GB RAM, 1 réplica inicial.

### Nemotron en H100

- Build variables: `MODEL_FAMILY=nemotron`, `GPU_TYPE=h100`.
- Fichero: `nemotron/model_h100.py`.
- Función: `predict`.
- Example Input: contenido de `examples/nemotron_input.json`.
- Recursos recomendados: 1 H100 80 GB, 8-16 vCPU, 96-128 GB RAM, 1 réplica inicial.

### Embeddings en L40S/A100/H100

- Build variables: `MODEL_FAMILY=embedding`, `GPU_TYPE=l40s`, `a100` o `h100`.
- Fichero: `embedding/model_l40s.py`, `embedding/model_a100.py` o `embedding/model_h100.py`.
- Función: `predict`.
- Example Input: contenido de `examples/embedding_input.json`.
- Recursos recomendados: 1 GPU, 2-4 vCPU, 8-16 GB RAM. El modelo también puede funcionar en CPU cambiando el código/dtype, pero estos artefactos están preparados para GPU.

Para el caso observado en SP2 con A100 use exactamente:

- Runtime: Nvidia GPU, Python 3.10.
- Build: `MODEL_FAMILY=embedding`, `GPU_TYPE=a100`.
- File: `embedding/model_a100.py`.
- Function: `predict`.
- Variables de ejecución: ninguna obligatoria.

Las dos variables del build solo permiten que el `cdsw-build.sh` común elija el instalador correcto. No se deben trasladar docenas de parámetros al deployment: el Python ya contiene valores seguros por defecto.

### Qwen3.8-27B-FP8 en A100

- Deje **Root Directory vacío** y use `Build Script Path=cdsw-build.sh`.
- Build variables: `MODEL_FAMILY=qwen3_8`, `GPU_TYPE=a100`.
- Fichero: `qwen3_8/model_a100.py`.
- Función: `predict`.
- Example Input: contenido de `examples/qwen3_8_input.json`.
- Runtime: Nvidia GPU Edition con Python 3.10 o 3.11. No seleccione Standard Edition aunque el formulario permita activar una GPU.
- Recursos: 1 A100 80 GB completa, 8-16 vCPU, 64-128 GiB RAM y 1 réplica.
- No son obligatorias variables de ejecución para la primera prueba. Configure `HF_TOKEN` y `HF_HOME` cuando corresponda.

El código carga el motor al iniciar la réplica para que el estado Ready signifique que los pesos y kernels están utilizables. La primera carga puede tardar varios minutos. El endpoint es no streaming y limita por defecto la salida a 512 tokens para no superar el timeout habitual de Workbench Models.

El build crea automáticamente `qwen3_8/.venv`. No configure manualmente `VIRTUAL_ENV`, `PYTHONPATH` ni cambie el comando de arranque: Workbench puede seguir ejecutando `qwen3_8/model_a100.py` con su Python base, y el propio fichero localiza las librerías aisladas tanto cuando se importa como módulo como cuando PBJ ejecuta su contenido en una celda Jupyter sin `__file__`, conservando acceso a `cml.models_v1`. También dirige al Python del venv el subproceso con el que vLLM inspecciona la arquitectura del modelo.

El primer arranque descarga los pesos. Para evitar arranques lentos y descargas por réplica, monte una caché persistente compartida y configure `HF_HOME` con esa ruta. No use una caché escribible compartida para arrancar muchas réplicas simultáneamente por primera vez: precargue primero el snapshot completo.

## Variables del Model Deployment

### Comunes de build

| Variable | Requerida | Valor | Descripción |
|---|---:|---|---|
| `MODEL_FAMILY` | Sí | `nemotron`, `embedding` o `qwen3_8` | Selecciona dependencias en `cdsw-build.sh` |
| `GPU_TYPE` | Sí | `l40s`, `a100` o `h100` según familia | Selecciona el instalador |
| `VLLM_VERSION` | No | Nemotron `0.12.0`; Qwen `0.29.0` | Nemotron usa su receta estable; Qwen3.8 necesita la rama reciente |
| `SENTENCE_TRANSFORMERS_VERSION` | No | `5.1.2` | Versión fijada para builds reproducibles |
| `TORCH_VERSION` | No | `2.9.1` | Embeddings: se instala desde el índice oficial CUDA 12.8 para evitar incompatibilidad con el driver de SP2 |
| `VLLM_CUDA_VARIANT` | No | Qwen: `129` | Selecciona el artefacto CUDA de vLLM/PyTorch |
| `VLLM_WHEEL_URL` | No | URL oficial calculada | Permite usar una rueda cu129 en un repositorio interno |
| `PYTORCH_INDEX_URL` | No | índice oficial cu129 | Permite sustituir PyTorch por un mirror interno |

### Comunes de ejecución

| Variable | Requerida | Valor recomendado | Descripción |
|---|---:|---|---|
| `HF_TOKEN` | Condicional | secreto de solo lectura | Token de Hugging Face; no imprimirlo en logs |
| `HF_HOME` | Muy recomendada | ruta persistente | Caché de modelos, tokenizadores y metadatos |
| `HF_HUB_OFFLINE` | No | `1` tras precargar | Impide accesos a Internet; solo usar con snapshot completo |
| `TRANSFORMERS_OFFLINE` | No | `1` tras precargar | Complementa el modo offline |

No configure `CUDA_VISIBLE_DEVICES` manualmente: Cloudera/Kubernetes lo proporciona según el perfil de GPU. Tampoco añada el token a Git o al fichero Python.

### Nemotron

| Variable | L40S | H100 | Notas |
|---|---|---|---|
| `LLM_MODEL_ID` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Puede ser una ruta local inmutable |
| `LLM_DTYPE` | no se usa (`auto`) | `bfloat16` | En H100 use `auto` si cambia a FP8 |
| `LLM_KV_CACHE_DTYPE` | `fp8` | `auto` | FP8 ahorra VRAM en L40S |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.90` | `0.92` | Baje a 0.85/0.90 si convive con agentes de monitorización GPU |
| `LLM_MAX_MODEL_LEN` | `262144` | `262144` | Máximo declarado por el checkpoint; perfiles de fallback: 131072, 65536, 32768 |
| `LLM_MAX_NUM_SEQS` | `1` | `1` | Aumentar eleva memoria/throughput; validar carga antes |
| `LLM_MAX_NUM_BATCHED_TOKENS` | `4096` | `8192` | Tamaño del bloque de prefill, no longitud total del chat |
| `LLM_MAMBA_SSM_CACHE_DTYPE` | `float16` | `float32` | L40S prioriza memoria/velocidad; H100 prioriza precisión |
| `LLM_MAX_OUTPUT_TOKENS` | `8192` | `8192` | Límite de seguridad del endpoint |
| `LLM_ENFORCE_EAGER` | `false` | `false` | Ponga `true` si CUDA graphs falla; será más lento |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `1` | no aplica en BF16 | El script L40S ya lo establece por defecto |
| `VLLM_FLASHINFER_MOE_BACKEND` | `throughput` | no aplica en BF16 | Recomendación oficial para FP8 |

### Embeddings

| Variable | L40S | A100 | H100 | Notas |
|---|---|---|---|---|
| `EMBEDDING_MODEL_ID` | `BAAI/bge-m3` | igual | igual | Modelo MIT o ruta local inmutable |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` | `cuda` | Estos artefactos esperan GPU |
| `EMBEDDING_MAX_SEQ_LENGTH` | `8192` | `8192` | `8192` | Máximo real; los chunks normales de RAG deberían ser menores |
| `EMBEDDING_MAX_BATCH_SIZE` | `32` | `32` | `64` | Límite de entrada; el lote efectivo por defecto es 8/8/16 |

Todas estas variables de ejecución son opcionales. Para la primera prueba en A100, no configure ninguna: los valores de la tabla ya son los defaults del código.

### Qwen3.8 en A100 80 GB

| Variable | Default | Notas |
|---|---|---|
| `QWEN_MODEL_ID` | `Qwen/Qwen3.8-27B-FP8` | También puede ser un snapshot local inmutable |
| `QWEN_SERVED_MODEL_NAME` | `qwen3.8-27b-fp8` | Nombre devuelto en la respuesta |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Una A100; evita IPC multiproceso entre GPU |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | No superar `0.95`; baje si hay agentes GPU residentes |
| `VLLM_MAX_MODEL_LEN` | `262144` | Fallbacks recomendados: `131072`, `65536`, `32768` |
| `VLLM_MAX_NUM_SEQS` | `1` | Prioriza contexto sobre concurrencia |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | Bloque de prefill, no longitud total del chat |
| `VLLM_KV_CACHE_DTYPE` | `fp8` | Reduce aproximadamente a la mitad la memoria de KV |
| `VLLM_ENFORCE_EAGER` | `true` | Evita un bloqueo observado en Ampere durante CUDA graph capture |
| `QWEN_ATTENTION_BACKEND` | `TRITON_ATTN` | Evita depender de compilación JIT FlashInfer en el Runtime |
| `QWEN_MAX_OUTPUT_TOKENS` | `512` | Límite del endpoint; puede elevarse hasta 8192 validando timeouts |
| `QWEN_MAX_IMAGES_PER_PROMPT` | `1` | Vídeo está deshabilitado en este perfil |
| `VLLM_ALLOWED_MEDIA_DOMAINS` | vacío | Lista separada por comas para restringir URLs de imágenes |
| `VLLM_CPU_OFFLOAD_GB` | `0` | Mantiene la ejecución en GPU; el offload reduce rendimiento |

No configure `VLLM_VERSION` ni las variables de URL como variables de ejecución: solo se leen durante el build. Para una prueba inicial de texto, bastan las dos variables obligatorias de build (`MODEL_FAMILY`, `GPU_TYPE`).

## Ejemplos para el menú de despliegue

En **New Model / Build Model** utilice estos valores:

| Modelo | File | Function | Example Input |
|---|---|---|---|
| Nemotron L40S | `nemotron/model_l40s.py` | `predict` | copie `examples/nemotron_input.json` |
| Nemotron H100 | `nemotron/model_h100.py` | `predict` | copie `examples/nemotron_input.json` |
| BGE-M3 L40S | `embedding/model_l40s.py` | `predict` | copie `examples/embedding_input.json` |
| BGE-M3 A100 | `embedding/model_a100.py` | `predict` | copie `examples/embedding_input.json` |
| BGE-M3 H100 | `embedding/model_h100.py` | `predict` | copie `examples/embedding_input.json` |
| Qwen3.8 A100 | `qwen3_8/model_a100.py` | `predict` | copie `examples/qwen3_8_input.json` |

El campo **Example Input** debe contener únicamente el objeto JSON, sin las marcas del bloque Markdown. Los ficheros `*_output.json` muestran la salida esperada para documentación o validación; normalmente no se pegan en el campo de entrada.

### Nemotron: entrada para pegar

```json
{
  "messages": [
    {"role": "system", "content": "Responde en español y de forma concisa."},
    {"role": "user", "content": "¿Qué es una arquitectura Mixture of Experts?"}
  ],
  "max_tokens": 256,
  "temperature": 0.2,
  "top_p": 0.95,
  "thinking": false
}
```

También acepta `prompt` y `system_prompt`. La respuesta contiene `text`, `reasoning` (si el modelo emite etiquetas de razonamiento), `finish_reason` y contadores de tokens. El límite `max_tokens` nunca puede superar `LLM_MAX_OUTPUT_TOKENS`.

### Nemotron: salida orientativa

```json
{
  "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
  "context_window": 262144,
  "text": "Una arquitectura Mixture of Experts utiliza varios expertos especializados y activa solo una parte de ellos para cada token.",
  "reasoning": null,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 31,
    "completion_tokens": 25,
    "total_tokens": 56,
    "remaining_context_tokens": 262088
  }
}
```

En H100, `model` mostrará el checkpoint `BF16`. El texto y los contadores pueden variar.

### BGE-M3: entrada para pegar

```json
{
  "input": [
    "Cloudera AI permite desplegar modelos como endpoints.",
    "Nemotron 3 Nano utiliza una arquitectura híbrida MoE."
  ],
  "input_type": "passage",
  "normalize": true,
  "batch_size": 2
}
```

Para buscar esos documentos, genere el vector de la consulta con `input_type=query`. BGE-M3 no añade prefijos; el campo solo documenta el tipo de entrada.

### BGE-M3: salida orientativa

```json
{
  "object": "list",
  "model": "BAAI/bge-m3",
  "dimension": 1024,
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789]
    }
  ]
}
```

La lista está abreviada: cada `embedding` real contiene 1024 números `float`.

### Qwen3.8: entrada para pegar

```json
{
  "messages": [
    {"role": "user", "content": "Explica en una frase qué es Cloudera AI."}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "enable_thinking": false,
  "reasoning_effort": "low"
}
```

También admite `{"prompt":"Hola","max_tokens":64}`. Los valores válidos de `reasoning_effort` son `low`, `medium` y `xhigh`. Qwen recomienda temperatura/top-p `1.0/0.95` para thinking y `0.7/0.80` para respuesta directa; el código aplica esos defaults si no se envían.

### Qwen3.8: salida orientativa

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
        "content": "Cloudera AI es una plataforma para desarrollar, desplegar y operar soluciones de inteligencia artificial sobre datos empresariales.",
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 25,
    "total_tokens": 45
  }
}
```

El texto, identificador, timestamp y contadores varían. Cuando se activa thinking, `reasoning_content` puede contener el razonamiento separado de la respuesta final.

## Integración con RAG Studio

RAG Studio exige que **todos** los endpoints utilicen el estándar OpenAI. Además, cuando usa el proveedor `Cloudera AI`, descubre los endpoints de Cloudera AI Inference service y usa su atributo de tarea:

- Nemotron/Qwen: `TEXT_GENERATION` (o `TEXT_TO_TEXT_GENERATION`), API standard `openai`.
- BGE-M3: `EMBED`, API standard `openai`.

Los `model_*.py` de este proyecto son endpoints clásicos de **Workbench Models** con `predict`; no exponen por sí mismos `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` ni streaming SSE. Por tanto, no deben registrarse directamente en RAG Studio como si fueran OpenAI. Son útiles para consumidores del API de Workbench y para probar la carga en la GPU.

Para RAG Studio, la ruta recomendada es:

1. En SP3, importar Nemotron y BGE-M3 desde Hugging Face en Cloudera AI Registry y desplegarlos con **Cloudera AI Inference service**.
2. Usar los parámetros de `rag_studio/*.args` según GPU.
3. Confirmar que las tareas son `TEXT_GENERATION` y `EMBED`, y que ambos endpoints aparecen correctamente en **Settings > Model Configuration** de RAG Studio.
4. Elegir `Cloudera AI` como proveedor e indicar el dominio de Inference service cuando la interfaz lo solicite.
5. Crear una colección vectorial nueva para BGE-M3; su dimensión es 1024.

Este perfil de Qwen3.8 usa vLLM 0.29.0, mientras que AI Inference service de SP3 incorpora vLLM 0.20.0 y no declara esta arquitectura en su inventario. Por tanto, el perfil Qwen de este repositorio es solo para Workbench Models hasta que Cloudera certifique una versión gestionada compatible. No añada sus argumentos a AI Inference SP3 esperando que el servidor 0.20 pueda cargarla.

En SP2, el Inference service gestionado lleva vLLM 0.8.5 y no soporta la arquitectura `NemotronH` de Nemotron 3. El wrapper de Workbench no soluciona el contrato OpenAI. Para integrar RAG Studio en SP2 se necesita un servidor/adaptador OpenAI-compatible separado que implemente descubrimiento, chat y embeddings, o bien actualizar a SP3. No basta con desactivar streaming.

Aunque Nemotron acepte 262K, para chats RAG conviene comenzar con un presupuesto operativo de 32K-64K: deja espacio para instrucciones, historial, fragmentos recuperados y respuesta, y reduce mucho la latencia de prefill. Use 262K para conversaciones o conjuntos documentales que realmente lo necesiten. RAG Studio permite limitar el número de documentos; 10 fragmentos de 512 tokens consumen aproximadamente 5K tokens antes de instrucciones e historial.

## SP2 frente a SP3

Hay dos rutas de serving diferentes y no conviene mezclarlas:

1. **Este repositorio usa Cloudera AI Workbench Models.** Las dependencias se instalan durante el build mediante `cdsw-build.sh`, por lo que los Python funcionan tanto en SP2 como en SP3 siempre que el Runtime/driver sea compatible con la rueda de vLLM y la GPU sea completa.
2. **Cloudera AI Inference service es el servicio gestionado.** SP2 incluye vLLM 0.8.5, anterior a Nemotron 3 Nano y sin esta arquitectura en su matriz. SP3 incluye Hugging Face Model Server con vLLM 0.20.0, declara `NemotronHForCausalLM` y ofrece NIM de Nemotron 3 Nano 30B. El perfil Qwen3.8 0.29.0 no debe desplegarse con esos servidores gestionados 0.8.5/0.20.0. En SP3, para Nemotron en producción con OpenAI API, streaming y escalado, valore Inference service/NIM en lugar del wrapper de Workbench.

Mejoras operativas relevantes de SP3:

- asignación automática de labels/taints de nodos GPU en ECS al refrescar el perfil, frente a la configuración manual requerida normalmente en SP2;
- vLLM gestionado 0.20.0 y soporte explícito de NemotronH en Inference service;
- NIM de Nemotron 3 Nano disponible en la matriz soportada;
- cuotas de Workbench y controles de acceso más finos.

Estas mejoras **no cambian** los valores de precisión del proyecto: L40S continúa necesitando FP8 y H100 puede usar BF16. Tampoco se debe sustituir `VLLM_VERSION=0.12.0` del build de Workbench por la versión interna de Inference service: son entornos distintos. Si el equipo decide migrar al servicio gestionado de SP3, configure allí los argumentos equivalentes (`--dtype`, `--kv-cache-dtype`, `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, `--trust-remote-code`) y no use estos Python.

## Ajuste y diagnóstico

- **Error observado en SP2 (`driver ... too old`, versión 12090):** no es lentitud ni exceso de variables. PyTorch fue compilado para una versión CUDA posterior a la soportada por el driver. Los instaladores de embeddings fijan ahora `torch==2.9.1` desde el índice `cu128`, que dispone de rueda para Python 3.10 y 3.13 y es compatible con el driver CUDA 12.9. Hay que crear un build nuevo; reiniciar el deployment antiguo no cambia sus dependencias.

- **OOM al iniciar L40S:** confirme que el ID termina en `FP8`, que hay 48 GB visibles y reduzca `LLM_MAX_MODEL_LEN` por escalones: 131072, 65536 y 32768. Si todavía falla, baje `LLM_GPU_MEMORY_UTILIZATION` a `0.85`.
- **OOM al iniciar H100 BF16:** confirme 80 GB sin MIG; use FP8 si el Runtime reserva demasiada VRAM.
- **`No space left on device`:** mueva `HF_HOME` a almacenamiento persistente con capacidad; la caché BF16 y sus temporales necesitan bastante más que el tamaño final de pesos.
- **Error de CUDA/rueda:** el driver del nodo es demasiado antiguo para el PyTorch/CUDA que instala vLLM. Actualice el driver/Runtime o construya un Runtime Add-on validado; no instale un toolkit CUDA diferente dentro del pod para ocultar un driver incompatible.
- **El modelo reinicia tras una petición:** reduzca longitud/lote y revise logs. Cloudera reinicia el modelo cuando `predict` lanza una excepción no controlada.
- **Resultados BGE-M3 pobres:** compruebe normalización consistente, chunks semánticos y que consulta/documentos se hayan generado con exactamente el mismo checkpoint. No reutilice el índice E5 de 384 dimensiones.
- **Build Qwen falla sin una sola línea de `pip`:** el script no llegó a ejecutarse. Deje Root Directory vacío, use `Build Script Path=cdsw-build.sh`, rutas completas para File y una Nvidia GPU Edition realmente instalada en el catálogo.
- **Build Qwen intenta CUDA 13:** confirme que se ejecutó `qwen3_8/install_a100.sh`, que `MODEL_FAMILY=qwen3_8`, `GPU_TYPE=a100` y que no se sobrescribió `VLLM_WHEEL_URL`. El log debe mostrar `cu129`.
- **`pip check` informa conflictos con `langchain-aws`, `pandas`, `matplotlib` o `raz-client`:** está ejecutando una revisión anterior que instalaba vLLM en el Python global. Cree un build nuevo con este código; el instalador actual usa `qwen3_8/.venv` y `pip check` solo valida ese entorno aislado. No fuerce `numpy<2` ni `protobuf==4.25.3` dentro del venv.
- **`Can not perform a '--user' install` al crear el venv:** está usando una revisión anterior del instalador con un Runtime que fuerza `PIP_USER=true`. El script actual lo desactiva solo para las instalaciones dentro de `qwen3_8/.venv`; mantenga `Build Script Path=cdsw-build.sh` y cree un build nuevo.
- **El modelo indica que no existe `qwen3_8/.venv`:** compruebe que el build terminó correctamente, que Root Directory está vacío y que el File es `qwen3_8/model_a100.py`. El entorno se crea en la misma imagen durante `cdsw-build.sh`.
- **`NameError: name '__file__' is not defined`:** está desplegando una revisión anterior del modelo. PBJ ejecuta el Python como una celda Jupyter, no como un módulo. El código actual admite ese modo y localiza el venv desde `/home/cdsw/qwen3_8/.venv`; cree una nueva versión del modelo con el último commit.
- **`Qwen3_5ForConditionalGeneration failed to be inspected`:** la arquitectura sí está incluida en vLLM 0.29. El error aparece cuando su inspector se ejecuta con el Python global de PBJ en lugar del venv. El código actual sustituye el comando interno del inspector para usar `qwen3_8/.venv/bin/python`; no cambie `PYTHONPATH` manualmente.
- **GitHub bloqueado durante el build:** copie la rueda `vllm-0.29.0+cu129` a un repositorio interno y configure `VLLM_WHEEL_URL`; haga lo mismo con PyTorch mediante `PYTORCH_INDEX_URL`.
- **Qwen queda cargando o falla en CUDA graph capture:** conserve `VLLM_ENFORCE_EAGER=true` y `QWEN_ATTENTION_BACKEND=TRITON_ATTN`.
- **Qwen informa KV cache insuficiente:** reduzca `VLLM_MAX_MODEL_LEN` a 131072, 65536 o 32768, sin elevar `VLLM_GPU_MEMORY_UTILIZATION` por encima de 0.95.
- **El endpoint Qwen responde timeout:** reduzca `max_tokens`; Workbench `predict` no hace streaming y la generación puede continuar después de que el cliente abandone la petición.

## Referencias

- [Modelo Nemotron 3 Nano 30B A3B FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- [Receta oficial vLLM para Nemotron 3 Nano](https://github.com/vllm-project/recipes/blob/main/NVIDIA/Nemotron-3-Nano-30B-A3B.md)
- [Modelo BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Modelo Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Qwen3.8: repositorio y despliegue oficial](https://github.com/QwenLM/Qwen3.8)
- [vLLM 0.29.0 y artefactos CUDA 12.9](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [vLLM: selección del attention backend](https://docs.vllm.ai/en/latest/design/attention_backends/)
- [Cloudera: hosting an LLM as a Workbench model](https://docs.cloudera.com/machine-learning/1.5.5/models/topics/ml-creating-a-cloudera-ai-model.html)
- [Cloudera SP3: inventario de servidores y modelos](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-model-server-inventory-sp3.html)
- [Cloudera SP3: novedades](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-whats-new-1-5-5-sp3.html)
- [Cloudera: configuración de RAG Studio](https://docs.cloudera.com/machine-learning/cloud/setup-ai-studios/topics/ml-configuring-rag-studio.html)
