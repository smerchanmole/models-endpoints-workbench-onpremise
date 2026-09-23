# Model endpoints en Cloudera AI Workbench on premises

[Documentación general en español](#model-endpoints-en-cloudera-ai-workbench-on-premises) · [General documentation in English](#english-version-model-endpoints-on-cloudera-ai-workbench-on-premises) · [Guía detallada Qwen en español](#guía-detallada-qwen38-27b-fp8-en-cloudera-ai-workbench-sp2sp3) · [Detailed Qwen guide in English](#detailed-qwen38-27b-fp8-guide-for-cloudera-ai-workbench-sp2sp3)

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

Las dependencias se instalan en `qwen3_8/.venv`, aisladas del Python global de Cloudera. Esto evita los conflictos de SP2 entre `numpy<2`/`protobuf==4.25.3` del Runtime y las versiones que necesita vLLM 0.29. `model_a100.py` permanece en el Python base para poder importar `cml.models_v1` y lanza `worker_a100.py` con el intérprete aislado; no hay que seleccionar otro intérprete ni modificar `PYTHONPATH`.

La guía completa bilingüe —español primero e inglés después—, incluida la configuración validada, arquitectura, API, lectura de logs y resolución de todos los errores encontrados, está al final de este mismo documento.

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
| Qwen3.8 | A100 80 GB | Pesos FP8 block-wise; cálculo W8A16/Marlin en Ampere | 262K contexto, 1 secuencia, prefill 8192, KV BF16, eager | A100 SM80 no admite KV FP8 con Triton; el checkpoint ocupa unos 31 GB |

Las cifras presuponen una GPU **completa**, sin MIG ni una vGPU con menos VRAM. Nemotron declara un máximo de 262144 tokens. Su arquitectura solo tiene 6 capas de atención y 2 cabezas KV, por lo que su caché KV crece mucho menos que la de un Transformer denso de 30B. El prefill por bloques evita procesar los 262K tokens de una vez. Aun así, 262K es un perfil de capacidad, no de baja latencia: debe validarse con el Runtime y driver reales.

En la H100 de 80 GB, BF16 deja poco margen; si el runtime concreto consume más memoria, use el checkpoint FP8 también en H100 cambiando `LLM_MODEL_ID` y `LLM_DTYPE=auto`, `LLM_KV_CACHE_DTYPE=fp8`. Si una L40S no inicia con 262K, reduzca primero a 131072 y después a 65536.

BGE-M3 ocupa aproximadamente 2.27 GB en FP32 y continúa siendo pequeño frente a estas GPU. Se elige frente a E5-small porque amplía el contexto de embedding de 512 a 8192 tokens y ofrece mejor encaje para documentos largos. No requiere prefijos diferentes para consultas y documentos; `input_type` se conserva en la API para mantener explícita la intención. No mezcle vectores creados con modelos, dimensiones o políticas de normalización distintas en el mismo índice: al cambiar desde E5 hay que reconstruir la colección vectorial.

Qwen3.8 declara 262144 tokens nativos y admite texto, imágenes y vídeo. Este perfil deshabilita vídeo y limita cada petición a una imagen para contener memoria. En una A100 Ampere los pesos FP8 usan un kernel compatible W8A16/Marlin, pero la caché KV debe ser BF16: Triton requiere SM89 o posterior para KV FP8 y la A100 es SM80. `QWEN_ENFORCE_EAGER=true` y `TRITON_ATTN` son valores conservadores para evitar bloqueos de CUDA graphs y compilación JIT de FlashInfer. Si 262K no deja caché suficiente, reduzca `QWEN_MAX_MODEL_LEN` primero a 131072 y luego a 65536.

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
│   ├── worker_a100.py
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

El código carga el motor al iniciar la réplica para que el estado Ready signifique que los pesos y kernels están utilizables. La primera carga puede tardar varios minutos. El endpoint es no streaming. El código usa un techo conservador de 512 tokens, pero el despliegue SP2 validado configuró `QWEN_MAX_OUTPUT_TOKENS=8192`; cada petición sigue usando 128 tokens si omite `max_tokens`.

El build crea automáticamente `qwen3_8/.venv`. No configure manualmente `VIRTUAL_ENV`, `PYTHONPATH` ni cambie el comando de arranque. Workbench ejecuta `model_a100.py` con el Python base para conservar `cml.models_v1`; este proxy inicia un único `worker_a100.py` persistente con `qwen3_8/.venv/bin/python` y comunica las peticiones por un socket local. De este modo NumPy 2, protobuf y las extensiones binarias de vLLM nunca se mezclan con NumPy 1.x, Pandas o RAZ del Runtime de Cloudera. La réplica no queda Ready hasta que el worker termina de cargar el modelo.

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
| `QWEN_TENSOR_PARALLEL_SIZE` | `1` | Una A100; evita IPC multiproceso entre GPU |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.90` | No superar `0.95`; baje si hay agentes GPU residentes |
| `QWEN_MAX_MODEL_LEN` | `262144` | Fallbacks recomendados: `131072`, `65536`, `32768` |
| `QWEN_MAX_NUM_SEQS` | `1` | Prioriza contexto sobre concurrencia |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | `8192` | Bloque de prefill, no longitud total del chat |
| `QWEN_KV_CACHE_DTYPE` | `bfloat16` | Obligatorio con Triton en A100 SM80; FP8 KV necesita SM89+ |
| `QWEN_ENFORCE_EAGER` | `true` | Evita un bloqueo observado en Ampere durante CUDA graph capture |
| `QWEN_ATTENTION_BACKEND` | `TRITON_ATTN` | Evita depender de compilación JIT FlashInfer en el Runtime |
| `QWEN_MAX_OUTPUT_TOKENS` | código `512`; SP2 validado `8192` | Es el techo de seguridad; cada petición decide su salida con `max_tokens` (default 128) |
| `QWEN_MAX_IMAGES_PER_PROMPT` | `1` | Vídeo está deshabilitado en este perfil |
| `VLLM_ALLOWED_MEDIA_DOMAINS` | vacío | Lista separada por comas para restringir URLs de imágenes |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Usa el sampler nativo y evita que FlashInfer necesite `ninja`/NVCC durante el warmup |
| `QWEN_CPU_OFFLOAD_GB` | `0` | Mantiene la ejecución en GPU; el offload reduce rendimiento |
| `QWEN_STARTUP_TIMEOUT_SECONDS` | `1800` | Espera máxima del proxy a que el worker aislado cargue pesos y kernels |

No configure `VLLM_VERSION` ni las variables de URL como variables de ejecución: solo se leen durante el build. Para una prueba inicial de texto, bastan las dos variables obligatorias de build (`MODEL_FAMILY`, `GPU_TYPE`).

Las variables antiguas `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_KV_CACHE_DTYPE`, `VLLM_ENFORCE_EAGER` y `VLLM_CPU_OFFLOAD_GB` siguen aceptándose como alias. Use los nombres `QWEN_*` de la tabla para evitar los avisos `Unknown vLLM environment variable`. En A100, una petición heredada de KV FP8 se corrige automáticamente a BF16.

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

Este perfil de Qwen3.8 usa vLLM 0.29.0 dentro de Workbench. AI Inference service de SP3 incorpora vLLM 0.20.0 y declara `Qwen3_5ForConditionalGeneration`, la arquitectura interna que resuelve este checkpoint, pero no certifica individualmente `Qwen/Qwen3.8-27B-FP8` en la matriz. Por tanto, una migración al servicio gestionado debe tratarse como una prueba de compatibilidad: no copie sin más las variables `QWEN_*`, porque AI Inference utiliza argumentos propios del servidor.

En SP2, el Inference service gestionado lleva vLLM 0.8.5 y no soporta la arquitectura `NemotronH` de Nemotron 3. El wrapper de Workbench no soluciona el contrato OpenAI. Para integrar RAG Studio en SP2 se necesita un servidor/adaptador OpenAI-compatible separado que implemente descubrimiento, chat y embeddings, o bien actualizar a SP3. No basta con desactivar streaming.

Aunque Nemotron acepte 262K, para chats RAG conviene comenzar con un presupuesto operativo de 32K-64K: deja espacio para instrucciones, historial, fragmentos recuperados y respuesta, y reduce mucho la latencia de prefill. Use 262K para conversaciones o conjuntos documentales que realmente lo necesiten. RAG Studio permite limitar el número de documentos; 10 fragmentos de 512 tokens consumen aproximadamente 5K tokens antes de instrucciones e historial.

## SP2 frente a SP3

Hay dos rutas de serving diferentes y no conviene mezclarlas:

1. **Este repositorio usa Cloudera AI Workbench Models.** Las dependencias se instalan durante el build mediante `cdsw-build.sh`, por lo que los Python funcionan tanto en SP2 como en SP3 siempre que el Runtime/driver sea compatible con la rueda de vLLM y la GPU sea completa.
2. **Cloudera AI Inference service es el servicio gestionado.** SP2 incluye vLLM 0.8.5, anterior a Nemotron 3 Nano y sin esta arquitectura en su matriz. SP3 incluye Hugging Face Model Server con vLLM 0.20.0, declara `NemotronHForCausalLM`, ofrece NIM de Nemotron 3 Nano 30B y declara `Qwen3_5ForConditionalGeneration`. Nemotron dispone por ello de una ruta gestionada clara. Para Qwen3.8, la arquitectura coincide pero el checkpoint FP8 concreto debe validarse antes de migrar; no se deben trasladar automáticamente las recetas de Workbench vLLM 0.29 al servidor gestionado 0.20.

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
- **`Qwen3_5ForConditionalGeneration failed to be inspected`:** la arquitectura sí está incluida en vLLM 0.29. En el diseño actual la inspección ocurre dentro de `worker_a100.py`, ejecutado enteramente por el Python del venv; el build comprueba previamente que la clase se puede importar. Cree un build nuevo y no cambie `PYTHONPATH` manualmente.
- **`numpy.dtype size changed` (`Expected 96 ... got 88`):** está ejecutando una revisión que inyecta NumPy 2 del venv dentro del proceso PBJ, donde Cloudera ya cargó extensiones compiladas para NumPy 1.x. El diseño actual no modifica `sys.path`: `model_a100.py` permanece en el Runtime base y vLLM se ejecuta en `worker_a100.py` mediante el Python aislado. Cree un build nuevo; no intente resolverlo bajando NumPy dentro del venv.
- **GitHub bloqueado durante el build:** copie la rueda `vllm-0.29.0+cu129` a un repositorio interno y configure `VLLM_WHEEL_URL`; haga lo mismo con PyTorch mediante `PYTORCH_INDEX_URL`.
- **Qwen queda cargando o falla en CUDA graph capture:** conserve `QWEN_ENFORCE_EAGER=true` y `QWEN_ATTENTION_BACKEND=TRITON_ATTN`.
- **Qwen indica `FP8 KV cache is not supported ... A100`:** use `QWEN_KV_CACHE_DTYPE=bfloat16` o elimine la variable antigua `VLLM_KV_CACHE_DTYPE=fp8`. El código actual aplica BF16 automáticamente en GPU con capacidad inferior a SM89.
- **Qwen falla con `No such file or directory: 'ninja'` durante el warmup:** conserve `VLLM_USE_FLASHINFER_SAMPLER=0`. Solo desactiva el sampler top-k/top-p de FlashInfer; la atención continúa en Triton y los pesos FP8 en Marlin. Así no depende de compilación JIT, `ninja` ni NVCC del Runtime.
- **Qwen informa KV cache insuficiente:** reduzca `QWEN_MAX_MODEL_LEN` a 131072, 65536 o 32768, sin elevar `QWEN_GPU_MEMORY_UTILIZATION` por encima de 0.95.
- **El endpoint Qwen responde timeout:** reduzca `max_tokens`; Workbench `predict` no hace streaming y la generación puede continuar después de que el cliente abandone la petición.

## Referencias

- [Modelo Nemotron 3 Nano 30B A3B FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- [Receta oficial vLLM para Nemotron 3 Nano](https://github.com/vllm-project/recipes/blob/main/NVIDIA/Nemotron-3-Nano-30B-A3B.md)
- [Modelo BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Modelo Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Qwen3.8: repositorio y despliegue oficial](https://github.com/QwenLM/Qwen3.8)

---

# English version: Model endpoints on Cloudera AI Workbench on premises

[Spanish overview](#model-endpoints-en-cloudera-ai-workbench-on-premises) · [Detailed Qwen guide in Spanish](#guía-detallada-qwen38-27b-fp8-en-cloudera-ai-workbench-sp2sp3) · [Detailed Qwen guide in English](#detailed-qwen38-27b-fp8-guide-for-cloudera-ai-workbench-sp2sp3)

This repository contains self-contained model deployments for Cloudera AI Workbench on premises SP2 and SP3:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on L40S and H100.
- `BAAI/bge-m3` multilingual embeddings on L40S, A100, and H100.
- `Qwen/Qwen3.8-27B-FP8` with vLLM 0.29.0+cu129 on A100 80 GB.

All Workbench endpoints register `predict(args)` through `@cml_model`. `api_wrapper` remains only as a compatibility alias; select **Function = `predict`** for new deployments.

## Deployment matrix

| Model | GPU | Model file | Build variables | Precision and initial profile |
|---|---|---|---|---|
| Nemotron | L40S 48 GB | `nemotron/model_l40s.py` | `MODEL_FAMILY=nemotron`, `GPU_TYPE=l40s` | FP8, 262K context, one sequence |
| Nemotron | H100 80 GB | `nemotron/model_h100.py` | `MODEL_FAMILY=nemotron`, `GPU_TYPE=h100` | BF16, 262K context, one sequence |
| BGE-M3 | L40S | `embedding/model_l40s.py` | `MODEL_FAMILY=embedding`, `GPU_TYPE=l40s` | FP16, 8192-token inputs |
| BGE-M3 | A100 | `embedding/model_a100.py` | `MODEL_FAMILY=embedding`, `GPU_TYPE=a100` | BF16, 8192-token inputs |
| BGE-M3 | H100 | `embedding/model_h100.py` | `MODEL_FAMILY=embedding`, `GPU_TYPE=h100` | BF16, larger batch |
| Qwen3.8 | A100 80 GB | `qwen3_8/model_a100.py` | `MODEL_FAMILY=qwen3_8`, `GPU_TYPE=a100` | FP8 weights through Marlin, BF16 KV, 262K context |

For every deployment, leave Root Directory empty or set it to `.`, use `cdsw-build.sh` as Build Script Path, and select the model file from the table. Use a Nvidia GPU Edition Runtime with Python 3.10 or 3.11. A full GPU is required; the figures do not assume MIG or vGPU partitions.

## Repository layout

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
│   ├── requirements.txt
│   ├── model_a100.py
│   └── worker_a100.py
├── examples/
│   ├── nemotron_input.json
│   ├── nemotron_output.json
│   ├── embedding_input.json
│   ├── embedding_output.json
│   ├── qwen3_8_input.json
│   └── qwen3_8_output.json
└── rag_studio/
    ├── nemotron_l40s.args
    ├── nemotron_h100.args
    └── embedding_bge_m3.args
```

Cloudera runs `cdsw-build.sh` in a clean build. Packages installed manually in an interactive session are not transferred to the model image. The dispatcher selects the correct installer using `MODEL_FAMILY` and `GPU_TYPE`.

## Prerequisites

- Outbound HTTPS access to Hugging Face, PyPI, PyTorch indexes, and GitHub Releases, or internal mirrors for those artifacts.
- A persistent Hugging Face cache with enough disk capacity.
- A read-only `HF_TOKEN` stored as a secret when required.
- A compatible NVIDIA driver and a full GPU visible to the pod.
- At least 8 vCPU and 64 GiB RAM for Qwen; 16 vCPU and 128 GiB are recommended.

Do not commit access tokens. For repeatable production deployments, pin an immutable Hugging Face snapshot or mirror the model internally.

## Common build variables

| Variable | Required | Purpose |
|---|---:|---|
| `MODEL_FAMILY` | Yes | `nemotron`, `embedding`, or `qwen3_8` |
| `GPU_TYPE` | Yes | Selects the matching GPU installer |
| `HF_TOKEN` | When required | Authenticates model downloads; store as a secret |
| `HF_HOME` | Recommended | Persistent Hugging Face cache path |
| `VLLM_VERSION` | No | Build-time vLLM override; do not use it as a runtime variable |
| `VLLM_WHEEL_URL` | No | Internal mirror or explicit vLLM wheel |
| `PYTORCH_INDEX_URL` | No | Internal or CUDA-specific PyTorch index |

## Nemotron runtime parameters

| Variable | L40S default | H100 default | Meaning |
|---|---:|---:|---|
| `LLM_MODEL_ID` | FP8 checkpoint | BF16 checkpoint | Model ID or immutable local snapshot |
| `LLM_DTYPE` | `auto` | `bfloat16` | Weight and compute dtype selection |
| `LLM_KV_CACHE_DTYPE` | `fp8` | `bfloat16` | KV-cache precision |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.92` | `0.90` | vLLM GPU-memory budget |
| `LLM_MAX_MODEL_LEN` | `262144` | `262144` | Maximum total context |
| `LLM_MAX_NUM_SEQS` | `1` | `1` | Active sequence count |
| `LLM_MAX_NUM_BATCHED_TOKENS` | `4096` | `8192` | Prefill block size, not total context |

L40S uses FP8 because BF16 weights and runtime caches do not fit safely in 48 GB. H100 can use BF16 for maximum fidelity. If startup fails, reduce context to 131072, 65536, and then 32768 before making unrelated changes.

## BGE-M3 runtime parameters

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL_ID` | `BAAI/bge-m3` | Model ID or immutable snapshot |
| `EMBEDDING_DEVICE` | `cuda` | These profiles expect GPU execution |
| `EMBEDDING_MAX_SEQ_LENGTH` | `8192` | Maximum embedding input length |
| `EMBEDDING_MAX_BATCH_SIZE` | L40S/A100 `32`, H100 `64` | Endpoint batch safety ceiling |

BGE-M3 produces 1024-dimensional dense multilingual vectors. Do not mix embeddings from another model, dimension, or normalization policy in the same vector index. Rebuild the collection when migrating from E5 or another embedding model.

## Qwen runtime summary

The tested A100 configuration uses one full A100 80 GB, FP8 weights through Marlin/W8A16, BF16 KV cache, Triton attention, one active sequence, 262144 context tokens, and an 8192-token prefill block. The validated SP2 deployment set `QWEN_MAX_OUTPUT_TOKENS=8192`; this is a ceiling, while each request defaults to 128 output tokens when `max_tokens` is omitted.

Qwen dependencies are installed into `qwen3_8/.venv`. `model_a100.py` stays in Cloudera's base Python so it can import `cml.models_v1`, while `worker_a100.py` runs with the isolated interpreter. This prevents NumPy 2, protobuf, Torch, and vLLM from colliding with binary packages in the SP2 Runtime.

The complete parameter-by-parameter Qwen explanation, SP2 evidence, SP3 tuning plan, streaming limitations, and troubleshooting guide are located at the end of this README in both Spanish and English.

## Example inputs

Select `predict` and paste the matching file from `examples/` into Example Input. Minimal forms are:

Nemotron:

```json
{
  "messages": [{"role": "user", "content": "Explain RAG in one sentence."}],
  "max_tokens": 128,
  "temperature": 0.2
}
```

BGE-M3:

```json
{
  "texts": ["First document", "Second document"],
  "normalize": true,
  "input_type": "document"
}
```

Qwen3.8:

```json
{
  "messages": [{"role": "user", "content": "What is Cloudera AI?"}],
  "max_tokens": 128,
  "temperature": 0.7,
  "enable_thinking": false,
  "reasoning_effort": "low"
}
```

The output examples are illustrative: generated text, IDs, timestamps, token counts, and embedding values vary.

## RAG Studio and streaming

The Python files in this repository implement classic Workbench Models with `predict`. They do not expose `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, or SSE streaming. Their OpenAI-like JSON shape does not turn them into OpenAI endpoints.

RAG Studio should use Cloudera AI Inference endpoints or a separate OpenAI-compatible gateway. In SP3, AI Inference uses managed model servers, supports OpenAI-style text-generation endpoints, and can provide streaming. The exact Qwen3.8 FP8 checkpoint must still be validated even though the inventory lists its internal `Qwen3_5ForConditionalGeneration` architecture.

For RAG chats, start with an operational context budget of 32K–64K even when a model supports 262K. This leaves room for instructions, history, retrieved chunks, and output while keeping prefill latency manageable.

## SP2 versus SP3

This repository uses Workbench builds and therefore controls its own Python dependencies. The same scripts can run on SP2 or SP3 when the selected Runtime, driver, and GPU remain compatible with the pinned wheels.

Cloudera AI Inference is a separate managed serving path. SP2 uses an older vLLM server. SP3 uses Hugging Face Model Server with vLLM 0.20, lists `NemotronHForCausalLM` and `Qwen3_5ForConditionalGeneration`, and offers a Nemotron 3 Nano NIM. Do not copy Workbench `QWEN_*` or `LLM_*` variables directly into the managed service; translate them to its supported command-line arguments and validate the checkpoint.

Relevant SP3 operational improvements include GPU quota management, finer endpoint access control, improved GPU node labeling/taints, diagnostic bundles, and managed inference options. They do not change the physical limits of L40S, A100, or H100.

## Troubleshooting summary

- CUDA driver errors mean the PyTorch/vLLM wheel targets a newer CUDA ABI than the node driver; use the pinned compatible wheel or update the Runtime/driver.
- `numpy.dtype size changed` means incompatible NumPy ABIs were mixed; keep Qwen in its isolated worker process.
- `Can not perform a '--user' install` means `PIP_USER=true` leaked into the virtual environment; the current installer neutralizes it.
- `FP8 KV cache is not supported ... A100` requires BF16 KV on SM80.
- `No such file or directory: 'ninja'` comes from FlashInfer sampler JIT; keep `VLLM_USE_FLASHINFER_SAMPLER=0`.
- OOM requires checking visible GPU memory and reducing context or batch size before increasing GPU utilization.
- A Workbench timeout can occur while generation continues because `predict` buffers the complete non-streaming response.

## English references

- [Nemotron 3 Nano 30B A3B FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- [Official Nemotron vLLM recipe](https://github.com/vllm-project/recipes/blob/main/NVIDIA/Nemotron-3-Nano-30B-A3B.md)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Official Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8)

---

## Guía detallada Qwen3.8-27B-FP8 en Cloudera AI Workbench SP2/SP3


Esta guía documenta el perfil probado de `Qwen/Qwen3.8-27B-FP8` como **Workbench Model** en una NVIDIA A100 PCIe de 80 GB. El endpoint usa la función `predict(args)` de Cloudera y devuelve una respuesta JSON similar a Chat Completions.

### Estado validado

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

### Configuración mínima que funciona

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

### Qué hace cada fichero

| Fichero | Responsabilidad |
|---|---|
| `cdsw-build.sh` | Selecciona el instalador usando `MODEL_FAMILY` y `GPU_TYPE` |
| `qwen3_8/install_a100.sh` | Crea `.venv`, instala la rueda vLLM cu129 y valida versiones/imports |
| `qwen3_8/requirements.txt` | Fija Transformers para builds reproducibles |
| `qwen3_8/model_a100.py` | Punto de entrada PBJ; importa Cloudera y actúa como proxy ligero |
| `qwen3_8/worker_a100.py` | Proceso aislado que importa vLLM, carga el modelo y genera respuestas |
| `examples/qwen3_8_input.json` | Entrada lista para pegar en el formulario |
| `examples/qwen3_8_output.json` | Forma orientativa de la respuesta |

### Por qué existe un virtualenv aislado

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

### Por qué se usa la rueda cu129

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

### Precisión, memoria y contexto

Hay que distinguir los pesos de la caché KV:

- **Pesos FP8:** válidos en A100 mediante el kernel Marlin weight-only FP8/W8A16. La A100 no tiene cálculo FP8 nativo y puede rendir menos que Hopper, pero el modelo carga correctamente.
- **Caché KV FP8:** no válida con Triton en A100 SM80. El backend exige SM89 o posterior.
- **Caché KV BF16:** configuración validada. El código la usa por defecto y transforma automáticamente una configuración heredada `fp8` en `bfloat16` cuando detecta una GPU anterior a SM89.
- **Contexto:** `262144` tokens quedó validado con 40.31 GiB de caché y capacidad calculada de 647,288 tokens.
- **Prefill:** `8192` limita el bloque procesado simultáneamente; no reduce la longitud total del chat.
- **Concurrencia:** el endpoint serializa las llamadas y usa `max_num_seqs=1` para priorizar contexto y estabilidad.

Si otro Runtime deja menos VRAM libre, reduzca `QWEN_MAX_MODEL_LEN` en este orden: `131072`, `65536`, `32768`. No aumente `QWEN_GPU_MEMORY_UTILIZATION` por encima de `0.95`.

### Triton, FlashInfer y ninja

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

### Variables de build

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

### Variables de ejecución: mapa completo

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

### Explicación didáctica de cada parámetro

#### Identidad y procedencia del modelo

##### `QWEN_MODEL_ID`

Indica de dónde se descargan configuración, tokenizer y pesos. El valor actual apunta a Hugging Face:

```text
Qwen/Qwen3.8-27B-FP8
```

En producción es preferible usar un snapshot fijado o una copia interna. Si se deja una rama móvil como `main`, dos builds realizados en fechas diferentes podrían descargar revisiones distintas. Cambiar esta variable por otro modelo no garantiza compatibilidad: el código, la cuantización, la arquitectura y la VRAM se validaron únicamente con este checkpoint.

##### `QWEN_SERVED_MODEL_NAME`

Es una etiqueta lógica. Aparece en el campo `model` de la respuesta, pero no selecciona los pesos. Puede adaptarse al catálogo corporativo sin impacto en memoria o rendimiento.

#### Distribución del modelo sobre GPU

##### `QWEN_TENSOR_PARALLEL_SIZE`

Define cuántas GPU colaboran para ejecutar una copia del modelo. Nuestro pod recibe una sola A100, por eso el único valor coherente es `1`.

Aunque el validador acepta hasta `8`, poner `2` con una sola GPU no duplica rendimiento: vLLM intentará crear dos particiones y fallará. Para usar dos GPU habría que asignarlas al mismo pod, comprobar NVLink/PCIe y volver a validar memoria, NCCL y latencia.

##### `QWEN_CPU_OFFLOAD_GB`

Permite trasladar parte de los pesos a RAM y copiarlos por PCIe cuando se necesitan. Es una solución para modelos que no caben en GPU, no una optimización gratuita. Aquí los pesos ocupan aproximadamente 28.9 GiB y caben; `0` evita latencia y tráfico PCIe.

#### Presupuesto de VRAM y contexto

##### `QWEN_GPU_MEMORY_UTILIZATION`

Es la fracción de VRAM que vLLM intenta utilizar para pesos, activaciones, kernels y caché KV. No significa que PyTorch reserve exactamente ese porcentaje en todo momento.

- `0.90`: valor seguro y validado.
- `0.92`: experimento razonable si se necesitara más caché.
- `0.95`: límite superior admitido por nuestro wrapper; deja poco margen a procesos del nodo y picos de memoria.
- Un valor menor, como `0.85`, reduce riesgo de OOM pero también la caché disponible.

En SP2 ya se observaron 40.31 GiB de KV y capacidad para 647,288 tokens, muy por encima del contexto configurado. Subir esta variable no hará que una respuesta corta sea más rápida ni permitirá superar el máximo nativo del modelo.

##### `QWEN_MAX_MODEL_LEN`

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

##### `QWEN_KV_CACHE_DTYPE`

La caché KV almacena información de atención de los tokens ya procesados. Crece con la longitud y la concurrencia, pero no es lo mismo que la precisión de los pesos.

La A100 es SM80: puede ejecutar los pesos FP8 mediante Marlin, pero Triton no admite en ella KV FP8 nativo. Por eso `bfloat16` no es un conservadurismo opcional sino el valor compatible comprobado. SP3 no cambia la generación de la GPU; una A100 continúa siendo SM80.

#### Prefill, batching y concurrencia

##### `QWEN_MAX_NUM_BATCHED_TOKENS`

Es el máximo de tokens que el scheduler procesa juntos en un paso de prefill. Una analogía útil: el contexto completo es un libro y este parámetro es el número de páginas que se llevan a la mesa cada vez.

- Un bloque mayor puede acelerar prompts largos al reducir el número de tandas.
- También eleva el pico de activaciones y puede perjudicar la latencia de otras peticiones.
- No amplía el contexto total.

`8192` quedó validado en SP2. En SP3 proponemos probar `16384`; `32768` solo después de medir memoria, tiempo hasta el primer token y estabilidad.

##### `QWEN_MAX_NUM_SEQS`

Limita cuántas secuencias mantiene vLLM activas simultáneamente. Aunque el log calculó capacidad teórica para 2.47 secuencias máximas, el proxy tiene un lock en `predict` y otro en el worker. Por ello `2` no aumenta el throughput actual: las llamadas siguen entrando de una en una.

Para aprovechar `2` habría que rediseñar IPC, cancelación y concurrencia. Hasta entonces `1` alinea el scheduler con el comportamiento real y prioriza contexto.

##### `QWEN_ENFORCE_EAGER`

Con `true`, vLLM ejecuta las operaciones inmediatamente y evita CUDA Graphs. Es más conservador, arranca con menos compilación y facilitó estabilizar Ampere/SP2.

Con `false`, vLLM puede capturar y reutilizar grafos CUDA:

- posible mejora de latencia y tokens/s;
- warmup más largo;
- consumo adicional de VRAM;
- posibilidad de bloqueos o incompatibilidades de captura.

Es el experimento de rendimiento más interesante para SP3, pero debe probarse solo, sin cambiar a la vez el prefill o la memoria.

#### Backends de ejecución

##### `QWEN_ATTENTION_BACKEND`

Selecciona la implementación de atención. `TRITON_ATTN` es la ruta que funcionó con A100 y KV BF16. Cambiarla puede alterar compatibilidad, memoria y kernels; no se recomienda escoger un backend solo porque tenga un nombre más rápido en otra GPU.

##### `VLLM_USE_FLASHINFER_SAMPLER`

Controla únicamente el sampler top-k/top-p de FlashInfer, no la atención. Con el default interno de vLLM, FlashInfer intentó compilar durante el warmup y falló porque no existía `ninja`. Una compilación completa también puede necesitar NVCC y headers CUDA.

Mantenga `0`. En SP3 solo tendría sentido probar `1` después de verificar `ninja`, `nvcc`, toolkit y compatibilidad del kernel. Para una única secuencia, la ganancia esperable es secundaria frente al riesgo.

#### Longitud de respuesta

##### `QWEN_MAX_OUTPUT_TOKENS`

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

#### Imágenes, red y seguridad

##### `QWEN_MAX_IMAGES_PER_PROMPT`

Limita imágenes por conversación. `1` reduce tokens visuales, activaciones y tiempo de prefill. El wrapper acepta hasta `4`, pero esos valores no han sido validados. Vídeo está fijado a cero en el código y no se habilita con esta variable.

##### `VLLM_ALLOWED_MEDIA_DOMAINS`

Es una lista separada por comas de dominios desde los que vLLM puede descargar imágenes remotas. En producción debe actuar como allowlist para evitar que el endpoint se use para consultar direcciones internas. También deben aplicarse controles de tamaño, tipo MIME y tiempo de descarga fuera del modelo.

#### Operación del contenedor

##### `QWEN_STARTUP_TIMEOUT_SECONDS`

Es el tiempo que el proxy espera a que el worker descargue/cargue pesos, cree la caché y caliente kernels. No es el timeout de una inferencia y no mejora rendimiento. `1800` segundos da margen al primer arranque; una caché persistente reduce cargas posteriores.

##### `HF_TOKEN` y `HF_HOME`

`HF_TOKEN` evita límites anónimos y permite acceder a repositorios autorizados. Debe almacenarse como secreto de solo lectura. `HF_HOME` decide dónde se guardan pesos y metadatos; conviene apuntarlo a almacenamiento persistente para no descargar el modelo en cada réplica.

### Qué no puede arreglar una variable

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

### Streaming: por qué no es un parámetro

vLLM puede producir tokens incrementalmente, pero este endpoint usa la interfaz síncrona `LLM.chat()`. Después el worker construye un JSON completo, lo manda por el socket IPC y `predict(args)` devuelve un único objeto a PBJ. Ninguno de esos tres pasos expone Server-Sent Events o WebSocket al cliente.

Cambiar el worker para enviar tokens parciales no bastaría: la función `@cml_model` seguiría esperando un resultado serializable. Las alternativas son:

1. Desplegar el modelo en **Cloudera AI Inference service**, que expone API OpenAI y `stream=true` cuando el checkpoint esté validado.
2. Crear una **Workbench Application** con FastAPI y SSE/WebSocket, usando un motor asíncrono.
3. Implementar un patrón de trabajo asíncrono más polling; no sería streaming real.

AI Inference SP3 usa vLLM 0.20 y declara `Qwen3_5ForConditionalGeneration`, la arquitectura interna resuelta por el checkpoint. Eso justifica una prueba, pero no equivale a certificación explícita del artefacto Qwen3.8 FP8.

### Plan de mejora para SP3

La regla principal es cambiar **una sola variable cada vez**. Si se modifican eager, prefill y memoria simultáneamente, un fallo no permite saber qué cambio lo causó.

#### Fase 0: reproducir la línea base

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

#### Fase 1: bloques de prefill mayores

Cambiar únicamente:

```text
QWEN_MAX_NUM_BATCHED_TOKENS=16384
```

Objetivo: reducir tiempo hasta el primer token con prompts RAG largos. Revertir a `8192` si sube demasiado el pico de memoria, empeora la latencia corta o el motor no inicia.

#### Fase 2: CUDA Graphs

Volver a una base conocida y cambiar únicamente:

```text
QWEN_ENFORCE_EAGER=false
```

Objetivo: mejorar latencia de decode y tokens/s. Revertir a `true` ante errores de graph capture, arranque excesivo, OOM o inestabilidad. Si funciona, repetir la prueba combinándolo después con prefill `16384`.

#### Fase 3: imágenes

Solo si existe un caso de uso real de comparación visual:

```text
QWEN_MAX_IMAGES_PER_PROMPT=2
```

Medir memoria, tokens visuales y tiempo hasta el primer token con la resolución objetivo. No saltar directamente a cuatro imágenes.

#### Fase 4: memoria GPU

Probar `0.92` únicamente si una fase anterior reduce la caché por debajo del contexto requerido:

```text
QWEN_GPU_MEMORY_UTILIZATION=0.92
```

No aporta velocidad por sí mismo. No usar `0.95` sin observar la GPU completa y los procesos residentes.

#### Fase 5: evaluar AI Inference

Registrar el checkpoint y comprobar si el servidor gestionado vLLM 0.20 lo acepta. Si funciona, evaluar:

- API `/v1/chat/completions`;
- streaming `stream=true`;
- integración directa con RAG Studio;
- autoscaling y autenticación Knox;
- tool calling y parser de razonamiento;
- imágenes;
- equivalencia de calidad y contexto con Workbench.

Si el servidor no reconoce el checkpoint o la cuantización, conservar Workbench 0.29 o crear una imagen/servidor OpenAI propio; no degradar silenciosamente librerías del entorno funcional.

#### Fase 6: concurrencia, que requiere código

Solo después de estabilizar el motor se puede rediseñar el proxy para varias peticiones. Entonces tendría sentido probar `QWEN_MAX_NUM_SEQS=2`. Con el código actual, cambiar esa variable no genera paralelismo útil.

### Cómo medir las mejoras

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

### Contrato de `predict`

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

#### Parámetros de petición

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

### Multimodalidad

El chat template acepta `content` como texto o lista multimodal. El motor limita cada prompt a una imagen y cero vídeos. Si se permiten imágenes por URL, configure `VLLM_ALLOWED_MEDIA_DOMAINS` con una allowlist explícita; no deje acceso abierto a hosts internos. Valide además el tamaño de las imágenes y los timeouts antes de exponer el endpoint.

### Cómo reconocer un arranque correcto

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

### Historial de fallos y solución

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

### Operación y seguridad

- Use un `HF_TOKEN` de solo lectura y guárdelo como secreto, no en Git.
- Para producción, fije `QWEN_MODEL_ID` a un snapshot inmutable o replique el modelo internamente.
- Dimensione el disco para el snapshot, cachés y artefactos temporales; deje al menos 40 GB libres como punto de partida.
- No arranque muchas réplicas simultáneamente contra una caché vacía. Precargue el snapshot.
- Mantenga una réplica durante la validación. Cada réplica necesita su propia A100 completa.
- `predict` no ofrece streaming. Limite `max_tokens` para respetar el timeout de Workbench.
- El socket IPC es local al contenedor, usa framing de 8 bytes y limita cada mensaje a 128 MiB.
- El endpoint serializa la inferencia mediante un lock. Para más throughput, escale réplicas con una GPU por réplica en lugar de aumentar concurrencia sin pruebas.

### Workbench frente a RAG Studio

Este código implementa un endpoint clásico de Workbench Models con `predict`. Aunque la respuesta se parece a OpenAI, no expone directamente `/v1/chat/completions`, `/v1/models`, streaming SSE ni autenticación OpenAI. No debe registrarse en RAG Studio como un endpoint OpenAI sin un adaptador/gateway que traduzca la API de Workbench.

En SP3, AI Inference service puede aportar una API gestionada y compatibilidad OpenAI para los modelos incluidos en su matriz. Este perfil Qwen usa vLLM 0.29 dentro de Workbench porque la versión gestionada debe soportar explícitamente la arquitectura concreta. Verifique la matriz de la versión instalada antes de migrarlo.

### Checklist final

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

### Fuentes oficiales de referencia

- [Ficha de Qwen3.8-27B-FP8 en Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Repositorio oficial de Qwen3.8](https://github.com/QwenLM/Qwen3.8)
- [Novedades de Cloudera AI on premises 1.5.5 SP3](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-whats-new-1-5-5-sp3.html)
- [Inventario de servidores y arquitecturas de AI Inference SP3](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-model-server-inventory-sp3.html)
- [Argumentos vLLM soportados por AI Inference SP3](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-caii-supported-vllm-command-line-arguments-sp3.html)
- [Interacción y streaming con Model Endpoints de AI Inference](https://docs.cloudera.com/machine-learning/1.5.5/ai-inference/ml-ai-inference.pdf)

---

## Detailed Qwen3.8-27B-FP8 guide for Cloudera AI Workbench SP2/SP3


This guide documents the tested `Qwen/Qwen3.8-27B-FP8` profile as a **Workbench Model** on one NVIDIA A100 PCIe 80 GB. The endpoint uses Cloudera's `predict(args)` function and returns a JSON response shaped like Chat Completions.

### Validated state

The configuration was validated on September 19, 2026, in an on-premises SP2 environment:

| Component | Observed value |
|---|---|
| GPU | NVIDIA A100 80GB PCIe, 79.3 GiB visible, SM80 |
| Runtime | Nvidia GPU Edition, Python 3.10 |
| vLLM | `0.29.0+cu129` |
| PyTorch | `2.13.0+cu129` |
| PyTorch CUDA build | `12.9` |
| Transformers | `5.15.0` |
| Worker NumPy | `2.2.6` |
| Weights | Block-wise FP8, executed as weight-only FP8 through Marlin/W8A16 |
| Observed weight memory | 28.9 GiB |
| KV cache | BF16, 40.31 GiB available |
| Observed KV capacity | 647,288 tokens |
| Configured context | 262,144 tokens |
| vLLM calculated concurrency | 2.47 requests at 262,144 tokens; the wrapper limits actual concurrency to one |
| Attention backend | `TRITON_ATTN` |
| Sampler | Native vLLM/PyTorch; FlashInfer sampler disabled |
| SP2 output ceiling | `QWEN_MAX_OUTPUT_TOKENS=8192` |
| Default output when `max_tokens` is omitted | 128 tokens |
| Result | Model loaded and endpoint operational |

These figures are diagnostic references, not guaranteed reservations. They can vary with the driver, Runtime, checkpoint revision, resident processes, and actual GPU allocation. This profile requires a full A100 80 GB and rejects less than 70 GiB visible so that MIG, vGPU, or A100 40 GB configurations fail before loading the weights.

### Minimal working deployment

Use the following values in **Deploy model from code**:

| Field | Value |
|---|---|
| Root Directory / Model Root Directory | empty or `.` |
| Build Script Path | `cdsw-build.sh` |
| Build variable | `MODEL_FAMILY=qwen3_8` |
| Build variable | `GPU_TYPE=a100` |
| File | `qwen3_8/model_a100.py` |
| Function | `predict` |
| Example Input | contents of `examples/qwen3_8_input.json` |
| Runtime | Nvidia GPU Edition, Python 3.10 or 3.11, Ubuntu 24.04 |
| GPU | 1 × full A100 80 GB, no MIG |
| CPU | 8 vCPU minimum; 16 recommended |
| RAM | 64 GiB minimum; 128 GiB recommended |
| Initial replicas | 1 |

The model starts without runtime variables because safe defaults are embedded in the code. The validated SP2 deployment added `QWEN_MAX_OUTPUT_TOKENS=8192` to permit long answers. This is a ceiling, not a request to always generate 8192 tokens: each request is still governed by `max_tokens`, whose default is 128.

### File responsibilities

| File | Responsibility |
|---|---|
| `cdsw-build.sh` | Selects the installer through `MODEL_FAMILY` and `GPU_TYPE` |
| `qwen3_8/install_a100.sh` | Creates `.venv`, installs the cu129 vLLM wheel, and validates versions/imports |
| `qwen3_8/requirements.txt` | Pins Transformers for reproducible builds |
| `qwen3_8/model_a100.py` | PBJ entry point; imports Cloudera and acts as a lightweight proxy |
| `qwen3_8/worker_a100.py` | Isolated process that imports vLLM, loads the model, and generates responses |
| `examples/qwen3_8_input.json` | Input ready to paste into the deployment form |
| `examples/qwen3_8_output.json` | Representative response shape |

### Why an isolated virtual environment is required

SP2 already contains libraries used by Cloudera, including extensions compiled for NumPy 1.x and constraints such as `protobuf==4.25.3`. vLLM 0.29 installs a newer stack, including NumPy 2. Mixing both stacks in one interpreter caused ABI errors such as:

```text
numpy.dtype size changed, may indicate binary incompatibility
```

The installer creates `qwen3_8/.venv`. It does not activate that environment in PBJ and does not inject its `site-packages` into the base process:

```text
Cloudera/PBJ base Python
        |
        | predict(args), JSON over a local socket
        v
model_a100.py  ----------------->  worker_a100.py with .venv/bin/python
  cml.models_v1                    torch + numpy 2 + vLLM + transformers
```

The proxy starts one persistent worker. The worker loads the weights once, reports `ready` after initializing cache and kernels, and then serves requests. Startup failures are returned with their complete traceback so that the root cause appears in the Cloudera log.

Do not set `PYTHONPATH`, `VIRTUAL_ENV`, `PIP_USER`, or another interpreter manually. `install_a100.sh` neutralizes `PIP_USER=true` only during the build because pip forbids user installs inside a virtual environment.

### Why the cu129 wheel is used

The standard vLLM 0.29.0 PyPI resolution can select a CUDA variant newer than the Runtime driver. The installer explicitly downloads:

```text
vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_<architecture>.whl
```

It then runs `pip check` inside the virtual environment and validates vLLM, Transformers, the CUDA build used by PyTorch, and the import of `Qwen3_5ForConditionalGeneration`. Qwen3.8 is the published model name; `Qwen3_5ForConditionalGeneration` is the internal architecture resolved by Transformers/vLLM and does not mean that a different model was loaded.

### Precision, memory, and context

Weights and KV cache are separate decisions:

- **FP8 weights:** valid on A100 through the Marlin weight-only FP8/W8A16 kernel. A100 has no native FP8 compute, so it may be slower than Hopper, but the model loads correctly.
- **FP8 KV cache:** incompatible with Triton on A100 SM80; native support requires SM89 or newer.
- **BF16 KV cache:** validated configuration. The code uses it by default and automatically converts a legacy FP8 request to BF16 on a GPU older than SM89.
- **Context:** 262,144 tokens were validated with 40.31 GiB of cache and a calculated capacity of 647,288 tokens.
- **Prefill:** 8192 is the processing block, not the total chat length.
- **Concurrency:** the endpoint serializes calls and uses one active sequence to prioritize context and stability.

If another Runtime exposes less free VRAM, reduce `QWEN_MAX_MODEL_LEN` in this order: `131072`, `65536`, `32768`. Do not increase `QWEN_GPU_MEMORY_UTILIZATION` beyond `0.95`.

### Triton, FlashInfer, and ninja

Three independent execution paths are involved:

1. Marlin executes FP8-weight linear layers on Ampere.
2. `TRITON_ATTN` executes attention with BF16 KV cache.
3. The native vLLM/PyTorch sampler performs top-k/top-p sampling.

FlashInfer tried to JIT-compile its sampler during warmup, but the Runtime did not contain `ninja`:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

Installing only `ninja` would not guarantee success because JIT compilation may also require a visible NVCC toolkit. The worker therefore sets `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM. This only replaces top-k/top-p sampling; it does not disable Triton, change weights to BF16, or reduce context length.

### Build variables

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `MODEL_FAMILY` | Yes | `nemotron` in the global dispatcher | Must be `qwen3_8` |
| `GPU_TYPE` | Yes | `l40s` in the global dispatcher | Must be `a100` |
| `PYTHON_BIN` | No | `python3` | Interpreter used to create the virtual environment |
| `VLLM_VERSION` | No | `0.29.0` | Official wheel version |
| `VLLM_CUDA_VARIANT` | No | `129` | CUDA wheel suffix |
| `VLLM_WHEEL_URL` | No | Official GitHub release | Allows an internal mirror |
| `PYTORCH_INDEX_URL` | No | PyTorch cu129 index | Allows an internal mirror |

Do not copy build variables into runtime configuration unless local Cloudera policy shares the same variable set between phases. Changing a build variable requires a new build.

### Runtime variables: complete map

Three values must not be confused:

- **Code default:** used when the variable is absent.
- **Validated SP2 value:** used by the deployment that worked.
- **Accepted range:** wrapper validation limits, not a promise that every value is appropriate for one A100.

| Variable | Code default | Validated SP2 | Accepted range | Recommendation |
|---|---:|---:|---:|---|
| `QWEN_MODEL_ID` | `Qwen/Qwen3.8-27B-FP8` | same | ID or path | Keep it or use an immutable snapshot |
| `QWEN_SERVED_MODEL_NAME` | `qwen3.8-27b-fp8` | same | text | Output metadata only |
| `QWEN_TENSOR_PARALLEL_SIZE` | `1` | `1` | `1-8` | Keep `1` with one A100 |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.90` | `0.90` | `0.50-0.95` | Keep `0.90`; test `0.92` only if more KV is needed |
| `QWEN_MAX_MODEL_LEN` | `262144` | `262144` | `2048-262144` | Keep native maximum; reduce on memory pressure |
| `QWEN_MAX_NUM_SEQS` | `1` | `1` | `1-64` | Keep `1`; the proxy serializes requests |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | `8192` | `8192` | `2048-131072` | SP3: test `16384` for long prefill |
| `QWEN_KV_CACHE_DTYPE` | `bfloat16` | `bfloat16` | vLLM-supported string | Must remain BF16 on A100 |
| `QWEN_ENFORCE_EAGER` | `true` | `true` | boolean | SP3: test `false` in isolation |
| `QWEN_ATTENTION_BACKEND` | `TRITON_ATTN` | same | vLLM backend | Keep Triton in this profile |
| `QWEN_CPU_OFFLOAD_GB` | `0` | `0` | `0-128` | Keep `0`; weights fit in GPU |
| `QWEN_MAX_OUTPUT_TOKENS` | `512` | **`8192`** | `1-8192` | Keep `8192` as a ceiling if timeouts permit |
| `QWEN_MAX_IMAGES_PER_PROMPT` | `1` | `1` | `0-4` | Keep `1`; test `2` only with measurements |
| `QWEN_STARTUP_TIMEOUT_SECONDS` | `1800` | `1800` | `60-7200` | Startup only; does not affect inference |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | `0` | `0/1` | Keep `0` without a validated `ninja`/NVCC toolchain |
| `VLLM_ALLOWED_MEDIA_DOMAINS` | empty | empty | CSV list | Use an explicit allowlist in production |
| `HF_TOKEN` | empty | environment-specific | secret | Read-only token; never commit it |
| `HF_HOME` | HF default | environment-specific | path | Persistent storage with enough free space |

Legacy names such as `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_KV_CACHE_DTYPE`, `VLLM_ENFORCE_EAGER`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_TENSOR_PARALLEL_SIZE`, and `VLLM_CPU_OFFLOAD_GB` remain accepted as aliases. The worker captures and removes them before importing vLLM because they are not official vLLM 0.29 environment variables. Use `QWEN_*` names for new deployments.

### Parameter-by-parameter explanation

#### Model identity

`QWEN_MODEL_ID` selects configuration, tokenizer, and weights. Pin an immutable snapshot or internal mirror for production reproducibility. `QWEN_SERVED_MODEL_NAME` only changes the logical model name returned in JSON.

#### GPU distribution

`QWEN_TENSOR_PARALLEL_SIZE` is the number of GPUs cooperating on one model instance. With one allocated A100, the only meaningful value is `1`; setting `2` does not double performance and will fail because a second device is unavailable.

`QWEN_CPU_OFFLOAD_GB` moves part of the weights to RAM and copies them over PCIe when required. It is useful when a model does not fit, but here the 28.9 GiB weights fit comfortably. Keep it at zero to avoid PCIe latency.

#### VRAM and context

`QWEN_GPU_MEMORY_UTILIZATION` is vLLM's VRAM budget for weights, activations, kernels, and KV cache. `0.90` is validated. `0.92` is a controlled experiment; `0.95` leaves little safety margin. Raising it does not make short answers faster or extend the native 262K limit.

`QWEN_MAX_MODEL_LEN` covers system instructions, history, RAG documents, visual tokens, and generated output together. Although 262K is available, routine RAG chats should use smaller operational budgets for lower prefill latency. Reduce it stepwise if another Runtime leaves less memory.

`QWEN_KV_CACHE_DTYPE=bfloat16` is required because A100 is SM80. SP3 cannot turn the same physical GPU into SM89; software upgrades do not enable Triton FP8 KV support that the hardware lacks.

#### Prefill and concurrency

`QWEN_MAX_NUM_BATCHED_TOKENS` is the prefill block size. Think of the entire context as a book and this parameter as how many pages are placed on the desk at once. Larger blocks can accelerate long prompts but increase peak activations. Test 16384 on SP3 before considering 32768.

`QWEN_MAX_NUM_SEQS` controls active sequences inside vLLM. The current proxy and worker both use locks, so raising it does not create useful parallelism. Concurrency requires an IPC and cancellation redesign first.

`QWEN_ENFORCE_EAGER=true` avoids CUDA Graph capture and was the stable SP2 choice. Setting it to `false` may improve decode latency and tokens/s, but adds warmup, graph memory, and capture risk. It is the most interesting SP3 performance experiment and must be tested alone.

#### Execution backends

`QWEN_ATTENTION_BACKEND=TRITON_ATTN` is the validated attention route. A backend that is faster on Hopper is not automatically suitable for A100.

`VLLM_USE_FLASHINFER_SAMPLER=0` disables only FlashInfer top-k/top-p sampling. Keep it disabled until `ninja`, NVCC, CUDA headers, and kernel compatibility have all been verified. Expected gains are secondary with one active sequence.

#### Output length

`QWEN_MAX_OUTPUT_TOKENS` is a safety ceiling, not an automatic generation length. The working SP2 deployment used 8192:

- no `max_tokens` in the request: at most 128 tokens;
- `"max_tokens": 1024`: allowed;
- `"max_tokens": 9000`: rejected by the wrapper.

An 8192-token ceiling is compatible with validation and context, but an actual 8192-token generation can exceed a client timeout because Workbench `predict` does not stream.

#### Images and network security

`QWEN_MAX_IMAGES_PER_PROMPT=1` controls visual input count. The wrapper accepts up to four, but those values have not been validated. More images consume visual tokens, activation memory, and prefill time. Video is hard-coded to zero and requires a code change.

`VLLM_ALLOWED_MEDIA_DOMAINS` should contain an explicit comma-separated allowlist in production so that users cannot make the endpoint fetch arbitrary internal addresses. File size, MIME type, and download timeout should also be enforced outside the model.

#### Container operation

`QWEN_STARTUP_TIMEOUT_SECONDS` is how long the proxy waits for download, weight loading, cache creation, and warmup. It is not an inference timeout. `HF_TOKEN` should be a read-only secret, and `HF_HOME` should use persistent storage to avoid downloading the snapshot for every replica.

### Limits that variables cannot fix

| Limit | Imposing layer | Consequence |
|---|---|---|
| FP8 KV unavailable | A100 SM80 + Triton | Keep BF16 KV even on SP3 |
| No native FP8 compute | Ampere hardware | Marlin/W8A16 weights work but are slower than Hopper |
| One GPU | Resource profile | Tensor parallel must remain 1 |
| FlashInfer JIT failure | No validated `ninja`/NVCC toolchain | Native sampler |
| CUDA 12.9 | Driver and available wheel | cu129 wheel, not the CUDA 13 PyPI wheel |
| NumPy/protobuf conflicts | Cloudera base libraries | Separate virtualenv worker process |
| No streaming | Workbench `predict`/PBJ contract | One final JSON response |
| No OpenAI API | Workbench endpoint type | RAG Studio needs AI Inference or a gateway |
| No video | Explicit worker limit | Code, decoder, and testing required |
| Real concurrency is one | Proxy and worker locks | Raising `MAX_NUM_SEQS` alone has no effect |

### Why streaming is not a parameter

vLLM can produce incremental tokens, but this endpoint calls synchronous `LLM.chat()`. The worker then builds one complete JSON object, sends it over IPC, and `predict(args)` returns one serializable object to PBJ. None of those layers exposes SSE or WebSocket events to the client.

Sending partial IPC messages would not solve the outer transport. The practical alternatives are:

1. Use **Cloudera AI Inference service**, which exposes an OpenAI API and `stream=true`, after validating this checkpoint.
2. Build a **Workbench Application** with FastAPI plus SSE/WebSocket and an asynchronous engine.
3. Implement job kickoff plus polling, which is asynchronous but not true token streaming.

SP3 AI Inference uses vLLM 0.20 and lists `Qwen3_5ForConditionalGeneration`, the internal checkpoint architecture. This supports a compatibility trial but is not explicit certification of the Qwen3.8 FP8 artifact.

### SP3 improvement plan

Change **one variable at a time**. Simultaneous changes to eager mode, prefill, and memory make failures impossible to attribute.

#### Phase 0: reproduce the baseline

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

The SP3 baseline is not accepted until text, long context, and image tests pass.

#### Phase 1: larger prefill blocks

Change only:

```text
QWEN_MAX_NUM_BATCHED_TOKENS=16384
```

Goal: reduce time to first token for long RAG prompts. Revert to 8192 if peak memory rises too far, short-prompt latency regresses, or startup fails.

#### Phase 2: CUDA Graphs

Return to a known baseline and change only:

```text
QWEN_ENFORCE_EAGER=false
```

Goal: improve decode latency and tokens/s. Revert on graph capture errors, excessive startup time, OOM, or instability. If successful, test it later together with prefill 16384.

#### Phase 3: images

Only when a real comparison use case exists, test `QWEN_MAX_IMAGES_PER_PROMPT=2`. Measure VRAM, visual tokens, and time to first token at the target resolution. Do not jump directly to four images.

#### Phase 4: GPU memory

Test `QWEN_GPU_MEMORY_UTILIZATION=0.92` only if an earlier phase reduces cache below the required context. It does not improve speed by itself. Do not use 0.95 without observing the full GPU and resident processes.

#### Phase 5: evaluate AI Inference

Register the checkpoint and determine whether managed vLLM 0.20 accepts it. If it does, evaluate OpenAI chat completions, `stream=true`, RAG Studio integration, autoscaling, Knox authentication, tool calling, reasoning parsing, images, quality, and context equivalence.

If the managed server rejects the checkpoint or quantization, retain Workbench 0.29 or build a dedicated OpenAI-compatible server. Do not silently downgrade the working environment's libraries.

#### Phase 6: concurrency requires code

Only after stabilizing the engine should the proxy be redesigned for multiple in-flight requests. At that point, `QWEN_MAX_NUM_SEQS=2` becomes meaningful. It does not provide useful parallelism with the current locks.

### Measuring improvements

Use the same test set for every phase:

| Metric | What it reveals |
|---|---|
| Startup time | Download, compilation, and warmup cost |
| Time to first token | Prefill performance; estimate through controlled tests with `predict` |
| Output tokens per second | Decode performance |
| Total latency | Actual consumer experience |
| Peak VRAM | OOM safety margin |
| Reported KV capacity | Whether the target context still fits |
| 262K success | Preservation of maximum capacity |
| Short and long text | Avoids optimizing only one extreme |
| One real image | Validates the vision processor |
| Three consecutive restarts | Detects nondeterministic warmup failures |

Accept a change only if it improves performance without changing functional output, reducing the required context, or failing any of three startup attempts. Record Runtime, driver, visible GPU, wheel, Torch, CUDA, and commit for reproducibility.

### `predict` contract

The Cloudera function must be exactly `predict`. The input is a JSON object and may use chat format:

```json
{
  "messages": [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is Cloudera AI?"}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.8,
  "enable_thinking": false,
  "reasoning_effort": "low"
}
```

The short form is also accepted:

```json
{"prompt": "Explain RAG in two sentences", "max_tokens": 128}
```

#### Request parameters

| Field | Default | Accepted values |
|---|---|---|
| `messages` | none | Non-empty array; roles `system`, `user`, `assistant`, `tool` |
| `prompt` | none | Text alternative to `messages` |
| `max_tokens` | `128` | `1` through `QWEN_MAX_OUTPUT_TOKENS` |
| `temperature` | `0.7` without thinking; `1.0` with thinking | `0.0-2.0` |
| `top_p` | `0.80` without thinking; `0.95` with thinking | `0.01-1.0` |
| `top_k` | `20` | `-1-1000` |
| `min_p` | `0.0` | `0.0-1.0` |
| `presence_penalty` | `1.5` without thinking; `0.0` with thinking | `-2.0-2.0` |
| `repetition_penalty` | `1.0` | `0.01-2.0` |
| `seed` | `0` | `0-2147483647` |
| `stop` | none | Value accepted by `SamplingParams` |
| `enable_thinking` | `false` | Boolean |
| `preserve_thinking` | `true` | Boolean passed to the chat template |
| `reasoning_effort` | `low` | `low`, `medium`, `xhigh` |

The response resembles OpenAI JSON but is returned through the Workbench Models API. If the model emits `<think>...</think>`, the wrapper puts that content in `reasoning_content` and the visible answer in `content`.

### Multimodality

Message `content` can be text or a multimodal list. The engine permits one image and zero videos per prompt. For remote images, set `VLLM_ALLOWED_MEDIA_DOMAINS` to an explicit allowlist and validate image size and timeouts before exposing the endpoint. This is an image-understanding model deployment; it returns text and does not generate image or video files.

### Recognizing a healthy startup

Relevant lines should appear in approximately this order:

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

The warning that A100 lacks native FP8 support is expected and confirms Marlin weight-only compression. Deprecation warnings for `cuda.nvrtc`, `cuda.cudart`, or `use_fast` did not block the validated startup.

### Failure history and current solution

| Symptom | Cause | Current solution |
|---|---|---|
| Build fails quickly or uses CUDA 13 | PyPI wheel does not match the driver | Explicit official `0.29.0+cu129` wheel |
| Pandas, matplotlib, RAZ, or protobuf conflicts | vLLM installed in global Python | Isolated `.venv` |
| `Can not perform a '--user' install` | Runtime exports `PIP_USER=true` | Installer sets `PIP_USER=false` during build |
| `__file__ is not defined` | PBJ runs the file as a cell | cwd-compatible discovery |
| Architecture inspection fails | Mixed interpreters | Inspection and engine entirely in worker |
| `numpy.dtype size changed` | NumPy 2 injected into NumPy 1 process | Separate proxy/worker processes |
| `FP8 KV cache is not supported ... A100` | SM80 cannot use Triton FP8 KV | BF16 KV plus automatic legacy correction |
| `No such file or directory: 'ninja'` | FlashInfer sampler attempts JIT | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| NCCL warning during shutdown | Consequence of an earlier EngineCore failure | Find the first preceding traceback |
| Anonymous Hugging Face warning | No `HF_TOKEN` | Read-only token and persistent cache |

### Operations and security

- Keep `HF_TOKEN` read-only and out of Git.
- Pin an immutable snapshot or internal model copy for production.
- Provide at least 40 GB of free storage for weights, metadata, and temporary files.
- Preload an empty shared cache before starting many replicas.
- Use one replica during validation; every replica requires its own full A100.
- Workbench `predict` does not stream, so request length must respect client timeouts.
- IPC is local to the container, uses an 8-byte frame header, and caps messages at 128 MiB.
- Inference is serialized. Scale replicas with one GPU each rather than raising concurrency without tests.

### Workbench versus RAG Studio

This is a classic Workbench `predict` endpoint. Although its response resembles OpenAI, it does not directly expose `/v1/chat/completions`, `/v1/models`, SSE streaming, or OpenAI authentication. RAG Studio requires AI Inference or an adapter/gateway.

SP3 AI Inference can provide a managed OpenAI-compatible endpoint for models in its compatibility matrix. Its vLLM 0.20 inventory lists the internal Qwen architecture, but this exact Qwen3.8 FP8 checkpoint still requires a deployment trial before migration.

### Final checklist

- [ ] Nvidia GPU Edition with Python 3.10/3.11.
- [ ] Full A100 80 GB, no MIG.
- [ ] Root Directory empty or `.`.
- [ ] Build Script Path `cdsw-build.sh`.
- [ ] Build variables `MODEL_FAMILY=qwen3_8` and `GPU_TYPE=a100`.
- [ ] File `qwen3_8/model_a100.py`.
- [ ] Function `predict`.
- [ ] Valid example input.
- [ ] Dependency sources reachable during build.
- [ ] Model or snapshot reachable during first startup.
- [ ] `QWEN_MAX_OUTPUT_TOKENS=8192` when the validated long-output ceiling is desired.
- [ ] `VLLM_USE_FLASHINFER_SAMPLER` absent or `0`.
- [ ] `QWEN_KV_CACHE_DTYPE` absent or `bfloat16`.
- [ ] No manual `PYTHONPATH` or `VIRTUAL_ENV` changes.
- [ ] Logs confirm Marlin, BF16 KV, and sufficient cache capacity.

### Official references

- [Qwen3.8-27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Official Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8)
- [What's new in Cloudera AI on premises 1.5.5 SP3](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-whats-new-1-5-5-sp3.html)
- [SP3 AI Inference server and architecture inventory](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-model-server-inventory-sp3.html)
- [vLLM arguments supported by SP3 AI Inference](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-caii-supported-vllm-command-line-arguments-sp3.html)
- [AI Inference Model Endpoint interaction and streaming](https://docs.cloudera.com/machine-learning/1.5.5/ai-inference/ml-ai-inference.pdf)
- [vLLM 0.29.0 y artefactos CUDA 12.9](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [vLLM: selección del attention backend](https://docs.vllm.ai/en/latest/design/attention_backends/)
- [Cloudera: hosting an LLM as a Workbench model](https://docs.cloudera.com/machine-learning/1.5.5/models/topics/ml-creating-a-cloudera-ai-model.html)
- [Cloudera SP3: inventario de servidores y modelos](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-model-server-inventory-sp3.html)
- [Cloudera SP3: novedades](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-whats-new-1-5-5-sp3.html)
- [Cloudera: configuración de RAG Studio](https://docs.cloudera.com/machine-learning/cloud/setup-ai-studios/topics/ml-configuring-rag-studio.html)
