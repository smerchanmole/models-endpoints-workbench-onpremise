# Nemotron 3 Nano y embeddings en Cloudera AI Workbench

Proyecto para desplegar dos Model Endpoints desde Hugging Face en **Cloudera AI on premises 1.5.5 SP2 o SP3**:

- LLM: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`.
- Embeddings: `BAAI/bge-m3` (568 M parámetros, 1024 dimensiones, más de 100 idiomas y contexto de 8192 tokens).

Los endpoints usan el contrato de Cloudera `api_wrapper(args)` y el decorador `@cml_model`. Cada combinación GPU/modelo tiene su instalador y su fichero Python.

## Elección de precisión y memoria

| Endpoint | GPU | Checkpoint / precisión | Configuración inicial | Motivo |
|---|---|---|---|---|
| Nemotron | L40S 48 GB | FP8, descarga aproximada 32.7 GB | 262K contexto, 1 secuencia, prefill por bloques de 4096 | BF16 no cabe con runtime y cachés |
| Nemotron | H100 80 GB | BF16, pesos aproximados 60 GB | 262K contexto, 1 secuencia, prefill por bloques de 8192 | Máxima calidad dentro de una H100 completa |
| BGE-M3 | L40S | FP16 | 8K contexto, lote 8, máximo 32 | Embedding denso multilingüe para RAG |
| BGE-M3 | H100 | BF16 | 8K contexto, lote 16, máximo 64 | Mayor lote y rango numérico |

Las cifras presuponen una GPU **completa**, sin MIG ni una vGPU con menos VRAM. Nemotron declara un máximo de 262144 tokens. Su arquitectura solo tiene 6 capas de atención y 2 cabezas KV, por lo que su caché KV crece mucho menos que la de un Transformer denso de 30B. El prefill por bloques evita procesar los 262K tokens de una vez. Aun así, 262K es un perfil de capacidad, no de baja latencia: debe validarse con el Runtime y driver reales.

En la H100 de 80 GB, BF16 deja poco margen; si el runtime concreto consume más memoria, use el checkpoint FP8 también en H100 cambiando `LLM_MODEL_ID` y `LLM_DTYPE=auto`, `LLM_KV_CACHE_DTYPE=fp8`. Si una L40S no inicia con 262K, reduzca primero a 131072 y después a 65536.

BGE-M3 ocupa aproximadamente 2.27 GB en FP32 y continúa siendo pequeño frente a estas GPU. Se elige frente a E5-small porque amplía el contexto de embedding de 512 a 8192 tokens y ofrece mejor encaje para documentos largos. No requiere prefijos diferentes para consultas y documentos; `input_type` se conserva en la API para mantener explícita la intención. No mezcle vectores creados con modelos, dimensiones o políticas de normalización distintas en el mismo índice: al cambiar desde E5 hay que reconstruir la colección vectorial.

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
    ├── install_l40s.sh
    ├── install_h100.sh
    ├── model_l40s.py
    └── model_h100.py
└── rag_studio/
    ├── nemotron_l40s.args
    ├── nemotron_h100.args
    └── embedding_bge_m3.args
```

Cloudera ejecuta `cdsw-build.sh` en un build limpio; las librerías instaladas manualmente en una sesión no se transfieren al modelo. El dispatcher selecciona uno de los cuatro instaladores mediante `MODEL_FAMILY` y `GPU_TYPE`.

## Requisitos previos del administrador

1. Nodo x86_64 con L40S 48 GB o H100 80 GB, drivers NVIDIA visibles desde Kubernetes y perfil de recursos de 1 GPU completa.
2. ML Runtime **Nvidia GPU Edition**, Python 3.10 o posterior.
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
- Función: `api_wrapper`.
- Recursos recomendados: 1 L40S 48 GB, 8 vCPU como mínimo, 64 GB RAM, 1 réplica inicial.

### Nemotron en H100

- Build variables: `MODEL_FAMILY=nemotron`, `GPU_TYPE=h100`.
- Fichero: `nemotron/model_h100.py`.
- Función: `api_wrapper`.
- Recursos recomendados: 1 H100 80 GB, 8-16 vCPU, 96-128 GB RAM, 1 réplica inicial.

### Embeddings en L40S/H100

- Build variables: `MODEL_FAMILY=embedding`, `GPU_TYPE=l40s` o `h100`.
- Fichero: `embedding/model_l40s.py` o `embedding/model_h100.py`.
- Función: `api_wrapper`.
- Recursos recomendados: 1 GPU, 2-4 vCPU, 8-16 GB RAM. El modelo también puede funcionar en CPU cambiando el código/dtype, pero estos artefactos están preparados para GPU.

El primer arranque descarga los pesos. Para evitar arranques lentos y descargas por réplica, monte una caché persistente compartida y configure `HF_HOME` con esa ruta. No use una caché escribible compartida para arrancar muchas réplicas simultáneamente por primera vez: precargue primero el snapshot completo.

## Variables del Model Deployment

### Comunes de build

| Variable | Requerida | Valor | Descripción |
|---|---:|---|---|
| `MODEL_FAMILY` | Sí | `nemotron` o `embedding` | Selecciona dependencias en `cdsw-build.sh` |
| `GPU_TYPE` | Sí | `l40s` o `h100` | Selecciona el instalador |
| `VLLM_VERSION` | No | `0.12.0` | Versión recomendada por la receta oficial del modelo; evite 0.15.0/0.15.1 por una regresión FP8 conocida |
| `SENTENCE_TRANSFORMERS_VERSION` | No | `5.1.2` | Versión fijada para builds reproducibles |

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

| Variable | L40S | H100 | Notas |
|---|---|---|---|
| `EMBEDDING_MODEL_ID` | `BAAI/bge-m3` | igual | Modelo MIT o ruta local inmutable |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` | Estos artefactos esperan GPU |
| `EMBEDDING_MAX_SEQ_LENGTH` | `8192` | `8192` | Máximo real del modelo; los chunks normales de RAG deberían ser menores |
| `EMBEDDING_MAX_BATCH_SIZE` | `32` | `64` | Límite de entrada; el lote efectivo por defecto es 8/16 |

## Entradas de prueba

Nemotron, conversación:

```json
{
  "messages": [
    {"role": "system", "content": "Responde en español y de forma concisa."},
    {"role": "user", "content": "Explica qué es una arquitectura MoE."}
  ],
  "max_tokens": 512,
  "temperature": 0.2,
  "top_p": 0.95,
  "thinking": true
}
```

También acepta `prompt` y `system_prompt`. La respuesta contiene `text`, `reasoning` (si el modelo emite etiquetas de razonamiento), `finish_reason` y contadores de tokens. El límite `max_tokens` nunca puede superar `LLM_MAX_OUTPUT_TOKENS`.

Embedding de documentos:

```json
{
  "input": [
    "Cloudera AI permite desplegar modelos como endpoints.",
    "Nemotron 3 Nano utiliza una arquitectura híbrida MoE."
  ],
  "input_type": "passage",
  "normalize": true
}
```

Para buscar esos documentos, genere el vector de la consulta con `input_type=query`. La salida sigue una forma similar a `/v1/embeddings`: `data[].embedding` y dimensión 1024. BGE-M3 no añade prefijos; el campo solo documenta el tipo de entrada.

## Integración con RAG Studio

RAG Studio exige que **todos** los endpoints utilicen el estándar OpenAI. Además, cuando usa el proveedor `Cloudera AI`, descubre los endpoints de Cloudera AI Inference service y usa su atributo de tarea:

- Nemotron: `TEXT_GENERATION` (o `TEXT_TO_TEXT_GENERATION`), API standard `openai`.
- BGE-M3: `EMBED`, API standard `openai`.

Los cuatro `model_*.py` de este proyecto son endpoints clásicos de **Workbench Models** con `api_wrapper`; no exponen por sí mismos `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` ni streaming SSE. Por tanto, no deben registrarse directamente en RAG Studio como si fueran OpenAI. Son útiles para consumidores del API de Workbench y para probar la carga en la GPU.

Para RAG Studio, la ruta recomendada es:

1. En SP3, importar ambos repositorios de Hugging Face en Cloudera AI Registry y desplegarlos con **Cloudera AI Inference service**.
2. Usar los parámetros de `rag_studio/*.args` según GPU.
3. Confirmar que las tareas son `TEXT_GENERATION` y `EMBED`, y que ambos endpoints aparecen correctamente en **Settings > Model Configuration** de RAG Studio.
4. Elegir `Cloudera AI` como proveedor e indicar el dominio de Inference service cuando la interfaz lo solicite.
5. Crear una colección vectorial nueva para BGE-M3; su dimensión es 1024.

En SP2, el Inference service gestionado lleva vLLM 0.8.5 y no soporta la arquitectura `NemotronH` de Nemotron 3. El wrapper de Workbench no soluciona el contrato OpenAI. Para integrar RAG Studio en SP2 se necesita un servidor/adaptador OpenAI-compatible separado que implemente descubrimiento, chat y embeddings, o bien actualizar a SP3. No basta con desactivar streaming.

Aunque Nemotron acepte 262K, para chats RAG conviene comenzar con un presupuesto operativo de 32K-64K: deja espacio para instrucciones, historial, fragmentos recuperados y respuesta, y reduce mucho la latencia de prefill. Use 262K para conversaciones o conjuntos documentales que realmente lo necesiten. RAG Studio permite limitar el número de documentos; 10 fragmentos de 512 tokens consumen aproximadamente 5K tokens antes de instrucciones e historial.

## SP2 frente a SP3

Hay dos rutas de serving diferentes y no conviene mezclarlas:

1. **Este repositorio usa Cloudera AI Workbench Models.** Las dependencias se instalan durante el build mediante `cdsw-build.sh`, por lo que los Python funcionan tanto en SP2 como en SP3 siempre que el Runtime/driver sea compatible con la rueda de vLLM y la GPU sea completa.
2. **Cloudera AI Inference service es el servicio gestionado.** SP2 incluye vLLM 0.8.5, anterior a Nemotron 3 Nano y sin esta arquitectura en su matriz; no es la ruta recomendada para este modelo. SP3 incluye Hugging Face Model Server con vLLM 0.20.0, declara `NemotronHForCausalLM` y ofrece NIM de Nemotron 3 Nano 30B. En SP3, para producción con OpenAI API, streaming y escalado, valore Inference service/NIM en lugar del wrapper de Workbench.

Mejoras operativas relevantes de SP3:

- asignación automática de labels/taints de nodos GPU en ECS al refrescar el perfil, frente a la configuración manual requerida normalmente en SP2;
- vLLM gestionado 0.20.0 y soporte explícito de NemotronH en Inference service;
- NIM de Nemotron 3 Nano disponible en la matriz soportada;
- cuotas de Workbench y controles de acceso más finos.

Estas mejoras **no cambian** los valores de precisión del proyecto: L40S continúa necesitando FP8 y H100 puede usar BF16. Tampoco se debe sustituir `VLLM_VERSION=0.12.0` del build de Workbench por la versión interna de Inference service: son entornos distintos. Si el equipo decide migrar al servicio gestionado de SP3, configure allí los argumentos equivalentes (`--dtype`, `--kv-cache-dtype`, `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, `--trust-remote-code`) y no use estos Python.

## Ajuste y diagnóstico

- **OOM al iniciar L40S:** confirme que el ID termina en `FP8`, que hay 48 GB visibles y reduzca `LLM_MAX_MODEL_LEN` por escalones: 131072, 65536 y 32768. Si todavía falla, baje `LLM_GPU_MEMORY_UTILIZATION` a `0.85`.
- **OOM al iniciar H100 BF16:** confirme 80 GB sin MIG; use FP8 si el Runtime reserva demasiada VRAM.
- **`No space left on device`:** mueva `HF_HOME` a almacenamiento persistente con capacidad; la caché BF16 y sus temporales necesitan bastante más que el tamaño final de pesos.
- **Error de CUDA/rueda:** el driver del nodo es demasiado antiguo para el PyTorch/CUDA que instala vLLM. Actualice el driver/Runtime o construya un Runtime Add-on validado; no instale un toolkit CUDA diferente dentro del pod para ocultar un driver incompatible.
- **El modelo reinicia tras una petición:** reduzca longitud/lote y revise logs. Cloudera reinicia el modelo cuando `api_wrapper` lanza una excepción no controlada.
- **Resultados BGE-M3 pobres:** compruebe normalización consistente, chunks semánticos y que consulta/documentos se hayan generado con exactamente el mismo checkpoint. No reutilice el índice E5 de 384 dimensiones.

## Referencias

- [Modelo Nemotron 3 Nano 30B A3B FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- [Receta oficial vLLM para Nemotron 3 Nano](https://github.com/vllm-project/recipes/blob/main/NVIDIA/Nemotron-3-Nano-30B-A3B.md)
- [Modelo BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Cloudera: hosting an LLM as a Workbench model](https://docs.cloudera.com/machine-learning/1.5.5/models/topics/ml-creating-a-cloudera-ai-model.html)
- [Cloudera SP3: inventario de servidores y modelos](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-model-server-inventory-sp3.html)
- [Cloudera SP3: novedades](https://docs.cloudera.com/machine-learning/1.5.5/release-notes-privatecloud/topics/ml-whats-new-1-5-5-sp3.html)
- [Cloudera: configuración de RAG Studio](https://docs.cloudera.com/machine-learning/cloud/setup-ai-studios/topics/ml-configuring-rag-studio.html)
