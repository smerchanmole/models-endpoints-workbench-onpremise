"""Punto de entrada PBJ de Cloudera para Qwen3.8.

Cloudera ejecuta este fichero como código de una celda Jupyter con el Python
base del ML Runtime. Ese proceso debe conservar ``cml.models_v1`` y las
extensiones de Cloudera, por lo que no se le añaden los site-packages de vLLM.
En su lugar crea un worker persistente con ``qwen3_8/.venv/bin/python`` y usa
un socket local para transportar JSON. Así se aíslan NumPy 2, protobuf, Torch
y vLLM del NumPy 1.x y de las extensiones binarias incluidas por SP2.

La única función que se registra en el Model Deployment es ``predict``. La
carga del módulo espera a que el worker confirme que modelo, caché y kernels
están listos; por tanto, una réplica marcada Ready puede atender inferencias.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any

try:
    import cml.models_v1 as models
except ImportError:  # Permite validar el wrapper fuera de Cloudera.
    class _Models:
        @staticmethod
        def cml_model(function):
            return function

    models = _Models()


def _model_dir() -> Path:
    """Localiza los artefactos tanto con import normal como bajo PBJ.

    PBJ puede ejecutar el fichero sin definir ``__file__``. Por eso se prueban
    también el directorio de trabajo y su subdirectorio ``qwen3_8``.
    """
    model_file = globals().get("__file__")
    candidates: list[Path] = []
    if model_file:
        candidates.append(Path(str(model_file)).resolve().parent)
    candidates.extend([Path.cwd() / "qwen3_8", Path.cwd()])
    for candidate in candidates:
        if (candidate / "worker_a100.py").is_file():
            return candidate
    raise RuntimeError(
        "No se encontró qwen3_8/worker_a100.py. Rutas comprobadas: "
        + ", ".join(map(str, candidates))
    )


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


_MODEL_DIR = _model_dir()
_VENV_PYTHON = _MODEL_DIR / ".venv" / "bin" / "python"
_WORKER_FILE = _MODEL_DIR / "worker_a100.py"
if not _VENV_PYTHON.is_file():
    raise RuntimeError(
        f"No existe el Python aislado esperado en {_VENV_PYTHON}. "
        "Ejecute cdsw-build.sh con MODEL_FAMILY=qwen3_8 y GPU_TYPE=a100."
    )

_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 128 * 1024 * 1024
_STARTUP_TIMEOUT = _env_int("QWEN_STARTUP_TIMEOUT_SECONDS", 1800, 60, 7200)
_IPC_LOCK = threading.Lock()


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            code = _WORKER.poll()
            raise RuntimeError(
                "El proceso aislado de Qwen cerró la conexión "
                f"(exit_code={code}). Revise el traceback anterior del worker."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    (size,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if size > _MAX_MESSAGE_BYTES:
        raise RuntimeError(f"Respuesta IPC demasiado grande: {size} bytes")
    value = json.loads(_recv_exact(sock, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("El worker devolvió una respuesta IPC inválida")
    return value


def _send_message(sock: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Petición IPC demasiado grande: {len(payload)} bytes")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


_PARENT_SOCKET, _CHILD_SOCKET = socket.socketpair()
_worker_env = os.environ.copy()
_worker_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
_worker_env.setdefault("TOKENIZERS_PARALLELISM", "false")
_WORKER = subprocess.Popen(
    [
        str(_VENV_PYTHON),
        str(_WORKER_FILE),
        "--ipc-fd",
        str(_CHILD_SOCKET.fileno()),
    ],
    cwd=str(_MODEL_DIR),
    env=_worker_env,
    pass_fds=(_CHILD_SOCKET.fileno(),),
    close_fds=True,
)
# El descriptor hijo ya pertenece al worker. Cerrarlo en el proxy es necesario
# para que un cierre del worker produzca EOF y no deje el socket vivo por error.
_CHILD_SOCKET.close()


def _stop_worker() -> None:
    """Cierra el worker de forma ordenada y lo fuerza solo si no responde."""
    try:
        _PARENT_SOCKET.close()
    finally:
        if _WORKER.poll() is None:
            _WORKER.terminate()
            try:
                _WORKER.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _WORKER.kill()


atexit.register(_stop_worker)

# Workbench no marca la réplica como Ready hasta que el proceso aislado ha
# cargado pesos y kernels o devuelve el traceback completo del fallo.
_PARENT_SOCKET.settimeout(_STARTUP_TIMEOUT)
try:
    _ready = _recv_message(_PARENT_SOCKET)
except socket.timeout as exc:
    _stop_worker()
    raise RuntimeError(
        f"Qwen no terminó de iniciar en {_STARTUP_TIMEOUT} segundos"
    ) from exc
finally:
    if _PARENT_SOCKET.fileno() != -1:
        _PARENT_SOCKET.settimeout(None)

if not _ready.get("ok"):
    _stop_worker()
    raise RuntimeError(
        "El proceso aislado de Qwen no pudo iniciar:\n"
        + str(_ready.get("error", "error desconocido"))
    )


@models.cml_model
def predict(args: dict[str, Any]) -> dict[str, Any]:
    """Envía una petición JSON al motor Qwen persistente y aislado.

    El lock serializa peticiones porque el perfil A100 está deliberadamente
    configurado para una secuencia concurrente y prioriza el contexto de 262K.
    Los errores del worker se devuelven con traceback para que aparezca la causa
    original en el log del Model Deployment.
    """
    if not isinstance(args, dict):
        raise ValueError("La entrada debe ser un objeto JSON")

    with _IPC_LOCK:
        if _WORKER.poll() is not None:
            raise RuntimeError(
                f"El proceso aislado de Qwen terminó con código {_WORKER.returncode}"
            )
        _send_message(_PARENT_SOCKET, {"command": "predict", "args": args})
        response = _recv_message(_PARENT_SOCKET)

    if not response.get("ok"):
        raise RuntimeError("La inferencia Qwen falló:\n" + str(response.get("error")))
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("El worker Qwen devolvió un resultado inválido")
    return result


api_wrapper = predict
