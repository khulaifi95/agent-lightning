# Copyright (c) Microsoft. All rights reserved.

"""
Client-Server Implementation for LightningStore

This module provides HTTP-based distributed access to a LightningStore instance, enabling
multi-process and multi-machine agent training scenarios.

Architecture Overview:
---------------------
The module implements a client-server pattern for the LightningStore:

1. **LightningStoreServer**:
   - Wraps any LightningStore implementation and exposes it via a FastAPI REST API
   - Runs a uvicorn server in a background thread
   - Handles process-aware delegation: in owner process, calls store directly;
     in other processes, automatically switches to HTTP client
   - Thread-safe with internal locking for concurrent access

2. **LightningStoreClient**:
   - HTTP client that communicates with a remote LightningStoreServer
   - Implements retry logic with exponential backoff for transient failures
   - Health checking and automatic server recovery detection
   - Thread-safe session management with per-event-loop ClientSession caching

Key Features:
-------------
- **Process-Aware**: Automatically detects if called from owner process or subprocess
- **Pickle-Safe**: Both client and server can be safely pickled and sent to subprocesses
- **Retry Logic**: Automatic retries on network failures with configurable backoff
- **Health Checking**: Probes server health before retrying failed requests
- **Error Handling**: Distinguishes application errors (4xx) from transient failures (5xx, network)
- **Async-First**: Full async/await support with proper event loop handling

UNSET Sentinel Pattern:
-----------------------
To distinguish "update field to None" from "don't update field", we use a sentinel value:
- Python side: UNSET (instance of Unset class)
- Pydantic/HTTP: PydanticUnset (serializable marker)
This allows partial updates without ambiguity.

Usage Example:
--------------
    # Server side
    store = InMemoryLightningStore()
    server = LightningStoreServer(store, "localhost", 8000)
    await server.start()

    # Client side (can be in a different process)
    client = LightningStoreClient("http://localhost:8000")
    rollout = await client.enqueue_rollout(input={"prompt": "Hello"})

    # The server can also be used directly in the owner process
    # and will automatically delegate to the underlying store
    rollout = await server.enqueue_rollout(input={"prompt": "Hello"})
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence, Union

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field

from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    RolloutStatus,
    Span,
    TaskInput,
)

from .base import UNSET, LightningStore, Unset

logger = logging.getLogger(__name__)


# ================================================================================================
# Pydantic Models for HTTP API Request/Response Serialization
# ================================================================================================
# These models define the contract between LightningStoreClient and LightningStoreServer.
# They enable JSON serialization/deserialization of store operations over HTTP.


class PydanticUnset(BaseModel):
    """
    Serializable marker for the UNSET sentinel value.

    Used in update operations to distinguish between:
    - "set field to None" (explicit None value)
    - "don't update this field" (PydanticUnset)

    This allows partial updates without ambiguity in HTTP requests.
    """

    _type: Literal["UNSET"] = "UNSET"


class RolloutRequest(BaseModel):
    """Request model for starting or enqueueing a rollout."""

    input: TaskInput
    mode: Optional[Literal["train", "val", "test"]] = None
    resources_id: Optional[str] = None
    config: Optional[RolloutConfig] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryRolloutsRequest(BaseModel):
    """Request model for querying rollouts with filters."""

    status: Optional[List[RolloutStatus]] = None
    rollout_ids: Optional[List[str]] = None


class WaitForRolloutsRequest(BaseModel):
    """Request model for waiting on multiple rollouts to complete."""

    rollout_ids: List[str]
    timeout: Optional[float] = None


class RolloutId(BaseModel):
    """Simple wrapper for rollout ID in requests."""

    rollout_id: str


class AddResourcesRequest(BaseModel):
    """Request model for adding new resources to the store."""

    resources: NamedResources


class UpdateRolloutRequest(BaseModel):
    """
    Request model for partial rollout updates.

    Uses PydanticUnset to distinguish "update to None" from "don't update".
    All fields except rollout_id are optional and default to UNSET (no update).
    """

    rollout_id: str
    input: Union[TaskInput, PydanticUnset] = Field(default_factory=PydanticUnset)
    mode: Union[Optional[Literal["train", "val", "test"]], PydanticUnset] = Field(default_factory=PydanticUnset)
    resources_id: Union[Optional[str], PydanticUnset] = Field(default_factory=PydanticUnset)
    status: Union[RolloutStatus, PydanticUnset] = Field(default_factory=PydanticUnset)
    config: Union[RolloutConfig, PydanticUnset] = Field(default_factory=PydanticUnset)
    metadata: Union[Dict[str, Any], PydanticUnset] = Field(default_factory=PydanticUnset)


class UpdateAttemptRequest(BaseModel):
    """
    Request model for partial attempt updates.

    Uses PydanticUnset to distinguish "update to None" from "don't update".
    All fields except rollout_id and attempt_id are optional and default to UNSET.
    """

    rollout_id: str
    attempt_id: Union[str, Literal["latest"]]
    status: Union[AttemptStatus, PydanticUnset] = Field(default_factory=PydanticUnset)
    worker_id: Union[str, PydanticUnset] = Field(default_factory=PydanticUnset)
    last_heartbeat_time: Union[float, PydanticUnset] = Field(default_factory=PydanticUnset)
    metadata: Union[Dict[str, Any], PydanticUnset] = Field(default_factory=PydanticUnset)


class LightningStoreServer(LightningStore):
    """
    Server wrapper that exposes a LightningStore via HTTP API.
    Delegates all operations to an underlying store implementation.

    Healthcheck and watchdog relies on the underlying store.

    `agl store` is a convenient CLI to start a store server.
    """

    def __init__(self, store: LightningStore, host: str, port: int):
        super().__init__()
        self.store = store
        self._lock = threading.Lock()
        self.host = host
        self.port = port
        self.app: FastAPI | None = FastAPI(title="LightningStore Server")
        self._setup_routes()
        self._uvicorn_config: uvicorn.Config | None = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="error"
        )
        self._uvicorn_server: uvicorn.Server | None = uvicorn.Server(self._uvicorn_config)

        self._serving_thread: Optional[threading.Thread] = None

        # Process-awareness:
        # LightningStoreServer holds a plain Python object (self.store) in one process
        # (the process that runs uvicorn/FastAPI).
        # When you multiprocessing.Process(...) and call methods on a different LightningStore instance
        # (or on a copy inherited via fork), you’re mutating another process’s memory, not the server’s memory.
        # So we need to track the owner process (whoever creates the server),
        # and only mutate the store in that process.
        self._owner_pid = os.getpid()
        self._client: Optional[LightningStoreClient] = None

    def __getstate__(self):
        """
        Control pickling to prevent server state from being sent to subprocesses.

        When LightningStoreServer is pickled (e.g., passed to a subprocess), we only
        serialize the underlying store and connection details. The FastAPI app and
        uvicorn server are excluded as they should not be transferred between processes.

        The subprocess should create its own server instance if needed.
        """
        return {
            "store": self.store,
            "host": self.host,
            "port": self.port,
            "_owner_pid": self._owner_pid,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore from pickle by reconstructing only the essential attributes.

        Note: This creates a new server instance without FastAPI/uvicorn initialized.
        Call __init__() pattern or create a new LightningStoreServer if you need
        a fully functional server in the subprocess.
        """
        self.store = state["store"]
        self.host = state["host"]
        self.port = state["port"]
        self._owner_pid = state["_owner_pid"]
        self._client = None
        self._lock = threading.Lock()
        # Do NOT reconstruct app, _uvicorn_config, _uvicorn_server
        # to avoid transferring server state to subprocess

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self):
        """Starts the FastAPI server in the background.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None
        logger.info(f"Starting server at {self.endpoint}")

        uvicorn_server = self._uvicorn_server

        def run_server_forever():
            asyncio.run(uvicorn_server.serve())

        self._serving_thread = threading.Thread(target=run_server_forever, daemon=True)
        self._serving_thread.start()

        # Wait for /health to be available
        if not await self._server_health_check():
            raise RuntimeError("Server failed to start within the 10 seconds.")

    async def _server_health_check(self) -> bool:
        """Checks if the server is healthy."""
        current_time = time.time()
        while time.time() - current_time < 10:
            async with aiohttp.ClientSession() as session:
                with suppress(Exception):
                    async with session.get(f"{self.endpoint}/health") as response:
                        if response.status == 200:
                            return True
            await asyncio.sleep(0.1)
        return False

    async def run_forever(self):
        """Runs the FastAPI server indefinitely.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None

        async def _wait_till_healthy():
            health = await self._server_health_check()
            if not health:
                raise RuntimeError("Server did not become healthy within the 10 seconds.")
            logger.info("Store server is online at %s", self.endpoint)

        # We run _wait_till_healthy and self._uvicorn_server.serve in parallel
        # until one of them raises an exception.
        await asyncio.gather(_wait_till_healthy(), self._uvicorn_server.serve())

    async def stop(self):
        """Gracefully stops the running FastAPI server.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None
        if self._uvicorn_server.started:
            logger.info("Stopping server...")
            self._uvicorn_server.should_exit = True
            if self._serving_thread is not None:
                self._serving_thread.join(timeout=10)
            self._serving_thread = None
            logger.info("Server stopped.")

    def _backend(self) -> LightningStore:
        """
        Returns the object to delegate to in *this* process.

        Process-Aware Delegation Strategy:
        -----------------------------------
        - **In the owner process** (pid == _owner_pid):
          Delegates directly to self.store for zero-overhead access.

        - **In a different process** (pid != _owner_pid):
          Automatically creates and delegates to a LightningStoreClient that
          communicates with the server via HTTP.

        This enables transparent multi-process usage:
        1. Parent process creates server and calls methods directly
        2. Subprocess receives pickled server (without FastAPI app)
        3. Subprocess calls same methods, which automatically use HTTP client

        Thread-Safety:
        --------------
        The _client is created lazily per process. Since each process has its own
        copy of the LightningStoreServer instance (via fork or pickle), there's no
        race condition across processes. Within a process, the first call creates
        the client and subsequent calls reuse it.
        """
        if os.getpid() == self._owner_pid:
            return self.store
        if self._client is None:
            self._client = LightningStoreClient(self.endpoint)
        return self._client

    def _setup_routes(self):
        """
        Set up FastAPI routes for all LightningStore operations.

        Creates a REST API with the following design principles:

        1. **Error Handling Strategy**:
           - Application errors (validation, business logic) → 400 Bad Request
           - Network/server failures → 5xx status codes
           - Clients use status code to decide whether to retry

        2. **Endpoint Mapping**:
           Each LightningStore method gets a corresponding HTTP endpoint:
           - POST for operations that modify state or require request body
           - GET for simple queries and retrievals

        3. **Request/Response Serialization**:
           - Request models (RolloutRequest, etc.) validate incoming JSON
           - Response models automatically serialize return values
           - UNSET sentinel is converted between PydanticUnset and UNSET

        4. **Middleware**:
           - Request timing logged for all endpoints
           - Exception handler converts Python exceptions to 400 responses
        """
        assert self.app is not None

        @self.app.exception_handler(Exception)
        async def _app_exception_handler(request: Request, exc: Exception):  # pyright: ignore[reportUnusedFunction]
            """
            Convert unhandled application exceptions into 400 Bad Request responses.

            Error Classification:
            ---------------------
            - **400 (this handler)**: Application errors (validation, business logic bugs)
              → Client should NOT retry automatically
            - **5xx**: Server/network failures (connection issues, timeouts)
              → Client WILL retry with backoff

            This distinction allows the client to implement smart retry logic:
            - Retry on transient failures (network, server restart)
            - Fail fast on application errors (bad request, invalid state)

            The traceback is included to aid debugging during development.
            """
            logger.exception("Unhandled application error", exc_info=exc)
            return JSONResponse(
                status_code=400,
                content={
                    "detail": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )

        @self.app.middleware("http")
        async def _log_time(  # pyright: ignore[reportUnusedFunction]
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ):
            """
            Middleware to log request timing and client information.

            Logs:
            - Client address (host:port)
            - HTTP method and path
            - Response status code
            - Request duration in milliseconds

            This provides visibility into store performance and client behavior.
            """
            start = time.perf_counter()
            response = await call_next(request)
            duration = (time.perf_counter() - start) * 1000
            client = request.client
            if client is None:
                client_address = "unknown"
            else:
                client_address = f"{client.host}:{client.port}"
            logger.info(
                f"{client_address} - "
                f'"{request.method} {request.url.path} HTTP/{request.scope["http_version"]}" '
                f"{response.status_code} in {duration:.2f} ms"
            )
            return response

        @self.app.get("/health")
        async def health():  # pyright: ignore[reportUnusedFunction]
            return {"status": "ok"}

        @self.app.post("/start_rollout", response_model=AttemptedRollout)
        async def start_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.start_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @self.app.post("/enqueue_rollout", response_model=Rollout)
        async def enqueue_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.enqueue_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @self.app.get("/dequeue_rollout", response_model=Optional[AttemptedRollout])
        async def dequeue_rollout():  # pyright: ignore[reportUnusedFunction]
            return await self.dequeue_rollout()

        @self.app.post("/start_attempt", response_model=AttemptedRollout)
        async def start_attempt(request: RolloutId):  # pyright: ignore[reportUnusedFunction]
            return await self.start_attempt(request.rollout_id)

        @self.app.post("/query_rollouts", response_model=List[Rollout])
        async def query_rollouts(request: QueryRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.query_rollouts(status=request.status, rollout_ids=request.rollout_ids)

        @self.app.get("/query_attempts/{rollout_id}", response_model=List[Attempt])
        async def query_attempts(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.query_attempts(rollout_id)

        @self.app.get("/get_latest_attempt/{rollout_id}", response_model=Optional[Attempt])
        async def get_latest_attempt(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_latest_attempt(rollout_id)

        @self.app.get("/get_rollout_by_id/{rollout_id}", response_model=Optional[Rollout])
        async def get_rollout_by_id(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_rollout_by_id(rollout_id)

        @self.app.post("/add_resources", response_model=ResourcesUpdate)
        async def add_resources(resources: AddResourcesRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.add_resources(resources.resources)

        @self.app.post("/update_resources", response_model=ResourcesUpdate)
        async def update_resources(update: ResourcesUpdate):  # pyright: ignore[reportUnusedFunction]
            return await self.update_resources(update.resources_id, update.resources)

        @self.app.get("/get_resources_by_id/{resources_id}", response_model=Optional[ResourcesUpdate])
        async def get_resources_by_id(resources_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_resources_by_id(resources_id)

        @self.app.get("/get_latest_resources", response_model=Optional[ResourcesUpdate])
        async def get_latest_resources():  # pyright: ignore[reportUnusedFunction]
            return await self.get_latest_resources()

        @self.app.post("/add_span", response_model=Span)
        async def add_span(span: Span):  # pyright: ignore[reportUnusedFunction]
            return await self.add_span(span)

        @self.app.get("/get_next_span_sequence_id/{rollout_id}/{attempt_id}", response_model=int)
        async def get_next_span_sequence_id(rollout_id: str, attempt_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_next_span_sequence_id(rollout_id, attempt_id)

        @self.app.post("/wait_for_rollouts", response_model=List[Rollout])
        async def wait_for_rollouts(request: WaitForRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.wait_for_rollouts(rollout_ids=request.rollout_ids, timeout=request.timeout)

        @self.app.get("/query_spans/{rollout_id}", response_model=List[Span])
        async def query_spans(  # pyright: ignore[reportUnusedFunction]
            rollout_id: str, attempt_id: Optional[str] = None
        ):
            return await self.query_spans(rollout_id, attempt_id)

        @self.app.post("/update_rollout", response_model=Rollout)
        async def update_rollout(request: UpdateRolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.update_rollout(
                rollout_id=request.rollout_id,
                input=request.input if not isinstance(request.input, PydanticUnset) else UNSET,
                mode=request.mode if not isinstance(request.mode, PydanticUnset) else UNSET,
                resources_id=request.resources_id if not isinstance(request.resources_id, PydanticUnset) else UNSET,
                status=request.status if not isinstance(request.status, PydanticUnset) else UNSET,
                config=request.config if not isinstance(request.config, PydanticUnset) else UNSET,
                metadata=request.metadata if not isinstance(request.metadata, PydanticUnset) else UNSET,
            )

        @self.app.post("/update_attempt", response_model=Attempt)
        async def update_attempt(request: UpdateAttemptRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.update_attempt(
                rollout_id=request.rollout_id,
                attempt_id=request.attempt_id,
                status=request.status if not isinstance(request.status, PydanticUnset) else UNSET,
                worker_id=request.worker_id if not isinstance(request.worker_id, PydanticUnset) else UNSET,
                last_heartbeat_time=(
                    request.last_heartbeat_time if not isinstance(request.last_heartbeat_time, PydanticUnset) else UNSET
                ),
                metadata=request.metadata if not isinstance(request.metadata, PydanticUnset) else UNSET,
            )

    # ============================================================================================
    # Delegate Methods
    # ============================================================================================
    # All LightningStore interface methods delegate to _call_store_method, which in turn
    # delegates to either the in-process store or the HTTP client depending on the current process.

    async def _call_store_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Unified delegation method that handles both in-process and cross-process calls.

        Delegation Logic:
        -----------------
        1. Determine backend (in-process store or HTTP client) via _backend()
        2. Get the method by name from the backend
        3. If backend is in-process store:
           a. For wait_for_rollouts: call without lock (can block for long time)
           b. For all other methods: call with lock held (thread-safety)
        4. If backend is HTTP client: call without lock (client has its own locking)

        Locking Strategy:
        -----------------
        The lock protects the in-process store from concurrent access within the owner process.
        - **wait_for_rollouts**: Can block indefinitely waiting for rollouts to complete.
          We don't hold the lock to avoid blocking other concurrent requests.
        - **Other methods**: Quick operations that modify or query store state.
          We hold the lock to ensure thread-safe access to the underlying store.
        - **HTTP client**: Has its own session management and thread-safety, no lock needed.

        Args:
            method_name: Name of the LightningStore method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            The result of calling the store method
        """
        backend = self._backend()
        method = getattr(backend, method_name)
        if backend is self.store:
            if method_name == "wait_for_rollouts":
                # wait_for_rollouts can block for a long time; avoid holding the lock
                # so other requests can make progress while we wait.
                return await method(*args, **kwargs)
            with self._lock:
                return await method(*args, **kwargs)
        return await method(*args, **kwargs)

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        return await self._call_store_method(
            "start_rollout",
            input,
            mode,
            resources_id,
            config,
            metadata,
        )

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        return await self._call_store_method(
            "enqueue_rollout",
            input,
            mode,
            resources_id,
            config,
            metadata,
        )

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        return await self._call_store_method("dequeue_rollout")

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        return await self._call_store_method("start_attempt", rollout_id)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        return await self._call_store_method("query_rollouts", status=status, rollout_ids=rollout_ids)

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        return await self._call_store_method("query_attempts", rollout_id)

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        return await self._call_store_method("get_latest_attempt", rollout_id)

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        return await self._call_store_method("get_rollout_by_id", rollout_id)

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        return await self._call_store_method("add_resources", resources)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        return await self._call_store_method("update_resources", resources_id, resources)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        return await self._call_store_method("get_resources_by_id", resources_id)

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        return await self._call_store_method("get_latest_resources")

    async def add_span(self, span: Span) -> Span:
        return await self._call_store_method("add_span", span)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        return await self._call_store_method("get_next_span_sequence_id", rollout_id, attempt_id)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        return await self._call_store_method(
            "add_otel_span",
            rollout_id,
            attempt_id,
            readable_span,
            sequence_id,
        )

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        return await self._call_store_method("wait_for_rollouts", rollout_ids=rollout_ids, timeout=timeout)

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        return await self._call_store_method("query_spans", rollout_id, attempt_id)

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        return await self._call_store_method(
            "update_rollout",
            rollout_id,
            input,
            mode,
            resources_id,
            status,
            config,
            metadata,
        )

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        return await self._call_store_method(
            "update_attempt",
            rollout_id,
            attempt_id,
            status,
            worker_id,
            last_heartbeat_time,
            metadata,
        )


class LightningStoreClient(LightningStore):
    """
    HTTP client that talks to a remote LightningStoreServer with intelligent retry logic.

    This client implements sophisticated failure handling to ensure reliable distributed
    operation even in the presence of network issues and server restarts.

    Key Features:
    -------------
    1. **Automatic Retry with Backoff**:
       - Retries transient failures (network errors, 5xx status codes)
       - Configurable exponential backoff schedule
       - Distinguishes retriable vs non-retriable errors

    2. **Health Checking**:
       - Probes server /health endpoint before retrying failed requests
       - Avoids wasted retries when server is known to be down
       - Separate health probe backoff schedule

    3. **Per-Event-Loop Session Management**:
       - Maintains one aiohttp.ClientSession per event loop
       - Handles multi-threaded scenarios (e.g., OpenTelemetry exporter on separate thread)
       - Thread-safe session creation and lookup

    4. **Smart Error Classification**:
       - **4xx (except 408)**: Application errors → Fail immediately, no retry
       - **5xx**: Server errors → Retry with health checking
       - **Network errors**: Connection issues → Retry with health checking

    5. **Special Handling for Polling Operations**:
       - dequeue_rollout: No retry (returns None immediately on failure)
       - wait_for_rollouts: Strict timeout limit to avoid blocking event loop

    Args:
        server_address: The URL of the LightningStoreServer (e.g., "http://localhost:8000")
        retry_delays: Backoff schedule in seconds for retrying failed requests.
            Default: (1.0, 2.0, 5.0) means retry after 1s, 2s, then 5s before giving up.
        health_retry_delays: Delays in seconds between /health probe attempts.
            Default: (0.1, 0.2, 0.5) for fast health detection.

    Thread-Safety:
    --------------
    This class is thread-safe. Multiple threads can share a single instance and
    make concurrent requests. Each thread's event loop gets its own ClientSession.

    Process-Safety:
    ---------------
    Safe to pickle and send to subprocesses. The __getstate__/__setstate__ methods
    ensure sessions are not transferred across process boundaries.

    Example:
        client = LightningStoreClient("http://localhost:8000")
        rollout = await client.enqueue_rollout(input={"prompt": "Hello"})
        attempt = await client.dequeue_rollout()  # Returns None if queue empty
    """

    def __init__(
        self,
        server_address: str,
        *,
        retry_delays: Sequence[float] = (1.0, 2.0, 5.0),
        health_retry_delays: Sequence[float] = (0.1, 0.2, 0.5),
    ):
        self.server_address = server_address.rstrip("/")
        self._sessions: Dict[int, aiohttp.ClientSession] = {}  # id(loop) -> ClientSession
        self._lock = threading.RLock()

        # retry config
        self._retry_delays = tuple(float(d) for d in retry_delays)
        self._health_retry_delays = tuple(float(d) for d in health_retry_delays)

        # Store whether the dequeue was successful in history
        self._dequeue_was_successful: bool = False
        self._dequeue_first_unsuccessful: bool = True

    def __getstate__(self):
        """
        When LightningStoreClient is pickled (e.g., passed to a subprocess), we only
        serialize the server address and retry configurations. The ClientSessions
        are excluded as they should not be transferred between processes.
        """
        return {
            "server_address": self.server_address,
            "_retry_delays": self._retry_delays,
            "_health_retry_delays": self._health_retry_delays,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore from pickle by reconstructing only the essential attributes.

        Replicating `__init__` logic to create another client instance in the subprocess.
        """
        self.server_address = state["server_address"]
        self._sessions = {}
        self._lock = threading.RLock()
        self._retry_delays = state["_retry_delays"]
        self._health_retry_delays = state["_health_retry_delays"]
        self._dequeue_was_successful = False
        self._dequeue_first_unsuccessful = True

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp.ClientSession for the current event loop.

        Problem: Multi-Threaded Event Loops
        ------------------------------------
        In the LLM proxy process, we have multiple event loops across different threads:

        1. **Main Event Loop (uvicorn/FastAPI thread)**:
           - Handles HTTP requests from runners
           - Calls client_store.get_next_span_sequence_id(...)

        2. **OpenTelemetry Exporter Loop (separate thread)**:
           - Runs span export/flush on its own private event loop
           - Calls client_store.add_otel_span(...) -> client_store.add_span(...)

        aiohttp.ClientSession is NOT loop-agnostic or thread-safe:
        - A session created on loop A cannot be used from loop B
        - Attempting to do so can hang indefinitely or raise runtime errors

        Solution: Per-Loop Session Cache
        ---------------------------------
        We maintain a mapping: event_loop_id -> ClientSession
        - Each event loop gets its own ClientSession instance
        - Sessions are created lazily on first use per loop
        - Thread-safe via RLock (protects the dictionary)

        Timeout Configuration:
        ----------------------
        - total: 30s - maximum time for entire request
        - connect: 5s - maximum time to establish connection
        - sock_connect: 5s - socket-level connection timeout
        - sock_read: 30s - maximum time to read response

        Returns:
            aiohttp.ClientSession bound to the current event loop
        """
        loop = asyncio.get_running_loop()
        key = id(loop)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None or sess.closed:
                timeout = aiohttp.ClientTimeout(total=30.0, connect=5.0, sock_connect=5.0, sock_read=30.0)
                sess = aiohttp.ClientSession(timeout=timeout)
                self._sessions[key] = sess
        return sess

    async def _wait_until_healthy(self, session: aiohttp.ClientSession) -> bool:
        """
        Probe the server's /health until it responds 200 or retries are exhausted.
        Returns True if healthy, False otherwise.
        """
        logger.info(f"Waiting for server to be healthy at {self.server_address}/health")
        for delay in [*self._health_retry_delays, 0.0]:
            try:
                async with session.get(f"{self.server_address}/health") as r:
                    if r.status == 200:
                        logger.info(f"Server is healthy at {self.server_address}/health")
                        return True
            except Exception:
                # swallow and retry
                if delay > 0.0:
                    logger.warning(f"Server is not healthy yet. Retrying in {delay} seconds.")
            if delay > 0.0:
                await asyncio.sleep(delay)
        logger.error(
            f"Server is not healthy at {self.server_address}/health after {len(self._health_retry_delays)} retry attempts"
        )
        return False

    async def _request_json(
        self,
        method: Literal["get", "post"],
        path: str,
        *,
        json: Any | None = None,
    ) -> Any:
        """
        Make an HTTP request with intelligent retry logic and error classification.

        Request Flow:
        -------------
        1. **Initial attempt**: Try the request immediately
        2. **On failure**:
           a. Classify the error (retriable vs non-retriable)
           b. For retriable errors: probe server health
           c. If healthy, sleep for backoff duration and retry
        3. **Repeat** until success or retries exhausted

        Error Classification:
        ---------------------
        1. **Non-Retriable (immediate failure)**:
           - 4xx status codes (except 408 Request Timeout)
           - These indicate application errors (bad request, invalid state)
           - Server marked these as 400 via exception handler
           - Retrying won't help; fail fast

        2. **Retriable (with health check and backoff)**:
           - 5xx status codes: Server errors (likely transient)
           - 408 Request Timeout: Transient timeout
           - Network errors:
             * ServerDisconnectedError: Server closed connection
             * ClientConnectorError: Cannot connect to server
             * ClientOSError: OS-level network error
             * TimeoutError: Request timed out

        Health Checking Before Retry:
        ------------------------------
        Before retrying a failed request, we probe the /health endpoint.
        This avoids wasting retry attempts when the server is known to be down.
        If health check fails, we abort retries immediately.

        Backoff Schedule:
        -----------------
        - Attempt 0: Immediate (no delay)
        - Attempt 1: Wait retry_delays[0] seconds (default: 1.0s)
        - Attempt 2: Wait retry_delays[1] seconds (default: 2.0s)
        - Attempt 3: Wait retry_delays[2] seconds (default: 5.0s)
        - If all retries exhausted, raise the last exception

        Args:
            method: HTTP method ("get" or "post")
            path: URL path (e.g., "/enqueue_rollout")
            json: Optional JSON payload for POST requests

        Returns:
            Parsed JSON response (dict, list, or scalar like int)

        Raises:
            aiohttp.ClientResponseError: For non-retriable 4xx errors
            Exception: Last exception if all retries exhausted

        Example:
            # This will retry on network failures
            result = await client._request_json("post", "/enqueue_rollout", json={...})

            # This will fail immediately on application error (e.g., invalid input)
            try:
                result = await client._request_json("post", "/invalid_endpoint")
            except aiohttp.ClientResponseError as e:
                print(f"Application error: {e.status}")
        """
        session = await self._get_session()
        url = f"{self.server_address}{path if path.startswith('/') else '/'+path}"

        # attempt 0 is immediate, then follow retry schedule
        attempts = (0.0,) + self._retry_delays
        last_exc: Exception | None = None

        for delay in attempts:
            if delay:
                logger.info(f"Waiting {delay} seconds before retrying {method}: {path}")
                await asyncio.sleep(delay)
            try:
                http_call = getattr(session, method)
                async with http_call(url, json=json) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientResponseError as cre:
                # Respect app-level 4xx as final (server marks app faults as 400)
                # 4xx => application issue; do not retry (except 408 which is transient)
                logger.debug(f"ClientResponseError: {cre.status} {cre.message}", exc_info=True)
                if 400 <= cre.status < 500 and cre.status != 408:
                    raise
                # 5xx and others will be retried below if they raise
                last_exc = cre
                logger.info(f"5xx and other status codes will be retried. Retrying the request {method}: {path}")
                # before next retry, ensure server is healthy
                if not await self._wait_until_healthy(session):
                    break  # server is not healthy, do not retry
            except (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientOSError,
                asyncio.TimeoutError,
            ) as net_exc:
                # Network/session issue: probe health before retrying
                logger.debug(f"Network/session issue: {net_exc}", exc_info=True)
                last_exc = net_exc
                logger.info(f"Network/session issue will be retried. Retrying the request {method}: {path}")
                if not await self._wait_until_healthy(session):
                    break  # server is not healthy, do not retry

        # exhausted retries
        assert last_exc is not None
        raise last_exc

    async def close(self):
        """Close the HTTP session."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        # close them on their own loops to avoid warnings
        async def _close(sess: aiohttp.ClientSession):
            if not sess.closed:
                await sess.close()

        # If called from one loop, best-effort close here.
        for s in sessions:
            try:
                await _close(s)
            except RuntimeError:
                # If created on a different loop/thread, schedule a thread-safe close
                # Fallback: close without awaiting (library tolerates it in practice),
                # or keep a per-loop shutdown hook where they were created.
                pass

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        data = await self._request_json(
            "post",
            "/start_rollout",
            json=RolloutRequest(
                input=input,
                mode=mode,
                resources_id=resources_id,
                config=config,
                metadata=metadata,
            ).model_dump(exclude_none=False),
        )
        return AttemptedRollout.model_validate(data)

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        data = await self._request_json(
            "post",
            "/enqueue_rollout",
            json=RolloutRequest(
                input=input,
                mode=mode,
                resources_id=resources_id,
                config=config,
                metadata=metadata,
            ).model_dump(exclude_none=False),
        )
        return Rollout.model_validate(data)

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """
        Dequeue a rollout from the server queue.

        Returns:
            AttemptedRollout if a rollout is available, None if queue is empty.

        Note:
            This method does NOT retry on failures. If any exception occurs (network error,
            server error, etc.), it logs the error and returns None immediately.
        """
        session = await self._get_session()
        url = f"{self.server_address}/dequeue_rollout"
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._dequeue_was_successful = True
                return AttemptedRollout.model_validate(data) if data else None
        except Exception as e:
            if self._dequeue_was_successful:
                if self._dequeue_first_unsuccessful:
                    logger.warning(f"dequeue_rollout failed with exception: {e}")
                    self._dequeue_first_unsuccessful = False
            logger.debug("dequeue_rollout failed with exception. Details:", exc_info=True)
            # Else ignore the exception because the server is not ready yet
            return None

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        data = await self._request_json(
            "post",
            "/start_attempt",
            json=RolloutId(rollout_id=rollout_id).model_dump(),
        )
        return AttemptedRollout.model_validate(data)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        data = await self._request_json(
            "post",
            "/query_rollouts",
            json=QueryRolloutsRequest(
                status=list(status) if status else None,
                rollout_ids=list(rollout_ids) if rollout_ids else None,
            ).model_dump(),
        )
        return [Rollout.model_validate(item) for item in data]

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        data = await self._request_json("get", f"/query_attempts/{rollout_id}")
        return [Attempt.model_validate(item) for item in data]

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """
        Get the latest attempt for a rollout.

        Args:
            rollout_id: ID of the rollout to query.

        Returns:
            Attempt if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", f"/get_latest_attempt/{rollout_id}")
            return Attempt.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_latest_attempt failed after all retries for rollout_id={rollout_id}: {e}", exc_info=True)
            return None

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        """
        Get a rollout by its ID.

        Args:
            rollout_id: ID of the rollout to retrieve.

        Returns:
            Rollout if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", f"/get_rollout_by_id/{rollout_id}")
            return Rollout.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_rollout_by_id failed after all retries for rollout_id={rollout_id}: {e}", exc_info=True)
            return None

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        request = AddResourcesRequest(resources=resources)
        data = await self._request_json("post", "/add_resources", json=request.model_dump())
        return ResourcesUpdate.model_validate(data)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        data = await self._request_json(
            "post",
            "/update_resources",
            json=ResourcesUpdate(resources_id=resources_id, resources=resources).model_dump(),
        )
        return ResourcesUpdate.model_validate(data)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """
        Get resources by their ID.

        Args:
            resources_id: ID of the resources to retrieve.

        Returns:
            ResourcesUpdate if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", f"/get_resources_by_id/{resources_id}")
            return ResourcesUpdate.model_validate(data) if data else None
        except Exception as e:
            logger.error(
                f"get_resources_by_id failed after all retries for resources_id={resources_id}: {e}", exc_info=True
            )
            return None

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """
        Get the latest resources.

        Returns:
            ResourcesUpdate if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", "/get_latest_resources")
            return ResourcesUpdate.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_latest_resources failed after all retries: {e}", exc_info=True)
            return None

    async def add_span(self, span: Span) -> Span:
        data = await self._request_json("post", "/add_span", json=span.model_dump(mode="json"))
        return Span.model_validate(data)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        data = await self._request_json("get", f"/get_next_span_sequence_id/{rollout_id}/{attempt_id}")
        # endpoint returns a plain JSON number
        return int(data)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        # unchanged logic, now benefits from retries inside add_span/get_next_span_sequence_id
        if sequence_id is None:
            sequence_id = await self.get_next_span_sequence_id(rollout_id, attempt_id)
        span = Span.from_opentelemetry(
            readable_span,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
        )
        await self.add_span(span)
        return span

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """Wait for rollouts to complete.

        Args:
            rollout_ids: List of rollout IDs to wait for.
            timeout: Timeout in seconds. If not None, the method will raise a ValueError if the timeout is greater than 0.1 seconds.

        Returns:
            List of rollouts that are completed.
        """
        if timeout is not None and timeout > 0.1:
            raise ValueError(
                "Timeout must be less than 0.1 seconds in LightningStoreClient to avoid blocking the event loop"
            )
        data = await self._request_json(
            "post",
            "/wait_for_rollouts",
            json=WaitForRolloutsRequest(rollout_ids=rollout_ids, timeout=timeout).model_dump(),
        )
        return [Rollout.model_validate(item) for item in data]

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        path = f"/query_spans/{rollout_id}"
        if attempt_id is not None:
            path += f"?attempt_id={attempt_id}"
        data = await self._request_json("get", path)
        return [Span.model_validate(item) for item in data]

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        payload: Dict[str, Any] = {"rollout_id": rollout_id}
        if not isinstance(input, Unset):
            payload["input"] = input
        if not isinstance(mode, Unset):
            payload["mode"] = mode
        if not isinstance(resources_id, Unset):
            payload["resources_id"] = resources_id
        if not isinstance(status, Unset):
            payload["status"] = status
        if not isinstance(config, Unset):
            payload["config"] = config.model_dump()
        if not isinstance(metadata, Unset):
            payload["metadata"] = metadata

        data = await self._request_json("post", "/update_rollout", json=payload)
        return Rollout.model_validate(data)

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        payload: Dict[str, Any] = {
            "rollout_id": rollout_id,
            "attempt_id": attempt_id,
        }
        if not isinstance(status, Unset):
            payload["status"] = status
        if not isinstance(worker_id, Unset):
            payload["worker_id"] = worker_id
        if not isinstance(last_heartbeat_time, Unset):
            payload["last_heartbeat_time"] = last_heartbeat_time
        if not isinstance(metadata, Unset):
            payload["metadata"] = metadata

        data = await self._request_json("post", "/update_attempt", json=payload)
        return Attempt.model_validate(data)
