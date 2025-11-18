# Agent Lightning Test Suite

This directory contains the comprehensive test suite for the Agent Lightning framework. The tests ensure reliability, correctness, and performance of all core components.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

The Agent Lightning test suite consists of:
- **47 test files** across 13 modules
- **~26,822 lines of test code** (82% of production code size)
- Unit, integration, and end-to-end tests
- CPU and GPU test variants
- Mock components and fixtures for isolated testing

### Test Philosophy

1. **Comprehensive Coverage**: Test all major code paths and edge cases
2. **Isolation**: Use mocks and fixtures to isolate units under test
3. **Fast Feedback**: Most tests run quickly; expensive tests are marked
4. **Realistic Scenarios**: Integration tests use realistic agent workflows
5. **CI-Friendly**: Tests are deterministic and work in containerized environments

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── common/                     # Common test utilities
│   ├── tracer.py              # Mock tracers for testing
│   ├── vllm.py                # vLLM test utilities
│   └── network.py             # Network utilities (port finding, etc.)
│
├── algorithm/                  # Algorithm tests
│   ├── test_decorator.py      # Algorithm decorator tests
│   ├── test_baseline.py       # Baseline algorithm tests
│   └── test_apo.py            # APO (Automatic Prompt Optimization) tests
│
├── adapter/                    # Data adapter tests
│   ├── test_llm_proxy.py      # LLM proxy adapter tests
│   └── test_messages.py       # Message adapter tests
│
├── emitter/                    # Manual emission tests
│   ├── test_reward.py         # Reward emission tests
│   └── test_emitter.py        # General emitter tests
│
├── execution/                  # Execution strategy tests
│   ├── test_shared_memory.py  # Shared memory execution tests
│   └── test_client_server.py  # Client-server execution tests
│
├── litagent/                   # LitAgent wrapper tests
│   ├── test_decorator.py      # Agent decorator tests
│   └── test_resources.py      # Resource management tests
│
├── llm_proxy/                  # LLM proxy tests
│   ├── test_cpu.py            # CPU-based proxy tests
│   └── test_gpu.py            # GPU-based proxy tests (requires GPU)
│
├── runner/                     # Runner tests
│   ├── test_agent_runner.py   # Agent runner unit tests
│   └── test_agent_integration.py  # Runner integration tests
│
├── store/                      # Store tests
│   ├── conftest.py            # Store-specific fixtures
│   ├── dummy_store.py         # Mock store for testing
│   ├── test_memory.py         # InMemoryLightningStore tests
│   ├── test_client_server.py  # Store client-server tests
│   ├── test_threading.py      # Thread-safe store wrapper tests
│   └── test_utils.py          # Store utility tests
│
├── trainer/                    # Trainer tests
│   ├── sample_components.py   # Sample components for testing
│   ├── test_trainer_init.py   # Trainer initialization tests
│   ├── test_trainer_dev.py    # Trainer dev mode tests
│   └── test_init_utils.py     # Initialization utility tests
│
├── tracer/                     # Tracer tests
│   ├── test_integration.py    # Tracer integration tests
│   ├── test_otel.py          # OpenTelemetry tracer tests
│   └── test_http.py          # HTTP tracer tests
│
├── types/                      # Type tests
│   └── test_types.py          # Core type tests
│
├── assets/                     # Test assets
│   └── ...                    # Test data files
│
├── test_client.py              # Legacy client API tests
└── test_config.py              # Configuration tests
```

## Running Tests

### Prerequisites

```bash
# Install dependencies with test extras
pip install -e ".[test]"

# Or with uv
uv pip install -e ".[test]"
```

### Basic Usage

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests for a specific module
pytest tests/store/

# Run a specific test file
pytest tests/store/test_memory.py

# Run a specific test function
pytest tests/store/test_memory.py::test_enqueue_dequeue

# Run tests matching a pattern
pytest -k "test_store"
```

### Test Markers

Tests are marked for selective execution:

```bash
# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run only GPU tests (requires GPU)
pytest -m "gpu"

# Run only unit tests
pytest -m "unit"

# Run only integration tests
pytest -m "integration"
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with 4 workers
pytest -n 4
```

### Coverage

```bash
# Run with coverage reporting
pytest --cov=agentlightning --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Coverage

### Store Module (`tests/store/`)

**Coverage**: In-memory store, client-server communication, threading

- `test_memory.py`: Core store operations
  - Rollout enqueueing and dequeueing
  - Attempt management and retries
  - Resource versioning and updates
  - Span collection and querying
  - Concurrent access patterns

- `test_client_server.py`: Distributed store
  - HTTP server startup/shutdown
  - Client retry logic and backoff
  - Health checking and recovery
  - Process-aware delegation
  - Pickle safety across processes

- `test_threading.py`: Thread-safe wrapper
  - Lock acquisition and release
  - Concurrent operations
  - Deadlock prevention

### Algorithm Module (`tests/algorithm/`)

**Coverage**: Training algorithms and optimization

- `test_decorator.py`: Algorithm decorator
  - Resource update callbacks
  - Dataset generation
  - State management

- `test_baseline.py`: Baseline algorithms
  - FastAlgorithm implementation
  - No-op algorithm behavior

- `test_apo.py`: Automatic Prompt Optimization
  - Prompt generation
  - Optimization loops
  - Feedback integration

### Adapter Module (`tests/adapter/`)

**Coverage**: Data transformation and triplet generation

- `test_llm_proxy.py`: LLM proxy adapter
  - Span-to-triplet conversion
  - Reward matching policies
  - Tree traversal and filtering

- `test_messages.py`: Message adapter
  - Span-to-message conversion
  - Message ordering
  - Format transformations

### Runner Module (`tests/runner/`)

**Coverage**: Agent execution and rollout management

- `test_agent_runner.py`: Core runner logic
  - Rollout polling and execution
  - Trace context management
  - Hook invocation
  - Error handling and retries

- `test_agent_integration.py`: End-to-end runner tests
  - Full rollout lifecycle
  - Integration with store and tracer
  - Realistic agent workflows

### Trainer Module (`tests/trainer/`)

**Coverage**: High-level orchestration

- `test_trainer_init.py`: Component initialization
  - Dependency injection
  - Strategy selection
  - Configuration validation

- `test_trainer_dev.py`: Dev mode
  - Single-rollout execution
  - Span collection
  - Debug output

### Tracer Module (`tests/tracer/`)

**Coverage**: Automatic span collection

- `test_integration.py`: End-to-end tracing
  - OpenTelemetry integration
  - AgentOps integration
  - Span propagation

- `test_otel.py`: OpenTelemetry specifics
  - Span processor
  - Exporter configuration
  - Context management

- `test_http.py`: HTTP tracer
  - Event collection
  - HTTP transport
  - Error handling

### LLM Proxy Module (`tests/llm_proxy/`)

**Coverage**: Proxy server and span export

- `test_cpu.py`: CPU-based tests
  - Server startup/shutdown
  - Model list updates
  - Middleware functionality
  - Token ID collection

- `test_gpu.py`: GPU-based tests (requires GPU)
  - vLLM backend integration
  - Real LLM inference
  - Performance benchmarks

### Execution Module (`tests/execution/`)

**Coverage**: Execution strategies

- `test_shared_memory.py`: Thread-based execution
  - Component lifecycle
  - Thread coordination
  - Resource sharing

- `test_client_server.py`: Process-based execution
  - Process spawning
  - HTTP communication
  - Graceful shutdown

## Writing New Tests

### Test Structure Template

```python
"""
Module: tests/module_name/test_feature.py
Description: Tests for [feature description]
"""

import pytest
from agentlightning import Component


class TestFeature:
    """Test suite for Feature X."""

    @pytest.fixture
    def component(self):
        """Fixture providing a component instance."""
        return Component()

    def test_basic_operation(self, component):
        """Test basic operation of the component."""
        result = component.do_something()
        assert result == expected_value

    @pytest.mark.asyncio
    async def test_async_operation(self, component):
        """Test async operation of the component."""
        result = await component.do_something_async()
        assert result == expected_value

    @pytest.mark.slow
    def test_expensive_operation(self, component):
        """Test expensive operation (marked as slow)."""
        # This test takes a long time
        result = component.expensive_operation()
        assert result == expected_value
```

### Best Practices

1. **Use Descriptive Names**
   - Test function names should describe what they test
   - Use docstrings to explain complex test scenarios

2. **Isolate Tests**
   - Use fixtures for setup/teardown
   - Mock external dependencies
   - Don't rely on test execution order

3. **Test Edge Cases**
   - Empty inputs
   - None values
   - Boundary conditions
   - Error conditions

4. **Use Markers**
   - `@pytest.mark.slow` for tests > 1 second
   - `@pytest.mark.gpu` for GPU-required tests
   - `@pytest.mark.asyncio` for async tests
   - `@pytest.mark.integration` for integration tests

5. **Mock External Services**
   ```python
   @pytest.fixture
   def mock_llm():
       """Mock LLM for testing without API calls."""
       return MockLLM(responses=["test response"])
   ```

6. **Use Parametrize for Multiple Cases**
   ```python
   @pytest.mark.parametrize("input,expected", [
       (1, 2),
       (2, 4),
       (3, 6),
   ])
   def test_doubling(input, expected):
       assert double(input) == expected
   ```

### Common Fixtures

Available in `conftest.py`:

- `store`: InMemoryLightningStore instance
- `mock_tracer`: Mock tracer for testing
- `temp_dir`: Temporary directory for test files
- `event_loop`: Asyncio event loop

## CI/CD Integration

### GitHub Actions Workflows

1. **CPU Tests** (`.github/workflows/tests.yml`)
   - Runs on every commit
   - Executes all CPU-compatible tests
   - Checks code formatting and linting
   - Reports coverage

2. **GPU Tests** (`.github/workflows/tests-full.yml`)
   - Runs on pull requests and main branch
   - Requires GPU runners
   - Tests vLLM integration
   - Performance benchmarks

3. **Examples Validation** (`.github/workflows/badge-examples.yml`)
   - Validates example code
   - Ensures examples run successfully
   - Checks documentation accuracy

### Running CI Tests Locally

```bash
# Run the same tests as CI
pytest --cov=agentlightning --cov-report=term-missing

# Check code formatting
black --check agentlightning tests
isort --check-only agentlightning tests

# Type checking
pyright agentlightning

# Linting
flake8 agentlightning tests
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Problem**: Tests fail with "Address already in use"

**Solution**:
```bash
# Kill processes using the port
lsof -ti:8000 | xargs kill -9

# Or use a different port in tests
pytest --port=8001
```

#### 2. Event Loop Already Running

**Problem**: `RuntimeError: This event loop is already running`

**Solution**: Use `pytest-asyncio` and mark tests with `@pytest.mark.asyncio`

#### 3. GPU Tests Failing

**Problem**: GPU tests fail on CPU-only machines

**Solution**: Skip GPU tests
```bash
pytest -m "not gpu"
```

#### 4. Slow Tests Timing Out

**Problem**: Integration tests timeout

**Solution**: Increase timeout or skip slow tests
```bash
pytest -m "not slow"
# Or increase timeout
pytest --timeout=300
```

#### 5. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'agentlightning'`

**Solution**: Install package in editable mode
```bash
pip install -e .
```

### Debug Mode

```bash
# Run with debug output
pytest -vv --log-cli-level=DEBUG

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s
```

## Performance Testing

### Benchmarking

```bash
# Run with profiling
pytest --profile

# Benchmark specific tests
pytest tests/store/test_memory.py --benchmark-only
```

### Memory Profiling

```bash
# Profile memory usage
pytest --memray

# Generate memory report
memray run pytest tests/store/test_memory.py
memray flamegraph output.bin
```

## Contributing Tests

When contributing new features:

1. **Write Tests First** (TDD approach)
   - Write failing test
   - Implement feature
   - Ensure test passes

2. **Maintain Coverage**
   - Aim for >80% code coverage
   - Test both success and failure paths
   - Include edge cases

3. **Document Tests**
   - Add docstrings explaining test purpose
   - Document complex setup/teardown
   - Include usage examples

4. **Run Full Test Suite**
   ```bash
   pytest --cov=agentlightning --cov-report=term-missing
   ```

5. **Update This README**
   - Document new test modules
   - Add troubleshooting tips
   - Update coverage information

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Agent Lightning Docs](https://docs.agentlightning.ai/)

## Contact

For questions about tests:
- Open an issue on GitHub
- Join the community Discord
- Check the documentation at https://docs.agentlightning.ai/
