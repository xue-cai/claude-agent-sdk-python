# Tools in Claude Agent SDK Python — Technical Analysis

## Table of Contents

1. [Overview](#overview)
2. [Tool Categories](#tool-categories)
3. [Built-in Tools](#1-built-in-tools)
4. [External MCP Servers](#2-external-mcp-servers)
5. [SDK MCP Servers (In-Process)](#3-sdk-mcp-servers-in-process)
6. [How SDK MCP Tools Actually Work](#how-sdk-mcp-tools-actually-work)
7. [Authentication & Auth Handling](#authentication--auth-handling)
8. [Tool Permission Callbacks](#tool-permission-callbacks)
9. [Hook System](#hook-system)
10. [Custom Agents (Skills)](#custom-agents-skills)
11. [Transport Layer](#transport-layer)
12. [Control Protocol](#control-protocol)
13. [Message Parsing](#message-parsing)
14. [Architecture Diagram](#architecture-diagram)

---

## Overview

The Claude Agent SDK Python provides a Python interface to the **Claude Code CLI**. The SDK does **not** call Anthropic APIs directly — instead, it spawns a **Claude Code CLI subprocess** and communicates with it over **stdin/stdout** using a JSON-based streaming protocol.

Tools in this SDK fall into three categories, and the SDK implements a sophisticated bidirectional **control protocol** for handling tool permissions, hooks, and in-process MCP servers.

---

## Tool Categories

| Category | Where it runs | How it's configured | Example |
|---|---|---|---|
| **Built-in Tools** | Inside Claude Code CLI | `allowed_tools` / `tools` | `Read`, `Write`, `Bash`, `Grep` |
| **External MCP Servers** | Separate process (stdio/SSE/HTTP) | `mcp_servers` dict | A Python calculator server |
| **SDK MCP Servers** | In-process (your Python app) | `create_sdk_mcp_server()` | `@tool` decorated functions |

---

## 1. Built-in Tools

Built-in tools are provided by the Claude Code CLI itself. They include file operations (`Read`, `Write`, `Edit`, `MultiEdit`), shell execution (`Bash`), search (`Grep`, `Glob`), and others. You enable them via `ClaudeAgentOptions`:

```python
# From src/claude_agent_sdk/types.py (lines 716-720)
@dataclass
class ClaudeAgentOptions:
    tools: list[str] | ToolsPreset | None = None
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
```

**Usage:**

```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],
    permission_mode="acceptEdits"
)
```

These get passed directly as CLI flags to the `claude` binary. From `subprocess_cli.py` (lines 195-205):

```python
if self._options.allowed_tools:
    cmd.extend(["--allowedTools", ",".join(self._options.allowed_tools)])

if self._options.disallowed_tools:
    cmd.extend(["--disallowedTools", ",".join(self._options.disallowed_tools)])
```

The SDK has **no implementation** of these tools — it simply tells the CLI which ones to enable, and the CLI handles execution internally.

---

## 2. External MCP Servers

External MCP (Model Context Protocol) servers run as **separate processes** and communicate via stdio, SSE, or HTTP. They are configured in `mcp_servers`:

```python
# Type definitions from src/claude_agent_sdk/types.py (lines 469-504)
class McpStdioServerConfig(TypedDict):
    type: NotRequired[Literal["stdio"]]
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]

class McpSSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]

class McpHttpServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]
```

**Usage:**

```python
options = ClaudeAgentOptions(
    mcp_servers={
        "my-server": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "my_mcp_server"]
        }
    }
)
```

The SDK serializes these configs and passes them to the CLI via `--mcp-config`. From `subprocess_cli.py` (lines 240-265):

```python
if self._options.mcp_servers:
    if isinstance(self._options.mcp_servers, dict):
        servers_for_cli: dict[str, Any] = {}
        for name, config in self._options.mcp_servers.items():
            if isinstance(config, dict) and config.get("type") == "sdk":
                # For SDK servers, strip the Python instance field
                sdk_config = {k: v for k, v in config.items() if k != "instance"}
                servers_for_cli[name] = sdk_config
            else:
                servers_for_cli[name] = config

        if servers_for_cli:
            cmd.extend(["--mcp-config", json.dumps({"mcpServers": servers_for_cli})])
```

The Claude Code CLI manages the lifecycle of external MCP servers (starting, connecting, communicating). The SDK just passes the configuration.

---

## 3. SDK MCP Servers (In-Process)

This is the most interesting tool category. SDK MCP servers run **directly inside your Python application** — no subprocess overhead or IPC needed.

### Defining Tools with `@tool` Decorator

From `src/claude_agent_sdk/__init__.py` (lines 79-154):

```python
@dataclass
class SdkMcpTool(Generic[T]):
    """Definition for an SDK MCP tool."""
    name: str
    description: str
    input_schema: type[T] | dict[str, Any]
    handler: Callable[[T], Awaitable[dict[str, Any]]]
    annotations: ToolAnnotations | None = None

def tool(
    name: str,
    description: str,
    input_schema: type | dict[str, Any],
    annotations: ToolAnnotations | None = None,
) -> Callable[[Callable[[Any], Awaitable[dict[str, Any]]]], SdkMcpTool[Any]]:
    def decorator(handler):
        return SdkMcpTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            annotations=annotations,
        )
    return decorator
```

The `@tool` decorator wraps an async function into an `SdkMcpTool` dataclass.

### Creating an SDK MCP Server

From `src/claude_agent_sdk/__init__.py` (lines 157-319):

```python
def create_sdk_mcp_server(
    name: str, version: str = "1.0.0", tools: list[SdkMcpTool[Any]] | None = None
) -> McpSdkServerConfig:
    from mcp.server import Server
    from mcp.types import ImageContent, TextContent, Tool

    server = Server(name, version=version)

    if tools:
        tool_map = {tool_def.name: tool_def for tool_def in tools}

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            tool_list = []
            for tool_def in tools:
                # Convert input_schema to JSON Schema format
                if isinstance(tool_def.input_schema, dict):
                    if "type" in tool_def.input_schema and "properties" in tool_def.input_schema:
                        schema = tool_def.input_schema  # Already JSON schema
                    else:
                        # Simple dict → JSON schema conversion
                        properties = {}
                        for param_name, param_type in tool_def.input_schema.items():
                            if param_type is str:
                                properties[param_name] = {"type": "string"}
                            elif param_type is int:
                                properties[param_name] = {"type": "integer"}
                            elif param_type is float:
                                properties[param_name] = {"type": "number"}
                            elif param_type is bool:
                                properties[param_name] = {"type": "boolean"}
                            else:
                                properties[param_name] = {"type": "string"}
                        schema = {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties.keys()),
                        }
                tool_list.append(Tool(
                    name=tool_def.name,
                    description=tool_def.description,
                    inputSchema=schema,
                    annotations=tool_def.annotations,
                ))
            return tool_list

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
            tool_def = tool_map[name]
            result = await tool_def.handler(arguments)
            content = []
            if "content" in result:
                for item in result["content"]:
                    if item.get("type") == "text":
                        content.append(TextContent(type="text", text=item["text"]))
                    if item.get("type") == "image":
                        content.append(ImageContent(
                            type="image", data=item["data"], mimeType=item["mimeType"]
                        ))
            return content

    return McpSdkServerConfig(type="sdk", name=name, instance=server)
```

**Key detail:** This creates a real `mcp.server.Server` instance from the official Python MCP SDK library and registers `list_tools` and `call_tool` handlers on it. The server instance is stored in `McpSdkServerConfig` and passed to the `Query` class.

### Full Usage Example

From `examples/mcp_calculator.py`:

```python
@tool("add", "Add two numbers", {"a": float, "b": float})
async def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] + args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} + {args['b']} = {result}"}]}

calculator = create_sdk_mcp_server(
    name="calculator",
    version="2.0.0",
    tools=[add_numbers, subtract_numbers, multiply_numbers, ...]
)

options = ClaudeAgentOptions(
    mcp_servers={"calc": calculator},
    allowed_tools=[
        "mcp__calc__add",      # Format: mcp__<server_name>__<tool_name>
        "mcp__calc__subtract",
        "mcp__calc__multiply",
    ],
)
```

---

## How SDK MCP Tools Actually Work

This is the core of how in-process tools operate. The flow is:

### Step 1: CLI Discovers SDK Server

When the SDK sends the MCP config to the CLI, SDK servers have `"type": "sdk"`. The CLI knows these are handled by the SDK over the control protocol.

### Step 2: CLI Sends JSONRPC Requests via Control Protocol

When Claude wants to list or call tools on an SDK server, the CLI sends a `control_request` with `subtype: "mcp_message"` over stdout. The SDK receives this in its message reader.

### Step 3: SDK Routes to In-Process MCP Server

From `src/claude_agent_sdk/_internal/query.py` (lines 301-316):

```python
elif subtype == "mcp_message":
    server_name = request_data.get("server_name")
    mcp_message = request_data.get("message")

    mcp_response = await self._handle_sdk_mcp_request(server_name, mcp_message)
    response_data = {"mcp_response": mcp_response}
```

### Step 4: Manual JSONRPC Routing

The SDK **manually routes JSONRPC methods** because the Python MCP SDK doesn't expose a raw Transport abstraction (unlike the TypeScript SDK). From `query.py` (lines 391-527):

```python
async def _handle_sdk_mcp_request(self, server_name: str, message: dict) -> dict:
    server = self.sdk_mcp_servers[server_name]
    method = message.get("method")
    params = message.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": server.name, "version": server.version or "1.0.0"},
            },
        }

    elif method == "tools/list":
        request = ListToolsRequest(method=method)
        handler = server.request_handlers.get(ListToolsRequest)
        if handler:
            result = await handler(request)
            tools_data = []
            for tool in result.root.tools:
                tool_data = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema.model_dump()
                        if hasattr(tool.inputSchema, "model_dump")
                        else tool.inputSchema,
                }
                if tool.annotations:
                    tool_data["annotations"] = tool.annotations.model_dump(exclude_none=True)
                tools_data.append(tool_data)
            return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"tools": tools_data}}

    elif method == "tools/call":
        call_request = CallToolRequest(
            method=method,
            params=CallToolRequestParams(
                name=params.get("name"),
                arguments=params.get("arguments", {})
            ),
        )
        handler = server.request_handlers.get(CallToolRequest)
        if handler:
            result = await handler(call_request)
            content = []
            for item in result.root.content:
                if hasattr(item, "text"):
                    content.append({"type": "text", "text": item.text})
                elif hasattr(item, "data") and hasattr(item, "mimeType"):
                    content.append({"type": "image", "data": item.data, "mimeType": item.mimeType})
            return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"content": content}}
```

This is a **critical architectural point**: the SDK acts as a bridge between the CLI's JSONRPC messages and the `mcp.server.Server` instance's registered handlers. It manually constructs MCP request objects and invokes handlers directly.

---

## Authentication & Auth Handling

**The SDK itself does NOT handle authentication.** All auth is delegated to the Claude Code CLI.

### Environment-Based Auth

From `subprocess_cli.py` (lines 345-358):

```python
process_env = {
    **os.environ,                        # Inherits system environment (ANTHROPIC_API_KEY, etc.)
    **self._options.env,                 # User-provided environment variables
    "CLAUDE_CODE_ENTRYPOINT": "sdk-py",  # SDK identifier
    "CLAUDE_AGENT_SDK_VERSION": __version__,
}

if self._options.enable_file_checkpointing:
    process_env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"
```

Authentication works by inheriting environment variables — the Claude Code CLI reads `ANTHROPIC_API_KEY` (or similar) from the environment. You can also pass custom env vars:

```python
options = ClaudeAgentOptions(
    env={"ANTHROPIC_API_KEY": "sk-ant-..."}
)
```

### MCP Server Auth

For external MCP servers using SSE or HTTP, auth headers can be provided:

```python
options = ClaudeAgentOptions(
    mcp_servers={
        "my-api": {
            "type": "http",
            "url": "https://api.example.com/mcp",
            "headers": {"Authorization": "Bearer my-token"}
        }
    }
)
```

### User Identity

The `user` parameter in `ClaudeAgentOptions` is passed to the subprocess:

```python
self._process = await anyio.open_process(
    cmd, stdin=PIPE, stdout=PIPE, stderr=stderr_dest,
    cwd=self._cwd, env=process_env,
    user=self._options.user,  # OS-level user for the subprocess
)
```

---

## Tool Permission Callbacks

The SDK provides a `can_use_tool` callback that intercepts **every tool call** before execution.

### Type Definition

From `types.py` (lines 124-157):

```python
@dataclass
class ToolPermissionContext:
    signal: Any | None = None
    suggestions: list[PermissionUpdate] = field(default_factory=list)

@dataclass
class PermissionResultAllow:
    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None        # Modify the tool's input
    updated_permissions: list[PermissionUpdate] | None = None  # Update permission rules

@dataclass
class PermissionResultDeny:
    behavior: Literal["deny"] = "deny"
    message: str = ""
    interrupt: bool = False   # If True, stops the entire conversation

CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResult]
]
```

### How It Works

When the CLI needs tool permission, it sends a control request. From `query.py` (lines 242-283):

```python
if subtype == "can_use_tool":
    permission_request = request_data
    original_input = permission_request["input"]

    context = ToolPermissionContext(
        signal=None,
        suggestions=permission_request.get("permission_suggestions", []) or [],
    )

    response = await self.can_use_tool(
        permission_request["tool_name"],
        permission_request["input"],
        context,
    )

    if isinstance(response, PermissionResultAllow):
        response_data = {
            "behavior": "allow",
            "updatedInput": response.updated_input if response.updated_input else original_input,
        }
        if response.updated_permissions is not None:
            response_data["updatedPermissions"] = [
                p.to_dict() for p in response.updated_permissions
            ]
    elif isinstance(response, PermissionResultDeny):
        response_data = {"behavior": "deny", "message": response.message}
```

**Important:** `can_use_tool` requires streaming mode and automatically sets `permission_prompt_tool_name="stdio"`.

---

## Hook System

Hooks let you intercept and modify tool execution at various lifecycle points.

### Hook Events

From `types.py` (lines 161-172):

```python
HookEvent = (
    Literal["PreToolUse"]          # Before a tool runs
    | Literal["PostToolUse"]       # After a tool runs successfully
    | Literal["PostToolUseFailure"] # After a tool fails
    | Literal["UserPromptSubmit"]  # When user submits a prompt
    | Literal["Stop"]             # When conversation stops
    | Literal["SubagentStop"]     # When a sub-agent stops
    | Literal["PreCompact"]       # Before context compaction
    | Literal["Notification"]     # On notifications
    | Literal["SubagentStart"]    # When a sub-agent starts
    | Literal["PermissionRequest"] # When permission is needed
)
```

### Hook Configuration

Hooks are configured using matchers that filter by tool name:

```python
@dataclass
class HookMatcher:
    matcher: str | None = None        # Tool name filter (e.g., "Bash", "Write|Edit")
    hooks: list[HookCallback] = field(default_factory=list)
    timeout: float | None = None      # Seconds (default: 60)
```

### Hook Registration Flow

During initialization, hooks are registered with the CLI. From `query.py` (lines 119-163):

```python
async def initialize(self):
    hooks_config = {}
    if self.hooks:
        for event, matchers in self.hooks.items():
            if matchers:
                hooks_config[event] = []
                for matcher in matchers:
                    callback_ids = []
                    for callback in matcher.get("hooks", []):
                        callback_id = f"hook_{self.next_callback_id}"
                        self.next_callback_id += 1
                        self.hook_callbacks[callback_id] = callback  # Store callback by ID
                        callback_ids.append(callback_id)
                    hooks_config[event].append({
                        "matcher": matcher.get("matcher"),
                        "hookCallbackIds": callback_ids,
                    })

    response = await self._send_control_request({
        "subtype": "initialize",
        "hooks": hooks_config if hooks_config else None,
    })
```

The CLI stores the callback IDs and sends `hook_callback` control requests when hooks should fire. The SDK maps IDs back to Python functions:

```python
elif subtype == "hook_callback":
    callback_id = hook_callback_request["callback_id"]
    callback = self.hook_callbacks.get(callback_id)
    hook_output = await callback(
        request_data.get("input"),
        request_data.get("tool_use_id"),
        {"signal": None},
    )
    # Convert Python names (async_, continue_) to CLI names (async, continue)
    response_data = _convert_hook_output_for_cli(hook_output)
```

### Python Keyword Conversion

Since `async` and `continue` are Python keywords, the SDK uses `async_` and `continue_` in Python and converts them for the CLI:

```python
def _convert_hook_output_for_cli(hook_output: dict) -> dict:
    converted = {}
    for key, value in hook_output.items():
        if key == "async_":
            converted["async"] = value
        elif key == "continue_":
            converted["continue"] = value
        else:
            converted[key] = value
    return converted
```

---

## Custom Agents (Skills)

The SDK supports defining custom agents (referred to as "skills" in some contexts). These are **not** the same as MCP tools — they are full agent definitions with their own prompts and tool sets.

### Agent Definition

From `types.py` (lines 42-49):

```python
@dataclass
class AgentDefinition:
    description: str
    prompt: str
    tools: list[str] | None = None
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None
```

### How Agents Are Sent

Agents are sent via the **initialize control request** (not CLI flags) to avoid command-line size limits. From `client.py` (lines 150-156):

```python
agents_dict = None
if self.options.agents:
    agents_dict = {
        name: {k: v for k, v in asdict(agent_def).items() if v is not None}
        for name, agent_def in self.options.agents.items()
    }
```

Then from `query.py` (lines 150-155):

```python
request = {
    "subtype": "initialize",
    "hooks": hooks_config if hooks_config else None,
}
if self._agents:
    request["agents"] = self._agents
```

---

## Transport Layer

The transport layer abstracts the communication channel between the SDK and the Claude Code CLI.

### Abstract Interface

From `_internal/transport/__init__.py`:

```python
class Transport(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def write(self, data: str) -> None: ...

    @abstractmethod
    def read_messages(self) -> AsyncIterator[dict[str, Any]]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    async def end_input(self) -> None: ...
```

### SubprocessCLITransport

The default (and only built-in) transport spawns the `claude` CLI binary.

**CLI Discovery** — from `subprocess_cli.py` (lines 64-95):

```python
def _find_cli(self) -> str:
    # 1. Check for bundled CLI in _bundled/ directory
    bundled_cli = self._find_bundled_cli()
    if bundled_cli:
        return bundled_cli

    # 2. System PATH
    if cli := shutil.which("claude"):
        return cli

    # 3. Common install locations
    locations = [
        Path.home() / ".npm-global/bin/claude",
        Path("/usr/local/bin/claude"),
        Path.home() / ".local/bin/claude",
        Path.home() / "node_modules/.bin/claude",
        Path.home() / ".yarn/bin/claude",
        Path.home() / ".claude/local/claude",
    ]
```

**Command Building** — builds a CLI invocation like:
```
claude --output-format stream-json --verbose \
    --system-prompt "..." \
    --allowedTools "Read,Write,Bash" \
    --max-turns 10 \
    --permission-mode default \
    --mcp-config '{"mcpServers": {...}}' \
    --input-format stream-json
```

**Message Reading** — uses speculative JSON parsing to handle large messages that may be split across multiple reads. From `subprocess_cli.py` (lines 519-564):

```python
async def _read_messages_impl(self):
    json_buffer = ""
    async for line in self._stdout_stream:
        json_buffer += line.strip()

        if len(json_buffer) > self._max_buffer_size:
            raise SDKJSONDecodeError("Buffer size exceeded")

        try:
            data = json.loads(json_buffer)
            json_buffer = ""  # Successfully parsed — reset buffer
            yield data
        except json.JSONDecodeError:
            continue  # Incomplete JSON — keep buffering
```

**Write Lock** — prevents concurrent writes from corrupting the stream:

```python
async def write(self, data: str) -> None:
    async with self._write_lock:
        if not self._ready or not self._stdin_stream:
            raise CLIConnectionError("Not ready")
        await self._stdin_stream.send(data)
```

---

## Control Protocol

The control protocol enables **bidirectional communication** between the SDK and CLI over the same stdin/stdout streams used for messages.

### Message Types

| Direction | Type | Purpose |
|---|---|---|
| SDK → CLI | `control_request` | Initialize, interrupt, set permissions, MCP |
| CLI → SDK | `control_response` | Response to SDK-initiated requests |
| CLI → SDK | `control_request` | Tool permission, hook callbacks, MCP messages |
| SDK → CLI | `control_response` | Response to CLI-initiated requests |

### Request/Response Flow

From `query.py` (lines 344-389):

```python
async def _send_control_request(self, request: dict, timeout: float = 60.0) -> dict:
    # Generate unique ID
    self._request_counter += 1
    request_id = f"req_{self._request_counter}_{os.urandom(4).hex()}"

    # Create event for async response matching
    event = anyio.Event()
    self.pending_control_responses[request_id] = event

    # Send via transport
    control_request = {
        "type": "control_request",
        "request_id": request_id,
        "request": request,
    }
    await self.transport.write(json.dumps(control_request) + "\n")

    # Wait for matching response
    with anyio.fail_after(timeout):
        await event.wait()

    result = self.pending_control_results.pop(request_id)
    if isinstance(result, Exception):
        raise result
    return result.get("response", {})
```

### Message Routing in `_read_messages`

From `query.py` (lines 172-231):

```python
async def _read_messages(self):
    async for message in self.transport.read_messages():
        msg_type = message.get("type")

        if msg_type == "control_response":
            # Match to pending request by ID
            request_id = response.get("request_id")
            if request_id in self.pending_control_responses:
                self.pending_control_results[request_id] = response
                self.pending_control_responses[request_id].set()  # Wake up waiter
            continue

        elif msg_type == "control_request":
            # Handle incoming CLI requests (permissions, hooks, MCP)
            self._tg.start_soon(self._handle_control_request, request)
            continue

        # Regular messages go to the application stream
        await self._message_send.send(message)
```

---

## Message Parsing

The message parser converts raw JSON from the CLI into typed Python dataclasses.

From `_internal/message_parser.py`:

```python
def parse_message(data: dict) -> Message:
    message_type = data.get("type")

    match message_type:
        case "user":
            # Parse text, tool_use, and tool_result content blocks
            return UserMessage(content=..., uuid=..., parent_tool_use_id=...)

        case "assistant":
            # Parse text, thinking, tool_use, and tool_result blocks
            return AssistantMessage(content=..., model=..., parent_tool_use_id=...)

        case "system":
            return SystemMessage(subtype=..., data=...)

        case "result":
            return ResultMessage(
                subtype=..., duration_ms=..., duration_api_ms=...,
                is_error=..., num_turns=..., session_id=...,
                total_cost_usd=..., usage=..., structured_output=...
            )

        case "stream_event":
            return StreamEvent(uuid=..., session_id=..., event=...)
```

**Content block types:**

```python
TextBlock(text: str)
ThinkingBlock(thinking: str, signature: str)
ToolUseBlock(id: str, name: str, input: dict)
ToolResultBlock(tool_use_id: str, content: str | list | None, is_error: bool | None)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  Your Python App                     │
│                                                      │
│  ┌──────────┐   ┌─────────────┐  ┌───────────────┐  │
│  │  query()  │   │ClaudeSDK-   │  │ SDK MCP Server│  │
│  │          │   │  Client     │  │ (@tool funcs) │  │
│  └────┬─────┘   └──────┬──────┘  └───────┬───────┘  │
│       │                │                  │          │
│       └────────┬───────┘                  │          │
│                │                          │          │
│         ┌──────▼──────┐                   │          │
│         │    Query     │◄─────────────────┘          │
│         │  (Control    │  (routes MCP requests)      │
│         │  Protocol)   │                             │
│         └──────┬───────┘                             │
│                │                                     │
│         ┌──────▼──────┐                              │
│         │  Transport   │                              │
│         │ (Abstract)   │                              │
│         └──────┬───────┘                              │
│                │                                     │
│      ┌─────────▼─────────┐                            │
│      │SubprocessCLI-     │                            │
│      │  Transport        │                            │
│      │ (stdin/stdout)    │                            │
│      └─────────┬─────────┘                            │
└────────────────┼────────────────────────────────────┘
                 │ JSON over stdin/stdout
       ┌─────────▼─────────┐
       │  Claude Code CLI   │
       │  (claude binary)   │
       │                    │
       │  ┌──────────────┐  │
       │  │ Built-in     │  │
       │  │ Tools (Read, │  │
       │  │ Write, Bash) │  │
       │  └──────────────┘  │
       │  ┌──────────────┐  │
       │  │ External MCP │  │
       │  │ Servers      │  │
       │  │ (subprocess) │  │
       │  └──────────────┘  │
       └────────────────────┘
```

### Key Takeaways

1. **No direct API calls** — The SDK communicates exclusively with the Claude Code CLI subprocess.
2. **SDK MCP servers are bridges** — Tools defined with `@tool` are routed through a manual JSONRPC bridge in `Query._handle_sdk_mcp_request()`.
3. **Auth is delegated** — All authentication is handled by the Claude Code CLI via environment variables.
4. **Everything is streaming** — Internally, the SDK always uses `--input-format stream-json` for bidirectional control.
5. **Hooks use callback IDs** — Python functions are registered by ID during initialization, and the CLI invokes them by ID.
6. **Permission callbacks are interceptors** — `can_use_tool` intercepts every tool call and can allow, deny, or modify the input.
7. **Agents are sent via initialize** — Custom agent definitions (skills) bypass CLI argument limits by being sent over the control protocol.
