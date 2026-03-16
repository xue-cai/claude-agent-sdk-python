# User-Facing Protocol — What's Between Your Agent and End Users?

## Table of Contents

1. [Question](#question)
2. [Short Answer](#short-answer)
3. [What the SDK Handles (Internal Plumbing)](#what-the-sdk-handles-internal-plumbing)
4. [What You Need to Build Yourself (User-Facing Layer)](#what-you-need-to-build-yourself-user-facing-layer)
5. [Architecture Diagram](#architecture-diagram)
6. [Protocol Options](#protocol-options)
7. [Conclusion](#conclusion)

---

## Question

> Does agents built with the Claude Agent SDK use a standard protocol between users and clients? For example, I use the SDK to build a travel agent that can research travel plans, book flights and hotels, and create Google Calendar events. I then host this agent to run in the cloud and expose it as a webapp to other users. Other users can use it for their own travels. What will the protocol between the agent and the users using it?

---

## Short Answer

The Claude Agent SDK does **not** define a standard protocol between your agent and end users. It is a **backend/server-side library** that manages the communication between **your application code** and the **Claude Code CLI** (via a subprocess stdin/stdout transport). It is not designed to be a user-facing protocol or API framework. The protocol between your agent and its end users is entirely up to you to define.

---

## What the SDK Handles (Internal Plumbing)

- **Your app ↔ Claude Code CLI**: The SDK uses an internal JSON-over-stdin/stdout protocol (`SubprocessCLITransport` in `src/claude_agent_sdk/_internal/transport/subprocess_cli.py`) to communicate with the Claude Code CLI subprocess. Messages are JSON objects with fields like `type`, `message`, `session_id`, etc. This is an **internal implementation detail**, not a user-facing protocol.
- **Claude Code CLI ↔ Claude API**: The CLI handles authentication, API calls, tool orchestration, and the agentic loop with Anthropic's API.

### Key code references

| Component | File | Role |
|---|---|---|
| `SubprocessCLITransport` | `src/claude_agent_sdk/_internal/transport/subprocess_cli.py` | Spawns CLI subprocess, reads/writes JSON over stdin/stdout |
| `Query` | `src/claude_agent_sdk/_internal/query.py` | Handles the control protocol (initialize, permissions, hooks, MCP) |
| `ClaudeSDKClient` | `src/claude_agent_sdk/client.py` | High-level bidirectional client for interactive sessions |
| `query()` | `src/claude_agent_sdk/query.py` | One-shot query function for stateless interactions |

---

## What You Need to Build Yourself (User-Facing Layer)

The SDK provides no built-in web server, HTTP API, WebSocket interface, or any standard protocol for end users to interact with your agent. **You** are responsible for:

1. **Choosing a protocol** between your webapp and end users (see [Protocol Options](#protocol-options) below).

2. **Building the web layer** — a web server (e.g., FastAPI, Flask, Django) that:
   - Accepts user requests via your chosen protocol
   - Translates them into SDK calls (`query()` or `ClaudeSDKClient`)
   - Streams or returns the agent's responses back to the user

3. **Managing user sessions** — mapping each end user's conversation to an appropriate SDK client instance or session.

---

## Architecture Diagram

For the travel agent example, the architecture would look like this:

```
┌──────────────┐     HTTP/WS/SSE      ┌──────────────────┐     SDK internal     ┌──────────────┐
│  End Users   │ ◄──── (you define ────►│  Your Web App    │ ◄── JSON protocol ──►│  Claude Code  │
│  (browsers)  │       this protocol)  │  (FastAPI etc.)  │     (subprocess)     │  CLI + API   │
└──────────────┘                       └──────────────────┘                      └──────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │  MCP Servers  │ (flights API,
                                       │  (tools)      │  hotels API,
                                       │               │  Google Calendar)
                                       └──────────────┘
```

### Data flow

1. **End user** sends a request (e.g., "Find flights from NYC to Tokyo in April") via your web app's API.
2. **Your web app** receives the request and calls the SDK (`query()` or `ClaudeSDKClient.query()`).
3. **The SDK** communicates with the Claude Code CLI subprocess over stdin/stdout.
4. **Claude** reasons about the request and invokes tools (MCP servers for flights, hotels, calendar) as needed.
5. **Tool results** flow back through the CLI to the SDK to your web app.
6. **Your web app** formats and returns the response to the end user via your chosen protocol.

---

## Protocol Options

When choosing a protocol for the user-facing layer, consider these options:

| Protocol | Best For | Streaming? | Complexity |
|---|---|---|---|
| **REST API (HTTP JSON)** | Simple request/response interactions | No (or polling) | Low |
| **WebSocket** | Real-time bidirectional communication | Yes | Medium |
| **Server-Sent Events (SSE)** | Server-to-client streaming | Yes (one-way) | Low-Medium |
| **Model Context Protocol (MCP)** | Exposing your agent as a tool server for other agents | Yes | Medium |
| **Agent-to-Agent (A2A)** | Interoperability with other agent frameworks | Yes | Higher |

### Recommendations

- **For a webapp with streaming responses**: Use **SSE** or **WebSocket**. SSE is simpler if you only need server-to-client streaming (user sends a message, agent streams back). WebSocket is better if users need to interrupt or send follow-up messages mid-stream.
- **For a simple API**: Use **REST** with JSON payloads if you don't need streaming.
- **For agent interoperability**: Consider **MCP** (which the SDK already supports for tool integration) or the emerging **A2A protocol** for agent-to-agent communication.

---

## Conclusion

The Claude Agent SDK is a **programmatic interface for driving Claude** from your backend code. It handles the complex internal protocol between your code and the Claude Code CLI, but the protocol between your agent and its end users is entirely your responsibility to design and implement. This gives you full flexibility to choose the right protocol for your use case — whether that's a REST API, WebSocket, SSE, or a standardized agent protocol like MCP or A2A.
