# ARCHITECTURE

> **Ally Framework**
>
> *Technical Architecture Specification*
>
> Version: Draft v0.1

---

# Overview

Ally is a **Personal Intelligence Runtime**.

Unlike traditional AI applications, Ally does **not** place the Language Model at the center of the system.

Instead, the LLM becomes one specialized component inside a larger cognitive architecture.

Every request flows through deterministic components before language generation.

---

# High-Level Architecture

```
                    User
                     │
                     ▼
               Application
                     │
                     ▼
                Ally SDK
                     │
                     ▼
              Ally Runtime
                     │
 ┌───────────────────┼────────────────────┐
 │                   │                    │
 ▼                   ▼                    ▼

Planner         Context Engine      Event Bus

 │                   │                    │

 ▼                   ▼                    ▼

Memory Engine   Knowledge Layer   Plugin Manager

        │             │

        └──────┬──────┘

               ▼

        Tool Orchestrator

               ▼

         Model Runtime

               ▼

      Language Model Backend

               ▼

      Natural Language Output
```

---

# Runtime Philosophy

The Runtime is responsible for intelligence orchestration.

The Language Model is responsible only for language understanding and generation.

Every other responsibility belongs to the Runtime.

---

# Core Modules

The Runtime is divided into independent modules.

Each module has a single responsibility.

```
runtime/

    planner/

    memory/

    context/

    tools/

    models/

    plugins/

    scheduler/

    events/

    storage/

    security/

    sdk/

    api/
```

Modules communicate only through public interfaces and events.

No module should depend directly on another implementation.

---

# Request Lifecycle

Every user interaction follows the same pipeline.

```
User

↓

Input

↓

Intent Recognition

↓

Planning

↓

Memory Retrieval

↓

Context Assembly

↓

Tool Selection

↓

Tool Execution

↓

Language Model

↓

Response

↓

Memory Update
```

This lifecycle is deterministic and observable.

---

# Planner

Purpose:

Transform user intentions into executable plans.

The Planner should answer questions like:

* What is the user trying to accomplish?
* Which tools are required?
* Is a response possible without the model?
* Is more information needed?

Example

User:

"I need to pay my credit card tomorrow."

Planner

```
Intent

finance.schedule_payment
```

Actions

```
Retrieve invoices

↓

Retrieve calendar

↓

Create reminder

↓

Generate confirmation
```

Planning should be deterministic whenever possible.

---

# Context Engine

The Context Engine decides what information reaches the Language Model.

The model never receives the complete conversation.

Instead, Ally builds a compact context package.

Example:

```
Recent Messages

Conversation Summary

Relevant Memories

Planner Results

Tool Results

Current User State
```

This dramatically reduces token usage while improving relevance.

---

# Memory Engine

Memory is a first-class component.

Not a prompt.

---

## Episodic Memory

Stores events.

Examples

* conversations
* purchases
* meetings
* reminders

---

## Semantic Memory

Stores facts.

Examples

* favorite bank
* preferred supermarket
* dietary restrictions
* work schedule

---

## Procedural Memory

Stores learned behaviors.

Examples

```
When discussing investments

↓

Always retrieve portfolio first.
```

Procedural memory allows continuous improvement without retraining.

---

# Knowledge Layer

Knowledge should not live inside model weights.

Sources may include:

* SQL databases
* Documents
* APIs
* Local files
* Internet
* Vector databases

The Knowledge Layer abstracts retrieval.

---

# Tool Orchestrator

Responsible for executing actions.

Every external interaction is a Tool.

Examples

```
Finance

Calendar

Filesystem

Weather

Maps

Email

Browser

Messaging

Camera

Microphone
```

The LLM never directly accesses these resources.

---

# Plugin Manager

Every capability is a Plugin.

```
plugins/

finance/

calendar/

health/

shopping/

travel/

knowledge/

home/
```

Plugins expose:

```
Capabilities

Permissions

Schemas

Tools

Events
```

Applications can install only the plugins they need.

---

# Event Bus

Every important action generates events.

Examples

```
ConversationStarted

ConversationEnded

ReminderCreated

TransactionImported

PluginInstalled

MemoryCreated

ToolExecuted
```

Modules communicate through events instead of direct dependencies.

This improves scalability and testing.

---

# Scheduler

Responsible for autonomous execution.

Examples

```
08:00

↓

Check calendar

↓

Prepare daily briefing

↓

Notify user
```

The Scheduler enables proactive assistants.

---

# Security Layer

Security is mandatory.

Every Tool declares:

```
Required Permissions

Read

Write

Network

Filesystem

Microphone

Camera
```

Applications decide which permissions to grant.

The Runtime enforces them.

---

# Storage Layer

Storage is abstracted.

Possible implementations:

SQLite

PostgreSQL

RocksDB

Cloud

Memory

Custom

The Runtime never depends on a specific database.

---

# Model Runtime

The Runtime abstracts inference engines.

Supported implementations:

```
llama.cpp

Ollama

ONNX Runtime

MLX

OpenAI Compatible

Anthropic Compatible

Custom Engines
```

Applications never interact directly with these engines.

---

# Language Model Interface

Every backend implements the same interface.

Conceptually:

```
Chat

Embedding

Summarization

Tool Calling

Completion

Token Counting

Streaming
```

Replacing the model should never affect the application.

---

# SDK

Applications interact exclusively with the SDK.

Example

```
Application

↓

SDK

↓

Runtime

↓

Modules
```

The SDK hides every implementation detail.

---

# API

The Runtime exposes a local API.

Possible transports:

HTTP

WebSocket

Unix Socket

Named Pipe

FFI

Applications choose whichever transport is most appropriate.

---

# Local-First Design

Every feature should work locally whenever possible.

Cloud services become optional enhancements.

Examples:

```
LLM

✓ Local

Memory

✓ Local

Calendar

✓ Local

Notes

✓ Local

Search

Optional

Cloud Sync

Optional
```

---

# Multi-Platform Strategy

The Runtime targets:

Windows

Linux

macOS

Android

iOS

ARM

x86

Embedded Devices

Raspberry Pi

Mini PCs

Future NPUs

Performance and portability have equal priority.

---

# PALM Integration

Future versions of Ally will support PALM.

```
PALM

↓

Model Runtime

↓

Same Interface

↓

Same SDK

↓

Same Applications
```

Applications should not know which Language Model is executing.

They simply interact with Ally.

---

# Example Application

```
Kyvo

↓

Ally SDK

↓

Planner

↓

Memory

↓

Finance Plugin

↓

Qwen 0.7B

↓

Response
```

Later:

```
Kyvo

↓

Ally SDK

↓

Planner

↓

Memory

↓

Finance Plugin

↓

PALM

↓

Response
```

No application code changes.

---

# Repository Structure

```
ally-framework/

    runtime/

    sdk/

    cli/

    plugins/

    examples/

    docs/

    benchmarks/

    models/

    tests/

    tools/
```

Future repositories

```
ally-framework
```

```
ally-models
```

```
ally-training
```

```
ally-benchmarks
```

---

# Design Goals

The architecture is guided by the following priorities:

1. Local-first execution.
2. Hardware independence.
3. Model independence.
4. Privacy by default.
5. Modular design.
6. Deterministic orchestration.
7. Extensible plugin ecosystem.
8. Efficient memory management.
9. Low resource consumption.
10. Long-term maintainability.

---

# Final Vision

The Ally Framework is not intended to become another LLM runtime.

Its goal is to become the **operating layer for Personal Intelligence**.

Applications should never worry about:

* which model is running;
* how memory is managed;
* how context is assembled;
* how tools are executed;
* how knowledge is retrieved.

Applications should simply ask:

> "Help the user."

Everything else is Ally's responsibility.

As Language Models evolve—or are replaced entirely—the architecture remains stable.

The Runtime becomes the permanent foundation.

Models become interchangeable components.

That is the future Ally is designed to support.
