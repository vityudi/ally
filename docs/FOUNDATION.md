# FOUNDATION

> **Ally Framework**
>
> *An Open Personal Intelligence Framework*

---

# Vision

Computers have become incredibly powerful, yet personal assistants remain largely dependent on cloud services, proprietary APIs, and increasingly larger language models.

Today's AI ecosystem is moving toward one direction: **more parameters, more hardware, more infrastructure, and more cost.**

We believe there is another path.

A personal assistant does **not** need to know everything about the world.

It does **not** need to solve university-level mathematics.

It does **not** need to write entire software systems.

It does **not** need to memorize every historical event ever recorded.

A personal assistant only needs to understand **one person**.

That changes everything.

---

# The Problem

Current LLMs are designed to be general intelligence systems.

They are trained to answer virtually any question imaginable.

As a consequence they become:

* enormous
* expensive
* hardware demanding
* cloud dependent
* difficult to run privately
* difficult to customize
* impossible to deeply personalize

For a Personal Assistant, this is unnecessary.

When a user asks:

> "How much did I spend this month?"

The model should not already know the answer.

It should know **how to obtain the answer**.

The intelligence should remain inside the model.

The knowledge should remain inside the user's system.

---

# Our Philosophy

The Ally Framework is built around one fundamental principle:

> **Separate Intelligence from Knowledge.**

Language understanding belongs to the Language Model.

Everything else belongs to the Runtime.

Knowledge should never be hardcoded inside model weights if it can be retrieved from trusted sources.

The model becomes an orchestrator rather than a database.

---

# What is Ally?

Ally is **not** an LLM.

Ally is **not** a chatbot.

Ally is **not** another AI wrapper.

Ally is a **Personal Intelligence Runtime**.

Its responsibility is to coordinate every component required for a local-first personal assistant.

Applications interact with Ally.

Ally interacts with models, memory, tools, storage, planners and external services.

The Language Model becomes only one component inside a much larger cognitive architecture.

---

# Core Principles

## Local First

Every decision starts locally.

Cloud providers should be optional.

A user should be able to download Ally, open it, and have a fully functional assistant without creating online accounts.

Privacy is the default.

---

## Model Agnostic

Ally must never depend on a specific model vendor.

Compatible examples include:

* Qwen 0.7B
* Gemma
* SmolLM
* Phi
* TinyLlama
* DeepSeek
* Ollama
* llama.cpp
* ONNX Runtime
* future PALM models

Changing the model should never require changing the application.

---

## Hardware Agnostic

The framework is designed to execute on:

* Windows
* Linux
* macOS
* Android
* iOS
* ARM
* x86
* Raspberry Pi
* Mini PCs
* Embedded devices

GPU acceleration is optional.

CPU execution is a first-class citizen.

NPUs should be used whenever available.

---

## Personal Intelligence

The objective is not general intelligence.

The objective is understanding one user better every day.

---

# Intelligence Architecture

Traditional applications usually work like this:

User

↓

LLM

↓

Answer

Ally introduces a different architecture:

User

↓

Intent Recognition

↓

Planning

↓

Memory Retrieval

↓

Tool Selection

↓

Tool Execution

↓

Context Assembly

↓

Language Model

↓

Natural Response

The Language Model is intentionally moved to the end of the pipeline.

Its role is language generation.

Not system orchestration.

---

# Runtime Architecture

```
Application
      │
      ▼
Ally Framework
│
├── Context Engine
├── Memory Engine
├── Planner
├── Tool Engine
├── Model Runtime
├── Knowledge Layer
├── Plugin System
├── Scheduler
└── Storage Layer
```

Each module has a single responsibility.

---

# Context Engine

Responsible for determining what information should be visible to the model.

Instead of sending thousands of previous messages, Ally builds a compact contextual package containing:

* conversation summary
* recent messages
* relevant memories
* tool outputs
* user preferences

The model receives only what matters.

---

# Memory Engine

Memory is divided into multiple layers.

## Episodic Memory

Past events.

Examples:

* meetings
* conversations
* purchases

---

## Semantic Memory

Persistent knowledge.

Examples:

* favorite bank
* preferred language
* dietary preferences

---

## Procedural Memory

Behavior.

Examples:

"When discussing finances, always retrieve transactions before answering."

This allows Ally to continuously improve without retraining the model.

---

# Planner

The Planner transforms intentions into executable actions.

Example:

User:

"I need to pay my bills tomorrow morning."

Planner:

* create reminder
* identify pending bills
* update calendar
* prepare notification

Planning becomes deterministic whenever possible.

---

# Tool Engine

The Tool Engine is responsible for interacting with the outside world.

Examples:

* Finance
* Calendar
* Files
* Contacts
* Email
* Maps
* Weather
* Notes
* Internet Search
* Smart Home
* Messaging

The model never directly accesses these resources.

Everything passes through permission-aware tools.

---

# Model Runtime

The Runtime abstracts every supported inference backend.

Supported backends may include:

* llama.cpp
* Ollama
* ONNX Runtime
* MLX
* OpenAI-compatible APIs
* Local inference engines

Applications never communicate directly with these implementations.

They communicate with Ally.

---

# Plugin System

Every capability should be installable.

Examples:

Finance Plugin

Health Plugin

Travel Plugin

Shopping Plugin

Home Automation Plugin

Knowledge Base Plugin

Applications may ship with their own plugins.

Third parties may build new ones.

---

# Storage Layer

Ally does not dictate storage technology.

Possible implementations:

* SQLite
* PostgreSQL
* pgvector
* Local files
* Cloud synchronization

Applications choose the persistence strategy.

---

# PALM

One long-term objective of this project is the creation of:

**PALM — Personal Assistant Language Model**

PALM does not exist today.

It is a future initiative.

Unlike modern LLMs, PALM will be intentionally specialized.

Its objective is not to become a general-purpose model.

Instead it will focus on:

* natural conversation
* planning
* long-term context
* tool usage
* personal organization
* financial reasoning
* memory utilization

Programming, advanced mathematics, scientific reasoning and broad world knowledge are intentionally outside its primary scope.

Knowledge should remain inside Ally.

Not inside the model weights.

Until PALM exists, Ally remains fully compatible with existing open models.

---

# Application Architecture

Applications should never depend directly on an LLM.

Instead they depend on Ally.

```
Application

↓

Ally SDK

↓

Ally Runtime

↓

Model

↓

Tools

↓

Storage
```

Changing the model should require zero application changes.

Changing the Runtime should require zero application changes.

---

# Why Rust?

The Runtime is planned to be implemented primarily in Rust.

Reasons include:

* native performance
* low memory usage
* zero-cost abstractions
* excellent ARM support
* WebAssembly compatibility
* mobile compatibility
* embedded compatibility
* small executable size
* safe concurrency

Performance is important.

Portability is essential.

---

# Long-Term Roadmap

## Phase 1

Runtime foundation.

Core architecture.

Plugin system.

---

## Phase 2

Memory Engine.

Planner.

Context Engine.

---

## Phase 3

Native local inference.

Multiple backend support.

Hardware optimization.

---

## Phase 4

Developer SDK.

Documentation.

Community plugins.

---

## Phase 5

Training pipeline.

Synthetic datasets.

Evaluation benchmarks.

---

## Phase 6

PALM.

A lightweight language model built specifically for personal assistants.

---

# Mission

Build the world's best local-first Personal Intelligence Runtime.

Not the largest model.

Not the smartest chatbot.

Not another AI wrapper.

A framework that empowers applications to provide intelligent, private and efficient personal assistants capable of running on virtually any device.

The future of personal AI is not bigger models.

It is better systems.

Ally exists to build those systems.
