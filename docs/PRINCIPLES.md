# PRINCIPLES

> **Ally Framework**
>
> *Engineering Principles*
>
> These principles define the non-negotiable foundations of the Ally Framework.
>
> Every architectural decision should reinforce them.
>
> If a proposed feature conflicts with one or more principles, the feature should be redesigned.

---

# 1. Local First

The user's computer is the primary execution environment.

Ally should never require cloud infrastructure when a local implementation is technically viable.

Cloud services are optional enhancements.

Not requirements.

---

# 2. Privacy by Default

Personal data belongs to the user.

Not to Ally.

Not to applications.

Not to cloud providers.

By default:

* conversations remain local
* memories remain local
* documents remain local
* embeddings remain local
* models execute locally

Nothing should leave the device without explicit user consent.

---

# 3. Intelligence is Orchestrated

The Language Model is not the system.

It is one component of the system.

Planning, memory, scheduling, permissions, tools, retrieval and context are Runtime responsibilities.

The Runtime owns intelligence orchestration.

---

# 4. Separate Intelligence from Knowledge

Language Models should understand.

They should not memorize.

Knowledge belongs to:

* databases
* files
* APIs
* user memories
* documents
* plugins
* external systems

Whenever knowledge can be retrieved, it should not be embedded into model weights.

---

# 5. Deterministic Before AI

Never ask a Language Model to perform work that software can perform deterministically.

Examples:

Do not use an LLM to:

* calculate totals
* filter transactions
* sort data
* search databases
* execute workflows
* validate schemas
* apply business rules

Software should compute.

Language Models should reason.

---

# 6. Tool Before Knowledge

If a Tool can answer a question more accurately than the model, the Tool must be used.

Example:

Wrong:

"What is my account balance?"

→ Model guesses.

Correct:

→ Finance Tool retrieves balance.

→ Model explains the result.

---

# 7. Memory Before Context

Increasing context windows is not a memory strategy.

Instead:

Retrieve only the information relevant to the current task.

Large prompts are a symptom of poor memory architecture.

---

# 8. LLM Last

The Language Model should participate at the end of the execution pipeline whenever possible.

Preferred flow:

User

↓

Planner

↓

Memory

↓

Tools

↓

Knowledge

↓

LLM

↓

Response

The model should generate language.

Not orchestrate the system.

---

# 9. Model Agnostic

No module may depend on a specific model vendor.

Applications should not know whether Ally is running:

* Qwen
* Gemma
* SmolLM
* Phi
* DeepSeek
* PALM
* any future model

Replacing a model must never require application changes.

---

# 10. Backend Agnostic

Inference engines are interchangeable.

Supported implementations may include:

* llama.cpp
* Ollama
* ONNX Runtime
* MLX
* OpenAI-compatible APIs
* custom runtimes

The Runtime exposes one unified interface.

---

# 11. Hardware Agnostic

Ally should run everywhere.

Desktop.

Laptop.

Mini PC.

Raspberry Pi.

ARM.

x86.

Android.

iPhone.

Future NPUs.

Performance optimizations must never compromise portability.

---

# 12. Modular by Design

Everything that can be modularized should become a module.

Everything that can become a plugin should become a plugin.

The core Runtime remains intentionally small.

---

# 13. Plugins are First-Class Citizens

Features belong in Plugins.

Not in the Core.

Examples:

Finance

Calendar

Health

Travel

Shopping

Knowledge

Messaging

Applications compose the Runtime by choosing Plugins.

---

# 14. Explicit Permissions

Every capability requires explicit permission.

Examples:

Filesystem

Camera

Microphone

Internet

Location

Contacts

Notifications

Plugins must declare permissions before execution.

---

# 15. Event-Driven Communication

Modules should communicate through events whenever possible.

Avoid direct dependencies.

This improves:

* scalability
* observability
* testing
* maintainability

---

# 16. Storage Independence

The Runtime never depends on a database implementation.

Possible storage backends include:

* SQLite
* PostgreSQL
* RocksDB
* local files
* cloud synchronization
* custom implementations

Applications choose persistence.

Not Ally.

---

# 17. Explainability

Every important decision made by the Runtime should be inspectable.

Developers should be able to answer:

Why was this Tool selected?

Why was this memory retrieved?

Why was this context assembled?

Why was this Plugin executed?

Invisible intelligence becomes impossible to debug.

---

# 18. Predictability

Given the same:

* user state
* memories
* plugins
* permissions
* inputs

The Runtime should produce equivalent execution plans.

Randomness should belong only to language generation.

---

# 19. Minimal Core

The Core Runtime should remain intentionally small.

Every feature added to the Core increases long-term maintenance cost.

Whenever possible:

Move functionality into Plugins.

---

# 20. The Runtime is Permanent

Language Models will evolve.

Inference engines will evolve.

Hardware will evolve.

AI paradigms may change completely.

The Runtime should survive all of them.

Ally is built around abstractions.

Not implementations.

---

# 21. PALM is a Goal, Not a Dependency

The Personal Assistant Language Model (PALM) is a long-term objective.

The Runtime must never depend on PALM's existence.

Until PALM exists, Ally should work equally well with existing open models.

---

# 22. Open Ecosystem

The framework should encourage contributions.

Plugins.

Models.

Storage providers.

Inference backends.

SDKs.

Applications.

The ecosystem should grow without requiring changes to the Runtime.

---

# 23. Applications Build Experiences

Applications should focus exclusively on user experience.

They should never need to solve:

* memory management
* model selection
* context retrieval
* tool orchestration
* planning
* permissions
* inference

Those responsibilities belong to Ally.

---

# 24. The User is the Center

Not the Language Model.

Not the Framework.

Not the Application.

Everything exists to help one person accomplish their goals.

The assistant should continuously adapt to the user.

Never the opposite.

---

# 25. Build for the Next Decade

Every design decision should answer one question:

> "Will this architecture still make sense if Language Models change completely in ten years?"

If the answer is "no",

the abstraction is probably wrong.

---

# Closing Statement

Ally is not being built to create another chatbot.

It is being built to establish a new software architecture for Personal Intelligence.

Language Models are temporary.

Personal computing is permanent.

The Ally Framework is designed so that intelligence, privacy and personalization remain under the user's control, regardless of which models, hardware or AI paradigms emerge in the future.

These principles are the project's constitution.

Everything else is implementation.
