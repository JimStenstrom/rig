# Rig Framework Architecture Map

## Overview

Rig is a Rust LLM orchestration framework organized as a Cargo workspace with 19 crates. The architecture follows a layered design with core abstractions, provider implementations, and companion integrations.

---

## Workspace Structure

```
rig/
├── rig-core/                    # Core library (main crate)
│   ├── src/
│   └── rig-core-derive/         # Procedural macros
├── Provider Companions/
│   ├── rig-bedrock/             # AWS Bedrock
│   ├── rig-eternalai/           # Eternal AI
│   ├── rig-fastembed/           # FastEmbed local embeddings
│   └── rig-vertexai/            # Google Vertex AI
├── Vector Store Companions/
│   ├── rig-lancedb/
│   ├── rig-mongodb/
│   ├── rig-neo4j/
│   ├── rig-postgres/
│   ├── rig-qdrant/
│   ├── rig-sqlite/
│   ├── rig-surrealdb/
│   ├── rig-milvus/
│   ├── rig-scylladb/
│   ├── rig-s3vectors/
│   └── rig-helixdb/
└── rig-wasm/                    # WebAssembly compatibility
```

---

## Core Module Map

```
rig-core/src/
│
├── lib.rs                       # Main exports & prelude
├── prelude.rs                   # Common imports
│
├── ══════════════════════════════════════════════════════════════
│   COMPLETION SYSTEM
│   ══════════════════════════════════════════════════════════════
│
├── completion/
│   ├── mod.rs                   # Module exports
│   ├── request.rs               # CompletionRequest, CompletionRequestBuilder
│   │                            # Traits: Prompt, Chat, Completion, CompletionModel
│   └── message.rs               # Message, UserContent, AssistantContent
│                                # Text, Image, Audio, Video, Document
│                                # ToolCall, ToolResult, Reasoning
│
├── streaming.rs                 # StreamingCompletionResponse
│                                # Traits: StreamingPrompt, StreamingChat
│                                # RawStreamingChoice, StreamedAssistantContent
│
├── ══════════════════════════════════════════════════════════════
│   AGENT SYSTEM
│   ══════════════════════════════════════════════════════════════
│
├── agent/
│   ├── mod.rs                   # Module exports
│   ├── builder.rs               # AgentBuilder, AgentBuilderSimple
│   ├── completion.rs            # Agent struct, trait implementations
│   ├── tool.rs                  # Agent tool integration
│   └── prompt_request/
│       ├── mod.rs               # PromptRequest, PromptResponse
│       └── streaming.rs         # StreamingPromptRequest, MultiTurnStreamItem
│
├── ══════════════════════════════════════════════════════════════
│   EMBEDDINGS SYSTEM
│   ══════════════════════════════════════════════════════════════
│
├── embeddings/
│   ├── mod.rs                   # Module exports
│   ├── embedding.rs             # EmbeddingModel trait, Embedding struct
│   ├── builder.rs               # EmbeddingsBuilder
│   ├── embed.rs                 # Embed trait, TextEmbedder
│   ├── tool.rs                  # ToolSchema for embedding tools
│   └── distance.rs              # Distance metrics (cosine, euclidean, etc.)
│
├── ══════════════════════════════════════════════════════════════
│   VECTOR STORE SYSTEM
│   ══════════════════════════════════════════════════════════════
│
├── vector_store/
│   ├── mod.rs                   # VectorStoreIndex, InsertDocuments traits
│   ├── in_memory_store.rs       # InMemoryVectorStore (default impl)
│   └── request.rs               # VectorSearchRequest, Filter traits
│
├── ══════════════════════════════════════════════════════════════
│   TOOL SYSTEM
│   ══════════════════════════════════════════════════════════════
│
├── tool/
│   ├── mod.rs                   # Tool trait, ToolEmbedding, ToolSet
│   │                            # ToolDyn, ToolSetBuilder
│   ├── server.rs                # ToolServer, ToolServerHandle
│   └── tools/                   # Built-in tool implementations
│
├── ══════════════════════════════════════════════════════════════
│   PIPELINE SYSTEM (DAG Workflows)
│   ══════════════════════════════════════════════════════════════
│
├── pipeline/
│   ├── mod.rs                   # PipelineBuilder, new(), with_error()
│   ├── op.rs                    # Op trait, Map, Then
│   ├── try_op.rs                # TryOp for error handling
│   ├── agent_ops.rs             # Lookup, Prompt, Extract operations
│   ├── parallel.rs              # parallel! macro
│   └── conditional.rs           # Conditional branching
│
├── ══════════════════════════════════════════════════════════════
│   CLIENT SYSTEM
│   ══════════════════════════════════════════════════════════════
│
├── client/
│   ├── mod.rs                   # Client<Ext, H>, ClientBuilder
│   ├── builder.rs               # DynClientBuilder
│   ├── completion.rs            # CompletionClient trait
│   ├── embeddings.rs            # EmbeddingsClient trait
│   ├── audio_generation.rs      # AudioGenerationClient
│   ├── image_generation.rs      # ImageGenerationClient
│   └── transcription.rs         # TranscriptionClient
│
├── ══════════════════════════════════════════════════════════════
│   PROVIDERS (18+ LLM Integrations)
│   ══════════════════════════════════════════════════════════════
│
├── providers/
│   ├── anthropic/               # Claude models
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   ├── completion.rs
│   │   └── streaming.rs
│   ├── openai/                  # GPT models
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   ├── completion/
│   │   ├── embedding.rs
│   │   ├── audio.rs
│   │   ├── transcription.rs
│   │   ├── image_generation.rs
│   │   └── responses_api/       # Responses API
│   ├── gemini/                  # Google Gemini
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   ├── completion.rs
│   │   ├── embedding.rs
│   │   └── streaming.rs
│   ├── cohere/                  # Cohere models
│   ├── mistral/                 # Mistral AI
│   ├── ollama/                  # Local Ollama
│   ├── azure.rs                 # Azure OpenAI
│   ├── deepseek.rs              # DeepSeek
│   ├── galadriel.rs             # Galadriel
│   ├── groq.rs                  # Groq
│   ├── huggingface/             # HuggingFace
│   ├── hyperbolic.rs            # Hyperbolic
│   ├── mira.rs                  # Mira
│   ├── moonshot.rs              # Moonshot
│   ├── openrouter/              # OpenRouter
│   ├── perplexity.rs            # Perplexity
│   ├── together.rs              # Together AI
│   ├── voyage_ai.rs             # Voyage AI
│   └── xai/                     # xAI (Grok)
│
├── ══════════════════════════════════════════════════════════════
│   SUPPORTING MODULES
│   ══════════════════════════════════════════════════════════════
│
├── extractor.rs                 # Extractor, ExtractorBuilder
│                                # Structured data extraction from LLM
│
├── loaders/                     # Document loading
│   ├── mod.rs
│   ├── file.rs                  # FileLoader
│   ├── pdf.rs                   # PdfFileLoader (feature: pdf)
│   └── epub/                    # EpubFileLoader (feature: epub)
│
├── http_client/                 # HTTP infrastructure
│   ├── mod.rs                   # HttpClientExt trait
│   ├── sse.rs                   # Server-Sent Events
│   └── retry.rs                 # Retry logic
│
├── telemetry/                   # OpenTelemetry integration
│   └── mod.rs                   # SpanCombinator, ProviderResponseExt
│
├── integrations/                # Pre-built integrations
│   ├── cli_chatbot.rs           # CLI chatbot helper
│   └── discord_bot.rs           # Discord bot (feature: discord-bot)
│
├── ══════════════════════════════════════════════════════════════
│   UTILITY MODULES
│   ══════════════════════════════════════════════════════════════
│
├── one_or_many.rs               # OneOrMany<T> container type
├── json_utils.rs                # JSON merge utilities
├── wasm_compat.rs               # WASM compatibility layer
├── evals.rs                     # Evaluation framework (experimental)
├── transcription.rs             # Audio transcription
├── image_generation.rs          # Image generation
└── audio_generation.rs          # Audio generation
```

---

## Derive Macros

```
rig-core-derive/src/
├── lib.rs                       # Macro exports
├── embed.rs                     # #[derive(Embed)] - Auto-implement Embed trait
├── client.rs                    # #[derive(ProviderClient)]
├── basic.rs                     # Basic macro support
└── custom.rs                    # Custom derive implementations
```

---

## Core Trait Hierarchy

```
                    ┌─────────────────────────────────────────┐
                    │           CompletionModel               │
                    │  (Base trait for all LLM providers)     │
                    ├─────────────────────────────────────────┤
                    │  + completion(request) → Response       │
                    │  + stream(request) → StreamingResponse  │
                    │  + completion_request(prompt) → Builder │
                    └────────────────────┬────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
     ┌────────────────┐        ┌────────────────┐        ┌────────────────┐
     │     Prompt     │        │      Chat      │        │   Completion   │
     │  (One-shot)    │        │  (w/ History)  │        │   (Low-level)  │
     ├────────────────┤        ├────────────────┤        ├────────────────┤
     │ prompt(msg)    │        │ chat(msg, hist)│        │ completion()   │
     │  → String      │        │  → String      │        │  → Builder     │
     └────────────────┘        └────────────────┘        └────────────────┘
              │                          │                          │
              └──────────────────────────┴──────────────────────────┘
                                         │
                                         ▼
                              ┌────────────────────┐
                              │       Agent        │
                              │  (Implements all)  │
                              ├────────────────────┤
                              │ + model            │
                              │ + preamble         │
                              │ + static_context   │
                              │ + dynamic_context  │
                              │ + tools            │
                              └────────────────────┘
```

---

## Message Type Hierarchy

```
Message
├── User { content: OneOrMany<UserContent> }
│   └── UserContent
│       ├── Text(Text)
│       ├── ToolResult(ToolResult)
│       ├── Image(Image)
│       ├── Audio(Audio)
│       ├── Video(Video)
│       └── Document(Document)
│
└── Assistant { id, content: OneOrMany<AssistantContent> }
    └── AssistantContent
        ├── Text(Text)
        ├── ToolCall(ToolCall)
        ├── Reasoning(Reasoning)
        └── Image(Image)

Supporting Types:
├── Text { text: String }
├── Image { data, media_type, detail }
├── Audio { data, media_type }
├── Video { data, media_type }
├── Document { data, media_type }
├── ToolCall { id, call_id, function: ToolFunction }
├── ToolResult { id, call_id, content }
└── Reasoning { id, reasoning: Vec<String>, signature }
```

---

## Tool System

```
┌─────────────────────────────────────────────────────────────┐
│                        Tool Trait                           │
├─────────────────────────────────────────────────────────────┤
│  const NAME: &'static str                                   │
│  type Error, Args, Output                                   │
│  fn definition(prompt) → ToolDefinition                     │
│  fn call(args) → Result<Output, Error>                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│    ToolEmbedding    │         │      ToolDyn        │
│  (RAG-capable)      │         │  (Dynamic dispatch) │
├─────────────────────┤         ├─────────────────────┤
│ + embedding_docs()  │         │  Wraps any Tool     │
│ + context()         │         │  for runtime use    │
│ + init(state, ctx)  │         │                     │
└─────────────────────┘         └─────────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │      ToolSet        │
              ├─────────────────────┤
              │  Collection of tools│
              │  + add_tool()       │
              │  + call(name, args) │
              │  + definitions()    │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │    ToolServer       │
              ├─────────────────────┤
              │  Runtime executor   │
              │  + static tools     │
              │  + dynamic tools    │
              │  + MCP integration  │
              └─────────────────────┘
```

---

## Embedding & Vector Store System

```
┌─────────────────────────────────────────────────────────────┐
│                    EmbeddingModel                           │
├─────────────────────────────────────────────────────────────┤
│  const MAX_DOCUMENTS: usize                                 │
│  fn ndims() → usize                                         │
│  fn embed_texts(texts) → Vec<Embedding>                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  EmbeddingsBuilder  │
              ├─────────────────────┤
              │  + simple_document()│
              │  + document()       │
              │  + build() → Vec    │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │      Embedding      │
              ├─────────────────────┤
              │  + document: String │
              │  + vec: Vec<f64>    │
              └─────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   VectorStoreIndex                          │
├─────────────────────────────────────────────────────────────┤
│  fn top_n(request) → Vec<(score, id, T)>                    │
│  fn top_n_ids(request) → Vec<(score, id)>                   │
└─────────────────────────────────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌───────────┐      ┌───────────┐      ┌───────────┐
│InMemory   │      │ MongoDB   │      │  Qdrant   │
│VectorStore│      │   Index   │      │   Index   │
└───────────┘      └───────────┘      └───────────┘
    ... and 9 more companion crates
```

---

## Pipeline System (DAG Workflows)

```
┌─────────────────────────────────────────────────────────────┐
│                      PipelineBuilder                        │
├─────────────────────────────────────────────────────────────┤
│  new() → Pipeline                                           │
│  with_error() → Pipeline (error handling)                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │     Op<In, Out>     │
              ├─────────────────────┤
              │  .map(fn)           │
              │  .then(async fn)    │
              │  .chain(op)         │
              │  .lookup(index, n)  │
              │  .prompt(agent)     │
              │  .extract(extractor)│
              └─────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│    TryOp<I,O,E>     │         │    parallel!()      │
│  (Error handling)   │         │  (Concurrent ops)   │
└─────────────────────┘         └─────────────────────┘

Example Pipeline:
  pipeline::new()
    .chain(lookup(index, 3))
    .chain(prompt(agent))
    .chain(extract(extractor))
```

---

## Provider Architecture Pattern

Each provider follows this structure:

```
providers/{name}/
├── mod.rs              # Re-exports, Client struct
├── client.rs           # HTTP client configuration
├── completion.rs       # CompletionModel implementation
│   ├── CompletionModel struct
│   ├── impl CompletionModel trait
│   ├── Request/Response types
│   └── Message conversions (TryFrom)
├── embedding.rs        # EmbeddingModel (if supported)
├── streaming.rs        # Streaming implementation
└── audio|image|etc.rs  # Additional capabilities
```

**Provider Trait Implementation:**
```rust
impl CompletionModel for ProviderModel {
    type Response = ProviderResponse;
    type StreamingResponse = ProviderStreamChunk;
    type Client = ProviderClient;

    fn make(client, model) → Self
    fn completion(request) → CompletionResponse
    fn stream(request) → StreamingCompletionResponse
}
```

---

## Data Flow

```
User Prompt
     │
     ▼
┌─────────────────┐
│     Agent       │
│  ┌───────────┐  │
│  │ Preamble  │  │
│  │ Context   │  │
│  │ Tools     │  │
│  └───────────┘  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│CompletionRequest│────▶│  VectorStore    │
│  Builder        │     │  (RAG lookup)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │◀──────────────────────┘
         ▼
┌─────────────────┐
│ CompletionModel │
│   (Provider)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HTTP Request    │
│ (to LLM API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│CompletionResponse│
│  ┌───────────┐  │
│  │ choice    │  │
│  │ usage     │  │
│  │ raw_resp  │  │
│  └───────────┘  │
└────────┬────────┘
         │
         ▼
   ┌─────┴─────┐
   │           │
   ▼           ▼
Text      ToolCall
   │           │
   │           ▼
   │    ┌───────────┐
   │    │ToolServer │
   │    │  execute  │
   │    └─────┬─────┘
   │          │
   │          ▼
   │    ToolResult
   │          │
   └────┬─────┘
        │
        ▼
  Final Response
```

---

## Feature Flags

```toml
[features]
default = ["reqwest-tls"]           # Default HTTP client
all = ["derive", "pdf", "rayon"]    # All optional features

# Capabilities
derive = ["rig-derive"]             # Derive macros
pdf = ["lopdf"]                     # PDF document loading
epub = ["epub", "quick-xml"]        # EPUB loading
rayon = ["dep:rayon"]               # Parallel processing

# Integrations
rmcp = ["dep:rmcp"]                 # MCP tool protocol
discord-bot = ["serenity"]          # Discord integration

# HTTP Clients
reqwest-tls = ["reqwest/default-tls"]
reqwest-native-tls = ["reqwest/native-tls"]
```

---

## External Dependencies

**Core:**
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `serde` / `serde_json` - Serialization
- `thiserror` - Error types
- `tracing` - Distributed tracing
- `futures` - Async utilities

**Provider SDKs (companions):**
- `mongodb` - MongoDB driver
- `sqlx` - PostgreSQL/SQLite
- `qdrant-client` - Qdrant
- `neo4rs` - Neo4j
- `aws-sdk-*` - AWS Bedrock
- `lancedb` - LanceDB

---

## Summary Statistics

| Component | Count |
|-----------|-------|
| Total Crates | 19 |
| Core Modules | ~25 |
| LLM Providers | 18+ |
| Vector Stores | 11 |
| Traits | ~15 major |
| Lines of Code | ~25,000 (core) |
