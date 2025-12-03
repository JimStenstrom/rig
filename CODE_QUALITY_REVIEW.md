# Rig Code Quality & Architecture Review

**Review Date:** December 2025
**Reviewer:** Claude (Opus 4)
**Codebase:** rig (Rust LLM Orchestration Framework)

---

## Architecture Assessment

Rig is a well-designed LLM orchestration framework with strong architectural foundations. The trait-based polymorphism pattern is excellent and allows seamless integration of 18+ providers. The core abstractions (`CompletionModel`, `EmbeddingModel`, `VectorStoreIndex`, `Tool`) are well-defined and compose cleanly.

**Main concerns:** Provider implementations contain significant code duplication, builder patterns are repeated rather than abstracted, and some files have grown beyond maintainable sizes (2000+ LOC). The `OneOrMany<T>` type adds complexity where standard Rust idioms (`Vec<T>` with validation) would suffice.

---

## Structural Issues

### Issue: God Modules in Providers

- **Location**: Multiple provider files
- **Smell**: God Module (1000+ lines per file)
- **Files affected**:
  - `providers/gemini/completion.rs` (2,169 LOC)
  - `providers/openai/responses_api/mod.rs` (1,512 LOC)
  - `providers/ollama.rs` (1,382 LOC)
  - `providers/anthropic/completion.rs` (1,239 LOC)
  - `providers/huggingface/completion.rs` (1,171 LOC)
  - `providers/azure.rs` (1,062 LOC)

- **Impact**: Hard to navigate, multiple responsibilities per file, difficult to test in isolation
- **Refactoring**:
  1. Split each provider into submodules:
     ```
     providers/gemini/
     ├── mod.rs           # Re-exports
     ├── client.rs        # Client struct
     ├── completion.rs    # CompletionModel impl (~200 LOC)
     ├── streaming.rs     # Streaming logic
     ├── types.rs         # Request/Response types
     └── conversions.rs   # Message type conversions
     ```
  2. Extract shared API types into a `types` submodule
  3. Move tests to separate test modules

### Issue: AgentBuilder Duplication

- **Location**: `agent/builder.rs`
- **Smell**: Parallel Structures / Copy-Paste
- **Impact**: Two nearly identical builder structs (`AgentBuilder` and `AgentBuilderSimple`) with duplicated methods

**Before (current state):**
```rust
// 16 nearly identical methods duplicated between AgentBuilder and AgentBuilderSimple:
// name(), description(), preamble(), without_preamble(), append_preamble(),
// context(), temperature(), max_tokens(), additional_params(), build(), etc.
```

**Refactoring:**
```rust
// Use a shared base with generic state machine pattern:
pub struct AgentBuilder<M, Tools = NoTools> {
    inner: AgentBuilderCore<M>,
    tools: Tools,
}

struct AgentBuilderCore<M> {
    name: Option<String>,
    model: M,
    preamble: Option<String>,
    // ... shared fields
}

impl<M, T> AgentBuilder<M, T> {
    // All shared methods implemented once
    pub fn name(mut self, name: &str) -> Self { ... }
    pub fn preamble(mut self, preamble: &str) -> Self { ... }
}
```

### Issue: Message Type Conversion Sprawl

- **Location**: Each provider's `completion.rs`
- **Smell**: Shotgun Surgery
- **Impact**: Adding a new content type (e.g., video) requires modifying 15+ files with similar `TryFrom` implementations

**Example of repetition across providers:**
```rust
// In anthropic/completion.rs, gemini/completion.rs, openai/completion.rs, etc:
impl TryFrom<message::UserContent> for ProviderContent {
    // Nearly identical match arms repeated 15+ times
    match content {
        message::UserContent::Text(Text { text }) => ...,
        message::UserContent::Image(image) => ...,
        message::UserContent::ToolResult(result) => ...,
        // etc.
    }
}
```

**Refactoring:**
Create a conversion trait with default implementations:
```rust
pub trait MessageConverter: Sized {
    type ProviderMessage;
    type Error;

    fn convert_text(text: &Text) -> Result<Self, Self::Error>;
    fn convert_image(image: &Image) -> Result<Self, Self::Error>;
    fn convert_tool_result(result: &ToolResult) -> Result<Self, Self::Error>;

    // Default implementation that calls the above
    fn from_user_content(content: UserContent) -> Result<Self, Self::Error> {
        match content {
            UserContent::Text(t) => Self::convert_text(&t),
            UserContent::Image(i) => Self::convert_image(&i),
            // ...
        }
    }
}
```

---

## DRY Violations

### Duplication: CompletionModel Implementations

- **Locations**: 20 files implementing `CompletionModel`
- **Shared Concept**: HTTP request/response cycle with tracing, error handling, JSON serialization
- **Pattern that repeats in every provider**:

```rust
impl CompletionModel for ProviderModel {
    async fn completion(&self, request: CompletionRequest) -> Result<...> {
        // 1. Create tracing span (duplicated)
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                // ... same fields in every provider
            )
        } else {
            tracing::Span::current()
        };

        // 2. Convert request (provider-specific)
        let request = ProviderRequest::try_from(request)?;

        // 3. Serialize and send HTTP (duplicated pattern)
        let body = serde_json::to_vec(&request)?;
        let response = self.client.post(...).body(body).send().await?;

        // 4. Handle response (duplicated error handling)
        if response.status().is_success() {
            let response: ProviderResponse = serde_json::from_slice(...)?;
            response.try_into()
        } else {
            Err(CompletionError::ProviderError(...))
        }
    }
}
```

**Unification Strategy:**
```rust
// Create a macro or helper trait:
pub trait CompletionModelHelper: Clone {
    type Request: TryFrom<CompletionRequest, Error = CompletionError> + Serialize;
    type Response: DeserializeOwned + TryInto<CompletionResponse<Self::Response>>;

    fn endpoint(&self) -> &str;
    fn client(&self) -> &impl HttpClientExt;

    // Default implementation handles the common pattern
    async fn send_completion(&self, request: CompletionRequest) -> Result<...> {
        let request = Self::Request::try_from(request)?;
        let body = serde_json::to_vec(&request)?;
        // ... common logic
    }
}
```

### Duplication: Streaming Response Processing

- **Locations**: Each provider's streaming module
- **Files**: `anthropic/streaming.rs`, `gemini/streaming.rs`, `openai/completion/mod.rs`, etc.
- **Shared Concept**: SSE parsing, chunk aggregation, final response assembly

### Duplication: Usage/Token Counting

- **Locations**: Every provider defines its own `Usage` struct
- **Pattern**: Convert provider-specific usage to `completion::Usage`
```rust
// Repeated in 15+ providers:
impl GetTokenUsage for ProviderUsage {
    fn token_usage(&self) -> Option<completion::Usage> {
        Some(completion::Usage {
            input_tokens: self.prompt_tokens as u64,
            output_tokens: self.completion_tokens as u64,
            total_tokens: self.total_tokens as u64,
        })
    }
}
```

---

## Complexity Hotspots

### Hotspot: gemini/completion.rs

- **Metrics**: 2,169 lines, 5+ nested levels in type conversions, 30+ type definitions
- **Decomposition Strategy**:
  1. Extract `gemini_api_types` module to separate file (currently inline module, ~1500 LOC)
  2. Move `Schema` flattening logic to a dedicated `schema.rs`
  3. Separate streaming logic to `streaming.rs`
  4. Extract test fixtures to `tests/fixtures.rs`

### Hotspot: OneOrMany<T>

- **Metrics**: 702 lines for a container type
- **Concern**: Re-implementing Vec functionality with extra complexity
- **Alternative consideration**: Use `#[serde(deserialize_with)]` on `Vec<T>` fields with validation, or use the `nonempty` crate

```rust
// Current complexity (custom iterators, serde, etc.)
pub struct OneOrMany<T> {
    first: T,
    rest: Vec<T>,
}

// Simpler alternative using existing crate:
use nonempty::NonEmpty;
type OneOrMany<T> = NonEmpty<T>;
```

### Hotspot: Agent completion method

- **Location**: `agent/completion.rs:92-186`
- **Metrics**: Single method with complex async streams and conditional logic
- **Issue**: RAG text extraction, dynamic context fetching, and tool definition retrieval all in one method

**Decomposition:**
```rust
impl<M> Agent<M> {
    async fn fetch_dynamic_context(&self, text: &str) -> Result<Vec<Document>, ...>;
    async fn fetch_tool_definitions(&self, text: Option<&str>) -> Result<Vec<ToolDefinition>, ...>;

    // Main method becomes orchestration
    async fn completion(&self, ...) -> Result<CompletionRequestBuilder<M>, ...> {
        let rag_text = self.extract_rag_text(&prompt, &chat_history);
        let dynamic_ctx = self.fetch_dynamic_context(&rag_text).await?;
        let tools = self.fetch_tool_definitions(rag_text.as_deref()).await?;
        // Build request
    }
}
```

---

## Abstraction Opportunities

### Opportunity: Provider Trait Hierarchy

Create an intermediate trait for providers that share REST API patterns:

```rust
pub trait RestCompletionProvider: CompletionModel {
    const PROVIDER_NAME: &'static str;
    const COMPLETION_ENDPOINT: &'static str;

    type ApiRequest: From<CompletionRequest> + Serialize;
    type ApiResponse: DeserializeOwned;

    fn create_span(&self, request: &CompletionRequest) -> tracing::Span {
        // Default implementation with provider name
    }
}
```

### Opportunity: Content Type Registry

Instead of match arms in every provider, use a registry pattern:

```rust
pub trait ContentTypeHandler<T> {
    fn can_handle(&self, content: &UserContent) -> bool;
    fn convert(&self, content: UserContent) -> Result<T, MessageError>;
}

// Providers register handlers they support
impl GeminiModel {
    fn content_handlers() -> Vec<Box<dyn ContentTypeHandler<Part>>> {
        vec![
            Box::new(TextHandler),
            Box::new(ImageHandler),
            Box::new(VideoHandler), // Gemini-specific
        ]
    }
}
```

### Opportunity: Builder Macro

Many builders follow the same pattern. A derive macro could reduce boilerplate:

```rust
#[derive(Builder)]
pub struct Agent<M> {
    #[builder(setter(into))]
    name: Option<String>,
    #[builder(setter(into))]
    preamble: Option<String>,
    model: M,
    // ...
}
```

---

## Rust-Specific Code Smells

### Smell: Clone Proliferation

- **Location**: `agent/` module (48 `.clone()` calls in 6 files)
- **Concern**: Potential performance impact with large message histories

```rust
// agent/completion.rs
let arc = Arc::new(self.clone()); // Cloning entire agent for each prompt
```

**Recommendation**: Use `Arc<Agent>` from the start or pass references where possible.

### Smell: Stringly-Typed Provider Names

- **Location**: Throughout providers
- **Current**:
```rust
const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-latest";
```

**Better**: Use newtype wrapper for type safety:
```rust
pub struct ModelId(String);

impl ModelId {
    pub const CLAUDE_3_5_SONNET: Self = Self("claude-3-5-sonnet-latest".to_string());
}
```

### Smell: Option Soup in Builders

- **Location**: All builders
- **Pattern**: Many `Option<T>` fields that get unwrapped with defaults in `build()`

```rust
pub struct AgentBuilder<M> {
    temperature: Option<f64>,      // Could be f64 with default
    max_tokens: Option<u64>,       // Could be u64 with default
    preamble: Option<String>,      // Legitimately optional
}
```

---

## Positive Patterns

### Trait-Based Polymorphism
The core trait design (`CompletionModel`, `EmbeddingModel`, `VectorStoreIndex`) is excellent and enables easy extensibility.

### Builder Pattern for Configuration
Fluent builder APIs (`AgentBuilder`, `CompletionRequestBuilder`) provide excellent ergonomics.

### Feature Flags for Optional Dependencies
Good use of Cargo features to make dependencies optional (`pdf`, `epub`, `rmcp`).

### Error Handling
Well-structured error types using `thiserror` with clear error chains.

### WASM Compatibility
Thoughtful abstraction layer (`wasm_compat`) enabling browser deployment.

### Telemetry Integration
OpenTelemetry integration is well-designed with `SpanCombinator` trait.

---

## Recommended Refactoring Priorities

### High Priority (Quick Wins)
1. **Split large provider files** into submodules (~4 hours per provider)
2. **Extract common tracing logic** into a helper function (~2 hours)
3. **Consolidate AgentBuilder and AgentBuilderSimple** (~3 hours)

### Medium Priority (Significant Impact)
4. **Create message conversion traits** to reduce TryFrom boilerplate (~1 day)
5. **Extract gemini_api_types** to separate module (~2 hours)
6. **Consider replacing OneOrMany** with `nonempty` crate (~1 day)

### Lower Priority (Long-term Health)
7. **Standardize provider structure** across all providers (~1 week)
8. **Create provider test harness** for consistent testing (~2 days)
9. **Document architectural decisions** in ADR format (~ongoing)

---

## Metrics Summary

| Metric | Current | Target |
|--------|---------|--------|
| Max file LOC | 2,169 | <500 |
| Provider implementations | 20 | 20 (but DRY) |
| Duplicated patterns | ~15 | ~3 |
| Builder structs | 20+ | <10 (with macro) |
| Clone calls in agent/ | 48 | <20 |

---

## Conclusion

Rig demonstrates solid architectural foundations with room for DRY improvements. The trait-based design is production-quality, but provider implementations have accumulated technical debt through copy-paste patterns. Investing in shared abstractions for common HTTP/serialization patterns would significantly reduce maintenance burden and make adding new providers much faster.

The codebase is maintainable but would benefit from the refactoring priorities outlined above, particularly splitting the large provider files and consolidating the builder patterns.
