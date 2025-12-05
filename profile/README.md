# Machine Learning in Rust

**Building a complete machine learning ecosystem in Rust**

ml-rust is an organization focused on developing high-performance, production-ready machine learning infrastructure entirely in Rust. Our goal is to provide a full stack for LLM development—from tokenization and training to inference—with an emphasis on speed, safety, and efficiency.

## Vision

Modern machine learning tooling is fragmented across languages and frameworks. Python dominates the research space but struggles with performance and deployment. We believe Rust's combination of zero-cost abstractions, memory safety, and fearless concurrency makes it ideal for production ML systems.

ml-rust aims to build an integrated ecosystem where each component is designed to work seamlessly together, from the ground up.

## Projects

### [oxidizr](https://github.com/farhan-syah/oxidizr)

**LLM training framework with next-generation architectures**

A production-grade training framework built on Candle that focuses on modern architectures beyond standard Transformers:

- **Mamba/Mamba2/Mamba3 SSM** - State Space Models with structured state space duality
- **MLA (Multi-Head Latent Attention)** - Compressed KV cache for memory efficiency
- **MoE (Mixture of Experts)** - Fine-grained expert routing with load balancing
- **Hybrid architectures** - Mix SSM and attention layers freely
- **CUDA acceleration** - GPU training with CUDA 12.x/13.x support
- **Multi-GPU support** - Data parallelism with NCCL backend
- **HuggingFace integration** - Direct model publishing

oxidizr is the core training engine of the ml-rust ecosystem, designed for researchers and engineers exploring efficient architectures.

**Status**: Stable. Full training pipeline for Transformer, Mamba2, Mamba3, and hybrid architectures.

---

### [blazr](https://github.com/farhan-syah/blazr)

**High-performance inference server with OpenAI-compatible API**

An inference server specifically designed for models trained with oxidizr. Supports cutting-edge architectures:

- **Mamba2** layers with SSM state management
- **Multi-Head Latent Attention (MLA)** with compressed KV cache
- **Mixture of Experts (MoE)** with sparse routing
- **OpenAI-compatible API** (`/v1/completions`, `/v1/chat/completions`)
- **Streaming support** via Server-Sent Events (SSE)
- **CUDA acceleration** for GPU inference

blazr bridges the gap between experimental architectures and production deployment, making it easy to serve oxidizr models with a familiar API.

**Status**: Production-ready. Full inference pipeline with streaming and OpenAI-compatible API.

---

### [splintr](https://github.com/farhan-syah/splintr)

**High-performance BPE tokenizer with Python bindings**

A Byte-Pair Encoding tokenizer built for speed and compatibility. Supports all major vocabulary formats:

- **OpenAI tiktoken** (cl100k_base, o200k_base for GPT-4/GPT-4o)
- **Meta Llama 3** (~128k tokens)
- **DeepSeek V3/R1** (~128k tokens)
- **Agent tokens** (54 tokens for chat, reasoning, tool-use)

Performance optimizations include:
- PCRE2 with JIT compilation (2-4x faster than fancy-regex)
- Rayon parallelism for batch encoding
- LRU caching for repeated chunks
- Aho-Corasick for O(N) special token matching
- Optional regexr backend with JIT and SIMD

splintr provides the tokenization layer for both oxidizr training and blazr inference, with Python bindings via PyO3 for easy integration.

**Status**: Production-ready for supported vocabularies. Actively adding new vocabulary support.

---

### [regexr](https://github.com/farhan-syah/regexr)

**Specialized pure-Rust regex engine for LLM tokenization**

A purpose-built regex engine designed specifically for tokenization workloads—not a general-purpose regex library. While Rust has the excellent `regex` crate for general use, regexr fills a specific gap: **lookarounds + JIT compilation + pure Rust**.

**Why regexr exists:**
- `regex` crate: Fast, safe, but lacks lookarounds (by design, for linear-time guarantee)
- `fancy-regex`: Supports lookarounds, but no JIT compilation
- `pcre2`: Full features with JIT, but requires C bindings

regexr provides lookarounds, backreferences, and JIT compilation while remaining 100% Rust with no C dependencies.

**Multiple execution backends** automatically selected based on pattern characteristics:
- **ShiftOr / JitShiftOr** - Bit-parallel matching for small patterns
- **LazyDFA / DFA JIT** - General patterns with SIMD prefiltering (Teddy/AVX2)
- **PikeVM / TaggedNfa** - Lookaround and non-greedy quantifiers
- **BacktrackingVm / BacktrackingJit** - Backreference handling

**Status**: Production-ready for tokenization patterns. Passes compliance tests for OpenAI cl100k_base and Meta Llama 3 vocabularies.

---

## The Ecosystem

These projects form a complete, specialized ML pipeline—each component purpose-built for its role:

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Pipeline Flow                        │
└─────────────────────────────────────────────────────────────┘

  regexr ────────────> splintr ────────> oxidizr ────────> blazr
(regex engine)      (tokenization)     (training)      (inference)
     │                   │                 │                │
     └───────────────────┴─────────────────┴────────────────┘
         Pure Rust • Specialized • Zero C Dependencies
```

1. **regexr** provides the regex backend—specialized for tokenization patterns, not general-purpose
2. **splintr** tokenizes training data and model outputs with 10-12x speedup over tiktoken
3. **oxidizr** trains models on modern architectures (Mamba2/3, MLA, MoE)
4. **blazr** serves trained models with an OpenAI-compatible API

Each component can be used independently, but they're designed to work seamlessly together.

## Roadmap

**Completed**:
- regexr with JIT compilation and SIMD acceleration (production-ready for tokenization use cases)
- splintr tokenizer with tiktoken, Llama 3, DeepSeek V3 support (10-12x faster than tiktoken)
- Mamba2/Mamba3 training pipeline in oxidizr with CUDA and multi-GPU support
- blazr inference server with OpenAI-compatible API and streaming

**In Progress** (Q1 2026):
- Distributed training support in oxidizr
- Quantization and optimization in blazr (INT8, FP16)
- End-to-end training-to-deployment documentation

**Planned** (2026+):
- Additional architectures (Attention-free models, hybrid SSMs)
- Browser-based inference via WebAssembly
- Expanded vocabulary support (Qwen, Gemini, Command-R)

## Why Rust for ML?

**Performance**: Zero-cost abstractions and control over memory layout rival C/C++ performance while maintaining safety.

**Safety**: Ownership and borrowing prevent entire classes of bugs common in ML systems (data races, use-after-free, buffer overflows).

**Concurrency**: Fearless concurrency makes it natural to build parallel and distributed systems without sacrificing correctness.

**Deployment**: Single binary deployment with no runtime dependencies simplifies production operations.

**Ecosystem**: Growing ML ecosystem (Candle, Burn, dfdx) with mature foundations (tokio, rayon, serde).

## Current State

ml-rust is **early-stage but usable**. Core functionality across all projects is production-ready for their intended use cases:

- **regexr** - Production-ready for tokenization workloads
- **splintr** - Production-ready, actively used in training and inference pipelines
- **oxidizr** - Stable for Mamba2/3 and hybrid architectures
- **blazr** - Production-ready for serving oxidizr models

These are specialized tools—not general-purpose libraries. We prioritize depth over breadth.

We welcome contributors, early adopters, and feedback from the community.

## Getting Started

Each project has its own documentation and examples:

- **oxidizr**: See [oxidizr/README.md](https://github.com/farhan-syah/oxidizr) for training guide
- **blazr**: See [blazr/README.md](https://github.com/farhan-syah/blazr) for server setup
- **splintr**: See [splintr/README.md](https://github.com/farhan-syah/splintr) for tokenization examples
- **regexr**: See [regexr/README.md](https://github.com/farhan-syah/regexr) for regex engine details

## Community

We're building in the open and value community input:

- **Questions?** Open an issue in the relevant repository
- **Ideas?** Start a discussion or submit a feature request
- **Contributing?** Check individual project CONTRIBUTING.md files

## License

Each project maintains its own license. Most projects use MIT or Apache-2.0. See individual repositories for details.

---

**ml-rust** • Building the future of ML infrastructure in Rust
