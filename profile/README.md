# ml-rust

Machine learning infrastructure in Rust, from tensors to inference servers. The stack needs no Python runtime.

## Projects

| Project | What it does |
| --- | --- |
| [blazr](https://github.com/ml-rust/blazr) | Inference server for LLMs, speech, and embeddings. OpenAI and Anthropic compatible API. Runs on CPU, CUDA, and WebGPU. C API for apps. |
| [oxidizr](https://github.com/ml-rust/oxidizr) | LLM training for Mamba, MLA, MoE, and hybrid models. |
| [voxr](https://github.com/ml-rust/voxr) | Audio and speech on boostr: decoding, TTS, ASR, and VAD. |
| [compressr](https://github.com/ml-rust/compressr) | Model conversion and quantization between SafeTensors and GGUF. |
| [boostr](https://github.com/ml-rust/boostr) | ML framework on numr: attention, MoE, Mamba, quantization, model loaders. |
| [numr](https://github.com/ml-rust/numr) | Tensors, linear algebra, and FFT on CPU, CUDA, and WebGPU. |
| [solvr](https://github.com/ml-rust/solvr) | Scientific computing on numr: optimization, ODE and PDE solvers, statistics. |
| [splintr](https://github.com/ml-rust/splintr) | BPE, WordPiece, and SentencePiece tokenizers. Python bindings. |
| [regexr](https://github.com/ml-rust/regexr) | Regex engine with lookarounds and JIT, built for tokenization. |
| [nexar](https://github.com/ml-rust/nexar) | Distributed runtime with QUIC transport and collectives. |
| [treeboost](https://github.com/ml-rust/treeboost) | Gradient boosted decision trees for large tabular data. |
| [fluxbench](https://github.com/ml-rust/fluxbench) | Benchmarking with crash isolation and bootstrap statistics. |
| [venus](https://github.com/ml-rust/venus) | Reactive notebooks for Rust, stored as plain `.rs` files. |
| [axp](https://github.com/ml-rust/axp) | Open protocol that gives AI agents a sandbox to run code and tools. |
| [cargo-reclaim](https://github.com/ml-rust/cargo-reclaim) | Background companion to `cargo clean` for large Rust projects. |

## How they fit

```
numr
├── solvr
└── boostr
    ├── blazr      inference
    ├── oxidizr    training
    ├── voxr       speech
    └── compressr  conversion

splintr, regexr    tokenization, used by boostr and blazr
```

## Start

```bash
cargo install blazr --locked
blazr pull Qwen/Qwen2.5-0.5B
blazr list
blazr serve --model <name>
```

See the [blazr README](https://github.com/ml-rust/blazr#readme) for models, formats, and the API.
