# CLEF and Candle research

Checked 2026-10-01 for [the system design](../../specs/clef-serving-design.md). No existing `docs` or `docs/research` directory was present when research began. The repository is a Rust 2024 template workspace with empty core/server implementations, a build/test Makefile, and no existing feature specification or pinned Rust toolchain. This record separates upstream observations from proposed design choices. No model inference or full-weight download was performed.

## Primary sources and method

| Source | Evidence used |
| --- | --- |
| [Cloudflare announcement](https://blog.cloudflare.com/clef-decision-models/) | Two Apache-2.0 decision releases; prefill-only/joint scoring concept; advertised context and hosted latency |
| [CLEF model card](https://huggingface.co/Cloudflare/clef) | Larger release, files and reference environment |
| [Flash model card](https://huggingface.co/Cloudflare/clef-flash) | Smaller release and same decision interface |
| [Workers AI CLEF docs](https://developers.cloudflare.com/workers-ai/models/clef/) | Hosted model endpoint/context; not a local Candle implementation |
| [CLEF Hub metadata](https://huggingface.co/api/models/Cloudflare/clef?blobs=true) / [Flash metadata](https://huggingface.co/api/models/Cloudflare/clef-flash?blobs=true) | Commit revisions, complete file listing, sizes and LFS digests |
| [Pinned CLEF configuration](https://huggingface.co/Cloudflare/clef/blob/2f3de3dd85f379784083b0814d997ab627200f0c/config.json) / [Flash configuration](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/config.json) | Actual architecture, text/vision shapes and hybrid layer schedule |
| [Pinned inference source](https://huggingface.co/Cloudflare/clef/blob/2f3de3dd85f379784083b0814d997ab627200f0c/joint_schema_model.py) | Encoding, hidden-state/output-embedding inputs, learned head and typed answer conversion |
| [Pinned head configuration](https://huggingface.co/Cloudflare/clef/blob/2f3de3dd85f379784083b0814d997ab627200f0c/joint_head_config.json) / [Flash head](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/joint_head_config.json) | Model-specific input width and shared head geometry |
| [Candle 0.11.0 model list](https://github.com/huggingface/candle/blob/0.11.0/candle-transformers/src/models/mod.rs) | No Qwen3.5 or CLEF implementation in the published module list |
| [Candle VarBuilder source](https://github.com/huggingface/candle/blob/0.11.0/candle-nn/src/var_builder.rs) | Safe buffer/slice loading versus unsafe mmap API |
| [Candle safetensors source](https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/safetensors.rs) | Tensor deserialization and device materialization |
| [Transformers 5.10.2 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | Backbone reference matching the model card's tested version |
| [Current Qwen3.5 docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5) | Hybrid DeltaNet/full-attention architecture and multimodal positions |
| [Hub Rust client 1.0.0](https://docs.rs/hf-hub/1.0.0/hf_hub/) | Current `HFClient` API, async networking and cache semantics |
| [Hub download documentation](https://huggingface.co/docs/hub/en/models-downloading) | Revision-pinned resolve URLs and download mechanisms |
| [Reqwest ClientBuilder](https://docs.rs/reqwest/0.13.5/reqwest/struct.ClientBuilder.html) / [feature definitions](https://github.com/seanmonstar/reqwest/blob/master/Cargo.toml) | Explicit timeout/redirect/proxy controls and current Rustls feature |
| [Rust stable channel manifest](https://static.rust-lang.org/dist/channel-rust-stable.toml) | Stable Rust version at research time |

Fetched small release files through Hugging Face `/resolve/{commit}/...`, inspected their JSON and Python source, and requested only the first 65,536 bytes of each joint-head safetensors file using HTTP Range. Read the safetensors header length and tensor shapes; no head payload or backbone shard was downloaded. This permits exact artifact size/shape research without spending bandwidth or disk space on 19–55 GB releases. Sources were inspected in a temporary directory outside the repository; this record contains conclusions and fingerprints, not copied upstream source.

Current moving-source commits observed: Candle `5ba5d5b468b5b1df40e82dd3d556987bedeea041`; Transformers `a005fc82babfe8871d87746decad2dbee100a125`; hf-hub `473de9430b0fbc544fb1d91102180eb713f8dc87`. Use published Candle 0.11.0 and the release-tested Transformers 5.10.2 baseline for implementation verification. Current `main` can differ from published crates: in particular, hf-hub main includes transfer machinery/features that must not be assumed to exist in 1.0.0.

## Artifact fingerprints and architectural conclusions

| Item | CLEF | Flash |
| --- | --- | --- |
| Hub commit | `2f3de3dd85f379784083b0814d997ab627200f0c` | `17f0b0ad64efb65d273590632833508766b2aae6` |
| `config.json` SHA-256 | `c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58` | `66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238` |
| `joint_head_config.json` SHA-256 | `890be585d75b981201eb96a35f98a9967afe37bc8d220cfdea2e72d56507534f` | `77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0` |
| Head LFS SHA-256 | `a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953` | `19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0` |
| Head file bytes | 256,125,024 | 243,538,016 |
| Backbone index tensor entries | 1,184 | 760 |
| Head tensor entries | 122, plus metadata | Same |

The identical Python source has SHA-256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`. The shared processor config has SHA-256 `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1`. Shared tokenizer config SHA-256 is `91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72`. Hub metadata reports tokenizer JSON SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` and 19,989,325 bytes; the tokenizer file itself was not downloaded in this research.

The model card labels the larger backbone Qwen3.8-27B, but executable configuration declares `qwen3_5`, just as Flash does. Therefore load/port from actual configuration and tensor names. Both mix three linear-attention layers with one full-attention layer. Both indexes have distinct `lm_head.weight`; the learned head explicitly uses its rows for lexical features even without token generation. These observations rule out substituting an ordinary Qwen3 text-generation wrapper.

One discrepancy needs explicit handling: CLEF's text config includes `output_gate_type: "swish"`, while the release-tested Transformers 5.10.2 `Qwen3_5Attention.forward` applies sigmoid to its gate without consulting that field. The release Python loader names that exact class. This was verified by reading the pinned configuration and reference source, not by executing the model. The initial design follows the executed reference and requires a gate-specific parity fixture; an alternative intended computation would need upstream clarification and independent qualification.

The backbone returns all final hidden states with caching disabled. The head normalizes evidence, pools instruction/option spans, routes each option to sequence memory, updates joint field representations through transformer decoders, then combines lexical-prior and learned joint/residual logits. It is not a fixed-label classification projection or independent logprob evaluation. The design specifies the complete translation and parity requirements rather than copying released Python into runtime.

The head tensor-shape products yield 128,056,324 parameters for CLEF and 121,762,820 for Flash. Summing Hub file sizes over backbone/head safetensors gives 54,969,731,792 and 19,063,259,136 bytes respectively. These are on-disk BF16 files, not measured peak RAM/VRAM. No performance, model accuracy, startup time or device fit was measured here.

## Reference contract details that affect correctness

- Questions preserve incoming dictionary order; choice options are sorted for encoding; score criteria preserve list order; `noul` always encodes true then false. Choice ties in answer conversion follow the caller's original criteria order.
- Tokenization happens on individual segments, with explicit span tracking and no added special tokens. Generic whole-prompt chat formatting is not equivalent by assumption.
- Null option descriptions are omitted from semantic objects. Empty/missing/null instructions fall back to the question ID. State strings render raw; nonstrings use compact, sorted JSON.
- The reference can truncate state token prefixes, and defaults to 16,384 tokens. The design intentionally rejects overflow by default and offers reference truncation only explicitly.
- The head uses learned affine LayerNorm and packed PyTorch multihead attention. Its norm/GELU choices differ from the Qwen backbone/vision tower.
- Final per-field softmax operates on F32 logits. `noul` is probability of true; score is an expected ordinal index; output-token count is zero. Four-decimal output rounding happens after full-precision selection/aggregation.

These findings come from the pinned [released inference source](https://huggingface.co/Cloudflare/clef/blob/2f3de3dd85f379784083b0814d997ab627200f0c/joint_schema_model.py). They require exact encoding and numerical fixtures before any compatibility claim.

## Dependency baseline

Queried the official [crates.io API](https://crates.io/data-access) for each crate's `max_stable_version` on 2026-10-01. Links below point to the exact published documentation. Versions are research observations and proposed implementation starting points, not dependency additions. Check advisory/maintenance/license status and the resolved feature graph before adding them. Avoid prereleases by default; tokenizers 1.0 release candidates exist, while 0.23.2 was the latest stable.

| Crate | Latest stable observed | Intended use |
| --- | --- | --- |
| [candle-core](https://docs.rs/candle-core/0.11.0/) | 0.11.0 | Tensors, CPU/CUDA/Metal devices |
| [candle-nn](https://docs.rs/candle-nn/0.11.0/) | 0.11.0 | Model building blocks and weight loading |
| [candle-transformers](https://docs.rs/candle-transformers/0.11.0/) | 0.11.0 | Evaluated; require actual reusable modules before adding |
| [tokenizers](https://docs.rs/tokenizers/0.23.2/) | 0.23.2 | Released tokenizer JSON execution |
| [safetensors](https://docs.rs/safetensors/0.8.0/) | 0.8.0 | Safe weight validation and tensor views |
| [hf-hub](https://docs.rs/hf-hub/1.0.0/) | 1.0.0 | Evaluated alternative; current API uses `HFClient` |
| [reqwest](https://docs.rs/reqwest/0.13.5/) | 0.13.5 | Explicit bounded Hub transport |
| [tokio](https://docs.rs/tokio/1.53.1/) | 1.53.1 | Async actors, channels and IO |
| [axum](https://docs.rs/axum/0.8.9/) | 0.8.9 | HTTP server |
| [tower](https://docs.rs/tower/0.5.3/) | 0.5.3 | Admission/middleware |
| [tower-http](https://docs.rs/tower-http/0.7.1/) | 0.7.1 | Body limits, tracing and HTTP policies |
| [config](https://docs.rs/config/0.15.27/) | 0.15.27 | Typed YAML configuration; inspect YAML parser choice |
| [yaml-rust2](https://docs.rs/yaml-rust2/0.13.0/) | 0.13.0 | Latest observed; config currently resolves the 0.11 line, so share its compatible parser for preflight |
| [validator](https://docs.rs/validator/0.21.0/) | 0.21.0 | Struct validation, supplemented by explicit byte caps |
| [typed-builder](https://docs.rs/typed-builder/0.23.2/) | 0.23.2 | Large configuration builders |
| [serde](https://docs.rs/serde/1.0.229/) / [serde_json](https://docs.rs/serde_json/1.0.151/) | 1.0.229 / 1.0.151 | Strict DTOs and bounded dynamic state |
| [thiserror](https://docs.rs/thiserror/2.0.21/) / [anyhow](https://docs.rs/anyhow/1.0.104/) | 2.0.21 / 1.0.104 | Library errors / application context |
| [tracing](https://docs.rs/tracing/0.1.44/) / [tracing-subscriber](https://docs.rs/tracing-subscriber/0.3.23/) | 0.1.44 / 0.3.23 | Structured diagnostics |
| [clap](https://docs.rs/clap/4.6.7/) | 4.6.7 | Administrative CLI |
| [sha2](https://docs.rs/sha2/0.11.0/) / [url](https://docs.rs/url/2.5.8/) | 0.11.0 / 2.5.8 | Integrity / validated URLs |
| [fs4](https://docs.rs/fs4/1.1.0/) | 1.1.0 | Safe advisory file locks and disk capacity |
| [secrecy](https://docs.rs/secrecy/0.10.3/) | 0.10.3 | Secret redaction |
| [openidconnect](https://docs.rs/openidconnect/4.0.1/) | 4.0.1 | Evaluated maintained auth primitive; resource-server JWT validation needs explicit review |
| [jsonwebtoken](https://docs.rs/jsonwebtoken/11.1.0/) | 11.1.0 | Access-token signature/claim validation with `aws_lc_rs`; published metadata calls maintenance passive, so review advisories and release activity |
| [bytes](https://docs.rs/bytes/1.12.1/) | 1.12.1 | Shared bounded HTTP/media payloads |
| [metrics](https://docs.rs/metrics/0.24.6/) / [metrics-exporter-prometheus](https://docs.rs/metrics-exporter-prometheus/0.18.3/) | 0.24.6 / 0.18.3 | Metrics through protected application routes, defaults disabled |
| [arc-swap](https://docs.rs/arc-swap/1.9.2/) | 1.9.2 | Optional immutable runtime configuration publication |
| [image](https://docs.rs/image/0.25.10/) | 0.25.10 | PNG/JPEG processing after modality qualification |
| [rstest](https://docs.rs/rstest/0.27.0/) / [proptest](https://docs.rs/proptest/1.11.0/) | 0.27.0 / 1.11.0 | Parameterized/property tests |
| [wiremock](https://docs.rs/wiremock/0.6.5/) | 0.6.5 | Download/auth fault fixtures |
| [criterion](https://docs.rs/criterion/0.8.2/) | 0.8.2 | Benchmarks after correctness, not early scaffolding |

Candle's CPU default has no accelerator features. CUDA and Metal are opt-in. Keep Candle crate versions aligned. Candle's own dependency versions are not necessarily each crate's latest stable; resolve compatibility rather than copying its development workspace wholesale.

Reqwest 0.13's feature is `rustls` (not blindly copied older `rustls-tls` examples), and its definition selects aws-lc-rs. hf-hub 1.0.0 has its own `rustls-tls` feature; the feature naming is crate-specific. Turning on TLS requires inspection of the resolved graph, not just the direct manifest. The design chooses a minimal controlled transport so redirect/DNS/byte/range guarantees can be tested directly.

Inspected the published config 0.15.27 crate: its YAML feature depends on yaml-rust2 0.11, and document parsing constructs a YAML tree before conversion. The design therefore preflights bounded parser events before calling the configuration loader, rejecting aliases/tags/duplicate keys. Do not force yaml-rust2 0.13 into config's internal types or duplicate parser versions without a compatibility review. Inspected jsonwebtoken 11.1.0's published manifest/README: use its `aws_lc_rs` provider, explicit access-token validation and bounded JWKS; disable optional PEM parsing when JWK inputs suffice. OpenID Connect login/ID-token validation is a separate concern. The metrics exporter defaults include its own listeners and push gateway; disable those and use the server's protected route.

Stable Rust from the channel manifest: `1.99.0 (b940084d7 2026-09-28)`, distribution date 2026-10-01. Pin latest stable when implementation starts rather than introducing an unrelated toolchain change in a specification-only task.

## Implications and remaining experiments

The design can be implemented in safe Rust/Candle, but published support for this exact model is absent and backend performance remains unproven. Full hidden-state evidence and full-attention blocks impose real memory costs despite no generation. Safe shard buffering avoids forbidden mmap unsafe calls, at the cost of loader staging RAM. Exact probability semantics and reference ordering/normalization matter as much as the architecture port.

Before production, run encoder/operator/head/full-model parity, establish numerical tolerances, measure actual peak memory and prefill performance, and qualify each backend/modality/context profile. A text-only release is a milestone, not full support for the released multimodal capability. No Helm chart or orchestration dependency is needed to define or implement the initial library/server design.
