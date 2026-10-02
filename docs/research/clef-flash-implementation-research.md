# Flash implementation evidence

Checked 2026-10-01 after reading [the existing research](clef-candle-research.md). Scope was narrowed by the user to Flash v1; the larger model was removed from executable presets/catalogs.

The immutable Flash source and artifact identities from the prior research remain the compatibility baseline. Production uses Candle 0.11.0, tokenizers 0.23.2, safetensors 0.8.0, Reqwest 0.13.5 with its Rustls/AWS-LC provider, Tokio 1.53, and jsonwebtoken 11.1.0 with AWS-LC. Exact resolved versions are in Cargo.lock. Rust stable 1.99.0 is pinned.

Primary source follow-up exposed two compatibility details that mattered during verification:

- [PyTorch 2.11 CPU resize source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp): the uint8 antialiased bicubic path uses adaptive signed integer coefficients and rounds/clips after each separable axis. A generic floating-point image resize produced different patches. The Rust implementation now follows the integer path and matches both square and asymmetric resize fixtures.
- [Transformers 5.10.2 Qwen3.5](https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py): hybrid attention, partial/multimodal rotary, gated delta recurrence, and the vision merger are implemented directly because Candle's published model list lacks this architecture.

The reference environment uses PyTorch 2.11.0, Transformers 5.10.2, torchvision 0.26.0, and CPU F32. `tests/reference/requirements.lock` captures verification-only dependencies; production never imports them. Reference source is checked against SHA-256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` before fixture generation. All model/tokenizer files are hashed before the full oracle loads them offline.

The real Hub transfer redirected large artifacts to `us.aws.cdn.hf.co`; this reviewed origin was added to the restricted transport allowlist. DNS is resolved and public addresses pinned before each TLS connection; authorization remains restricted to the Hub origin.

Dependency audit identified an outdated anyhow version with a RustSec advisory; Cargo.lock was updated to anyhow 1.0.104. The only explicit advisory exception is RUSTSEC-2024-0436, the unmaintained compile-time `paste` macro required transitively by the pinned Candle/gemm/tokenizer ecosystem. No vulnerability is ignored. See deny.toml for the reason.

CI was updated after checking the official [checkout releases](https://github.com/actions/checkout/releases) and [Cargo CI guidance](https://doc.rust-lang.org/cargo/guide/continuous-integration.html). It uses checkout 7.0.1, read-only repository permission, no persisted credentials, the pinned Rust toolchain, and separate CPU/vision and supply-chain gates; no portable all-features GPU claim is made.

JPEG decoder follow-up: image/zune-jpeg and jpeg-decoder differed from Pillow by up to two RGB byte levels on a 37×61 baseline JPEG. The full Flash probability error reached 0.00708, exceeding the unchanged 0.001 gate. Both use a different fixed-point IDCT/conversion path; they were rejected for this runtime. The safe [turbojpeg 1.5.1 API](https://docs.rs/turbojpeg/1.5.1/turbojpeg/struct.Decompressor.html) exposes bounded header probing, output-buffer ownership, and a progressive scan limit. Its bundled C codec is older than the current [official libjpeg-turbo 3.2.0 release](https://github.com/libjpeg-turbo/libjpeg-turbo/releases/tag/3.2.0), so the build instead installs the SHA-256-pinned official 3.2.0 source through `make native-jpeg` and links it explicitly/static. Exact decoded pixels and both baseline 4:4:4/progressive 4:2:0 full-model probabilities now pass. Tolerances were not widened. Native invariants and the mutation harness are documented in the usage and verification records; license notices are retained under docs/licenses.
