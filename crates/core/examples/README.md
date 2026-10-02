# Core examples

Fetch the pinned model with `clef fetch`, then run `cargo run --release -p clef-rs-core --example decision -- /path/to/model-cache`. This example performs offline CPU F32 inference and writes the same SystemOne response as the HTTP adapter. It requires approximately 19 GB of artifacts and a large-memory machine; inspect the conservative memory plan before loading.
