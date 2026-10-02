.DELETE_ON_ERROR:

CARGO ?= cargo
PYTHON ?= python3
CLEF_TARGET_DIRECTORY ?= $(shell $(CARGO) metadata --no-deps --format-version=1 | $(PYTHON) -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
CLEF_BINARY ?= $(CLEF_TARGET_DIRECTORY)/release/clef
CLEF_RELEASE_CACHE ?= $(HOME)/.cache/clef-rs
MEDIA_FEATURES ?= vision
CLEF_RELEASE_ORACLE ?= $(CURDIR)/crates/core/fixtures/release/flash-f32.json

build:
	$(CARGO) build --workspace --locked

build-release:
	$(CARGO) build --workspace --locked --release

test:
	$(CARGO) test --workspace --locked

verify: verify-cpu verify-artifacts
	$(CARGO) audit --ignore RUSTSEC-2024-0436
	$(CARGO) deny check

verify-cpu: build test native-jpeg
	$(CARGO) +nightly fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets -- -D warnings -W clippy::pedantic
	$(CARGO) check --workspace --no-default-features
	$(CARGO) test --workspace --features vision
	$(CARGO) clippy --workspace --all-targets --features vision -- -D warnings -W clippy::pedantic
	RUSTDOCFLAGS='-D warnings' $(CARGO) doc --workspace --no-deps --features vision

verify-artifacts:
	$(CARGO) test -p clef-rs-core artifacts::
	$(CARGO) test -p clef-rs-core models::weights::

verify-parity: native-jpeg
	$(CARGO) test -p clef-rs-core models:: --features vision
	CLEF_TOKENIZER='$(CLEF_TOKENIZER)' $(CARGO) test -p clef-rs-core test_should_match_reference_token_ids_and_spans -- --ignored

verify-release:
	CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' CLEF_RELEASE_ORACLE='$(CLEF_RELEASE_ORACLE)' $(CARGO) test -p clef-rs-core test_should_match_full_flash_python_probabilities --release -- --ignored --nocapture
	CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' $(CARGO) test -p clef-rs-server test_should_match_embedded_decisions_over_authenticated_http --release -- --ignored --nocapture

verify-cuda:
	@echo 'CUDA is not qualified for v1. A CUDA runner and separate BF16 full-model parity report are required.'
	@exit 1

verify-metal: native-jpeg verify-metal-operators
	$(CARGO) build --workspace --features metal,vision
	$(CARGO) test --workspace --features metal,vision
	$(CARGO) clippy --workspace --all-targets --features metal,vision -- -D warnings -W clippy::pedantic

reference-env:
	uv venv .venv-reference --python 3.12
	uv pip sync --python .venv-reference/bin/python tests/reference/requirements.lock

reference-fixtures:
	$(PYTHON) tests/reference/generate.py

reference-release:
	$(PYTHON) tests/reference/qualify_flash.py --cache '$(CLEF_RELEASE_CACHE)' --output '$(CLEF_RELEASE_ORACLE)'

release:
	@cargo release tag --execute
	@git cliff -o CHANGELOG.md
	@git commit -a -n -m "Update CHANGELOG.md" || true
	@git push origin master
	@cargo release push --execute

update-submodule:
	@git submodule update --init --recursive --remote

.PHONY: build build-release test verify verify-cpu verify-artifacts verify-parity verify-release verify-cuda verify-metal reference-env reference-fixtures reference-release release update-submodule

reference-extended:
	$(PYTHON) tests/reference/qualify_flash.py --cache '$(CLEF_RELEASE_CACHE)' --output '$(CURDIR)/crates/core/fixtures/release/flash-extended-f32.json' --extended

verify-media-release: native-jpeg
	CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' CLEF_RELEASE_ORACLE='$(CURDIR)/crates/core/fixtures/release/flash-extended-f32.json' $(CARGO) test -p clef-rs-core --features $(MEDIA_FEATURES) test_should_match_full_flash_image_and_context_probabilities --release -- --ignored --nocapture

verify-serving: native-jpeg
	$(CARGO) build -p clef-rs-server --release --features vision
	$(PYTHON) tests/reference/serve_smoke.py --binary '$(CLEF_BINARY)' --cache '$(CLEF_RELEASE_CACHE)'

.PHONY: reference-extended verify-media-release verify-serving

native-jpeg: .native/jpeg-3.2.0/install/lib/libturbojpeg.a

.native/libjpeg-turbo-3.2.0.tar.gz:
	mkdir -p .native
	curl --proto '=https' --tlsv1.2 --fail --location --max-time 120 --retry 3 https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/libjpeg-turbo-3.2.0.tar.gz --output '$@.partial'
	echo '6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e  $@.partial' | shasum -a 256 --check
	mv '$@.partial' '$@'

.native/libjpeg-turbo-3.2.0/.verified: .native/libjpeg-turbo-3.2.0.tar.gz
	echo '6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e  $<' | shasum -a 256 --check
	tar -xzf '$<' -C .native
	touch '$@'

.native/jpeg-3.2.0/install/lib/libturbojpeg.a: .native/libjpeg-turbo-3.2.0/.verified
	cmake -S .native/libjpeg-turbo-3.2.0 -B .native/jpeg-build -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_SIMD=OFF -DWITH_JAVA=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX='$(CURDIR)/.native/jpeg-3.2.0/install' -DCMAKE_INSTALL_LIBDIR=lib
	cmake --build .native/jpeg-build --parallel 8
	cmake --install .native/jpeg-build

fuzz-media: native-jpeg
	$(CARGO) test -p clef-rs-core --features vision test_should_fuzz_bounded_image_decoders --release -- --ignored

.PHONY: native-jpeg fuzz-media

reference-jpeg:
	$(PYTHON) tests/reference/qualify_flash.py --cache '$(CLEF_RELEASE_CACHE)' --output '$(CURDIR)/crates/core/fixtures/release/flash-jpeg-f32.json' --jpeg

verify-jpeg-release: native-jpeg
	CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' CLEF_RELEASE_ORACLE='$(CURDIR)/crates/core/fixtures/release/flash-jpeg-f32.json' $(CARGO) test -p clef-rs-core --features $(MEDIA_FEATURES) test_should_match_full_flash_jpeg_probabilities --release -- --ignored --nocapture

.PHONY: reference-jpeg verify-jpeg-release

CLEF_BENCH_BINARY ?= $(CLEF_TARGET_DIRECTORY)/release/examples/benchmark
BENCHMARK_CONFIG ?= $(CURDIR)/examples/clef.benchmark.yaml
BENCHMARK_RESULTS ?= $(CURDIR)/docs/benchmarks/flash-cpu-metal
BENCHMARK_THREADS ?= 8
MBP_BENCHMARK_RESULTS ?= $(CURDIR)/docs/benchmarks/flash-metal-m5
METAL_PROFILE_CONFIG ?= $(CURDIR)/examples/clef.profile.yaml
METAL_PROFILE_RESULTS ?= $(CLEF_TARGET_DIRECTORY)/clef-metal-profile
METAL_PROFILE_STAMP := $(shell date -u +%Y%m%dT%H%M%SZ)
METAL_PROFILE_RUN = $(METAL_PROFILE_RESULTS)/flash-$(METAL_PROFILE_STAMP)

verify-metal-operators:
	$(CARGO) test -p clef-rs-core --features metal models::metal --release -- --ignored

verify-metal-release:
	CLEF_RELEASE_DEVICE=metal CLEF_RELEASE_DTYPE=f32 CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' CLEF_RELEASE_ORACLE='$(CLEF_RELEASE_ORACLE)' $(CARGO) test -p clef-rs-core --features metal test_should_match_full_flash_python_probabilities --release -- --ignored --nocapture
	CLEF_RELEASE_DEVICE=metal CLEF_RELEASE_DTYPE=f16 CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' CLEF_RELEASE_ORACLE='$(CLEF_RELEASE_ORACLE)' $(CARGO) test -p clef-rs-core --features metal test_should_match_full_flash_python_probabilities --release -- --ignored --nocapture

verify-metal-media: native-jpeg
	CLEF_RELEASE_DEVICE=metal CLEF_RELEASE_DTYPE=f32 $(MAKE) verify-media-release verify-jpeg-release CARGO='$(CARGO)' MEDIA_FEATURES=metal,vision CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)'
	CLEF_RELEASE_DEVICE=metal CLEF_RELEASE_DTYPE=f16 $(MAKE) verify-media-release verify-jpeg-release CARGO='$(CARGO)' MEDIA_FEATURES=metal,vision CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)'

verify-metal-serving: native-jpeg
	CLEF_RELEASE_DEVICE=metal CLEF_RELEASE_DTYPE=f32 CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' $(CARGO) test -p clef-rs-server --features metal,vision test_should_match_embedded_decisions_over_authenticated_http --release -- --ignored --nocapture
	CLEF_RELEASE_DEVICE=metal CLEF_RELEASE_DTYPE=f16 CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)' $(CARGO) test -p clef-rs-server --features metal,vision test_should_match_embedded_decisions_over_authenticated_http --release -- --ignored --nocapture
	$(MAKE) verify-metal-cli CARGO='$(CARGO)' PYTHON='$(PYTHON)' CLEF_RELEASE_CACHE='$(CLEF_RELEASE_CACHE)'

verify-metal-cli: native-jpeg
	$(CARGO) build -p clef-rs-server --release --features metal,vision
	$(PYTHON) tests/reference/serve_smoke.py --binary '$(CLEF_BINARY)' --cache '$(CLEF_RELEASE_CACHE)' --device metal --dtype f32
	$(PYTHON) tests/reference/serve_smoke.py --binary '$(CLEF_BINARY)' --cache '$(CLEF_RELEASE_CACHE)' --device metal --dtype f16

verify-benchmark:
	$(CARGO) test -p clef-rs-core --example benchmark

bench-domain:
	$(CARGO) bench -p clef-rs-core --bench domain
	$(PYTHON) tests/performance/domain_report.py --target '$(CLEF_TARGET_DIRECTORY)' --output '$(BENCHMARK_RESULTS)'

bench-report:
	$(PYTHON) tests/performance/run.py --render-only --binary '$(CLEF_BENCH_BINARY)' --cache '$(CLEF_RELEASE_CACHE)' --config '$(BENCHMARK_CONFIG)' --output '$(BENCHMARK_RESULTS)'

bench-inference:
	$(CARGO) build -p clef-rs-core --example benchmark --release --features metal
	$(PYTHON) tests/performance/run.py --binary '$(CLEF_BENCH_BINARY)' --cache '$(CLEF_RELEASE_CACHE)' --config '$(BENCHMARK_CONFIG)' --output '$(BENCHMARK_RESULTS)' --threads '$(BENCHMARK_THREADS)'

bench-cpu:
	$(CARGO) build -p clef-rs-core --example benchmark --release
	$(PYTHON) tests/performance/run.py --binary '$(CLEF_BENCH_BINARY)' --cache '$(CLEF_RELEASE_CACHE)' --config '$(BENCHMARK_CONFIG)' --output '$(BENCHMARK_RESULTS)' --threads '$(BENCHMARK_THREADS)' --profiles cpu:f32

bench-metal:
	$(CARGO) build -p clef-rs-core --example benchmark --release --features metal
	$(PYTHON) tests/performance/run.py --binary '$(CLEF_BENCH_BINARY)' --cache '$(CLEF_RELEASE_CACHE)' --config '$(BENCHMARK_CONFIG)' --output '$(BENCHMARK_RESULTS)' --threads '$(BENCHMARK_THREADS)' --profiles metal:f32 metal:f16

bench-mbp:
	$(MAKE) bench-metal BENCHMARK_CONFIG='$(CURDIR)/examples/clef.mbp-benchmark.yaml' BENCHMARK_RESULTS='$(MBP_BENCHMARK_RESULTS)'

profile-metal:
	$(CARGO) build -p clef-rs-core --example benchmark --release --features metal
	mkdir -p '$(METAL_PROFILE_RESULTS)'
	env -u CANDLE_METAL_COMPUTE_PER_BUFFER xctrace record --template 'Metal System Trace' --output '$(METAL_PROFILE_RUN).trace' --time-limit 120s --no-prompt --target-stdout '$(METAL_PROFILE_RUN).stdout.log' --env RAYON_NUM_THREADS=$(BENCHMARK_THREADS) --env CANDLE_NUM_THREADS=$(BENCHMARK_THREADS) --launch -- '$(CLEF_BENCH_BINARY)' --cache-dir '$(CLEF_RELEASE_CACHE)' --config '$(METAL_PROFILE_CONFIG)' --device metal --dtype f16 --output '$(METAL_PROFILE_RUN).json'
	xctrace export --input '$(METAL_PROFILE_RUN).trace' --toc --output '$(METAL_PROFILE_RUN).toc.xml'

verify-metal-neural:
	$(CARGO) test -p clef-rs-core --features metal test_should_match_metal4_projections_and_profile_flash_shapes --release -- --ignored --nocapture

profile-metal-kernels:
	$(CARGO) test -p clef-rs-core --features metal test_should_profile_full_width_delta_kernel --release -- --ignored --nocapture

.PHONY: verify-metal-operators verify-metal-release verify-metal-media verify-metal-serving verify-metal-cli verify-benchmark bench-domain bench-report bench-inference bench-cpu bench-metal bench-mbp profile-metal profile-metal-kernels verify-metal-neural
