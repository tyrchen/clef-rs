"""Exploratory MLX operators, not a CLEF inference or accuracy benchmark.

Run through ``make profile-metal-candidates`` in an isolated MLX 0.32.3 environment.
Inputs are deterministic synthetic arrays. Quantization and compilation are outside
the measurements. Every timed invocation constructs and completes fresh GPU work;
candidate order rotates each round. No model, serving route, or Rust dependency changes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]


def timings(operations, samples, warmups):
    """Return all synchronized wall samples, with independent excluded warmups."""
    names = list(operations)
    result = {name: [] for name in names}
    for round_index in range(warmups + samples):
        for offset in range(len(names)):
            name = names[(round_index + offset) % len(names)]
            started = time.perf_counter()
            output = operations[name]()
            mx.eval(output)
            elapsed = (time.perf_counter() - started) * 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("invalid operator timing")
            if round_index >= warmups:
                result[name].append(elapsed)
            del output
    return {
        name: {
            "samplesMs": values,
            "meanMs": statistics.mean(values),
            "medianMs": statistics.median(values),
            "stddevMs": statistics.stdev(values),
        }
        for name, values in result.items()
    }


def error(actual, expected):
    """Diagnostic tensor drift; these values are not decision probabilities."""
    delta = actual.astype(mx.float32) - expected.astype(mx.float32)
    maximum = mx.max(mx.abs(delta))
    mean = mx.mean(mx.abs(delta))
    relative = mx.sqrt(mx.mean(delta * delta) / mx.mean(expected.astype(mx.float32) ** 2))
    mx.eval(maximum, mean, relative)
    values = {
        "maxAbsolute": maximum.item(),
        "meanAbsolute": mean.item(),
        "relativeRmse": relative.item(),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite diagnostic tensor drift")
    return values


def projections(samples, warmups):
    results = []
    for columns, inner in [(12288, 4096), (4096, 12288), (8192, 4096), (4096, 4096)]:
        weight = (mx.random.normal((columns, inner)) * 0.02).astype(mx.float16)
        quantized = {}
        storage = {"f16": weight.nbytes}
        for name, mode, bits, group in [
            ("affine4", "affine", 4, 64),
            ("affine8", "affine", 8, 64),
            ("mxfp4", "mxfp4", 4, 32),
            ("mxfp8", "mxfp8", 8, 32),
        ]:
            buffers = mx.quantize(weight, group_size=group, bits=bits, mode=mode)
            mx.eval(buffers)
            quantized[name] = (buffers, mode, bits, group)
            storage[name] = sum(buffer.nbytes for buffer in buffers)
        mx.eval(weight)
        for rows in [1024, 4096]:
            x = mx.random.normal((rows, inner)).astype(mx.float16)
            mx.eval(x)
            operations = {"f16": lambda: x @ weight.T}
            for name, (buffers, mode, bits, group) in quantized.items():
                operations[name] = lambda buffers=buffers, mode=mode, bits=bits, group=group: (
                    mx.quantized_matmul(x, *buffers, group_size=group, bits=bits, mode=mode)
                )
                if mode == "mxfp8":
                    operations["mxfp8Activation8"] = lambda buffers=buffers: mx.qqmm(
                        x, buffers[0], buffers[1], group_size=32, bits=8, mode="mxfp8"
                    )
            expected = operations["f16"]()
            mx.eval(expected)
            drift = {name: error(operation(), expected) for name, operation in operations.items()}
            result = {
                "rows": rows, "columns": columns, "inner": inner,
                "weightBytes": storage, "tensorDriftVsF16": drift,
                "timings": timings(operations, samples, warmups),
            }
            results.append(result)
            print("projection", rows, columns, inner, {
                name: round(value["meanMs"], 3) for name, value in result["timings"].items()
            }, flush=True)
        del operations, expected, x, weight, quantized, buffers
        mx.clear_cache()
    return results


def attention(samples, warmups):
    results = []
    for tokens in [1024, 4096]:
        inputs = [mx.random.normal((1, heads, tokens, 256)) for heads in [16, 4, 4]]
        half = [value.astype(mx.float16) for value in inputs]
        mx.eval(inputs, half)
        operations = {
            name: lambda values=values: mx.fast.scaled_dot_product_attention(
                *values, scale=1 / 16, mask="causal", force_fused=True
            )
            for name, values in [("f32", inputs), ("f16", half)]
        }
        expected = operations["f32"]()
        mx.eval(expected)
        result = {
            "tokens": tokens, "queryHeads": 16, "kvHeads": 4, "width": 256,
            "f16TensorDriftVsF32": error(operations["f16"](), expected),
            "timings": timings(operations, samples, warmups),
        }
        results.append(result)
        print("attention", tokens, {
            name: round(value["meanMs"], 3) for name, value in result["timings"].items()
        }, flush=True)
    return results


def preparations(samples, warmups):
    def prepare(q, k, v, g, beta):
        q = mx.repeat(q, 2, axis=1)
        k = mx.repeat(k, 2, axis=1)
        q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        q = q / math.sqrt(128)
        k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        return mx.concatenate([q, k, v, mx.exp(g)[..., None], beta[..., None]], axis=2)

    compiled = mx.compile(prepare)
    results = []
    for tokens in [1024, 4096]:
        values = [mx.random.normal((tokens, heads, width))
                  for heads, width in [(16, 128), (16, 128), (32, 128)]]
        values.extend([-mx.abs(mx.random.normal((tokens, 32))),
                       mx.sigmoid(mx.random.normal((tokens, 32)))])
        mx.eval(values)
        operations = {"eager": lambda: prepare(*values), "compiled": lambda: compiled(*values)}
        expected = operations["eager"]()
        mx.eval(expected)
        result = {
            "tokens": tokens, "compiledTensorDriftVsEager": error(operations["compiled"](), expected),
            "timings": timings(operations, samples, warmups),
        }
        results.append(result)
        print("delta preparation", tokens, {
            name: round(value["meanMs"], 3) for name, value in result["timings"].items()
        }, flush=True)
    return results


def recurrences(samples, warmups):
    """Compare the repository shader with MLX's built-in fused chunk algorithm.

    Normalization/packing are excluded from both timed operations. The built-in
    operation also returns final state; the repository operation returns output
    alone. Both start from zero state. Read our shader rather than duplicate it.
    """
    shader = (ROOT / "crates/core/src/models/delta.metal").read_text()
    declaration = shader.index("kernel void clef_delta(")
    body_start = shader.index("{", declaration)
    header = shader[:declaration].replace(
        "constant ulong CLEF_KEY_DIM [[function_constant(0)]];",
        "constant ulong CLEF_KEY_DIM = 128;",
    )
    body = """
        uint tokens = dimensions[0];
        uint heads = dimensions[1];
        uint value_dim = dimensions[2];
        uint group = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_threadgroup;
    """ + shader[body_start + 1:shader.rindex("}")]
    native = mx.fast.metal_kernel(
        name="clef_recurrence_diagnostic", input_names=["input", "dimensions"], output_names=["output"],
        header="#define CLEF_VALUES 4\n" + header, source=body,
        compile_options={"math_mode": "safe"},
    )
    results = []
    for tokens in [1024, 4096]:
        dimensions = mx.array([tokens, 32, 128], dtype=mx.uint32)
        q = mx.random.normal((1, tokens, 16, 128))
        k = mx.random.normal((1, tokens, 16, 128))
        q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6) / math.sqrt(128)
        k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        v = mx.random.normal((1, tokens, 32, 128))
        gamma = mx.exp(-mx.abs(mx.random.normal((1, tokens, 32))) * 0.2)
        beta = mx.sigmoid(mx.random.normal((1, tokens, 32)))
        packed = mx.concatenate([
            mx.repeat(q, 2, axis=2), mx.repeat(k, 2, axis=2),
            v, gamma[..., None], beta[..., None],
        ], axis=3)
        mx.eval(q, k, v, gamma, beta, packed, dimensions)
        operations = {
            "clefShader": lambda: native(
                inputs=[packed, dimensions],
                grid=(32 * 8 * 128, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[v.shape], output_dtypes=[mx.float32],
            ),
            "mlxFusedChunk": lambda: mx.fast.gated_delta_update(q, k, v, gamma, beta),
        }
        expected = operations["clefShader"]()[0]
        mx.eval(expected)
        result = {
            "tokens": tokens, "keyHeads": 16, "valueHeads": 32, "width": 128,
            "clefShaderSha256": hashlib.sha256(shader.encode()).hexdigest(),
            "mlxOutputTensorDriftVsClefShader": error(operations["mlxFusedChunk"]()[0], expected),
            "timings": timings(operations, samples, warmups),
        }
        stress = {}
        for name, decay in [("zeroDecay", 0.0), ("tinyDecay", 1e-12)]:
            stress_gamma = mx.full(gamma.shape, decay, dtype=mx.float32)
            packed = mx.concatenate([
                mx.repeat(q, 2, axis=2), mx.repeat(k, 2, axis=2),
                v, stress_gamma[..., None], beta[..., None],
            ], axis=3)
            mx.eval(packed, stress_gamma)
            reference = operations["clefShader"]()[0]
            candidate = mx.fast.gated_delta_update(q, k, v, stress_gamma, beta)[0]
            stress[name] = error(candidate, reference)
        result["decayStressTensorDrift"] = stress
        results.append(result)
        print("recurrence", tokens, {
            name: round(value["meanMs"], 3) for name, value in result["timings"].items()
        }, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20, choices=range(2, 101))
    parser.add_argument("--warmups", type=int, default=10, choices=range(1, 101))
    args = parser.parse_args()
    if mx.__version__ != "0.32.3" or not mx.metal.is_available():
        raise RuntimeError("requires isolated MLX 0.32.3 and an available Metal GPU")
    mx.random.seed(42)
    record = {
        "schemaVersion": 1, "measuredAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Synthetic operator exploration; no CLEF model or probability qualification",
        "sourceCommit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip(),
        "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mlxVersion": mx.__version__, "device": mx.device_info(),
        "osVersion": platform.mac_ver()[0], "seed": 42,
        "samplesPerCandidate": args.samples, "excludedWarmupsPerCandidate": args.warmups,
        "timedActivationQuantization": ["mxfp8Activation8"],
        "gatedDeltaEnvironment": {
            name: os.environ.get(name) for name in ["GATED_DELTA_CHUNK", "GATED_DELTA_THRESH"]
        },
        "projection": projections(args.samples, args.warmups),
        "attention": attention(args.samples, args.warmups),
        "deltaPreparation": preparations(args.samples, args.warmups),
        "recurrence": recurrences(args.samples, args.warmups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
