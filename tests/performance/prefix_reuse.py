"""Sequential cold/capture/hit qualification and matched cold executable comparison."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

from run import ROOT, hardware, validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(f"invalid reuse report: {message}")


def validate_reuse(report, profile):
    device, dtype = profile.split(":")
    check(report["complete"] and report["schemaVersion"] == 1, "incomplete report or schema mismatch")
    check(report["profile"] == f"{device}:0:{dtype}:text:4096", f"profile mismatch: {report['profile']}")
    check(report["revision"] == "17f0b0ad64efb65d273590632833508766b2aae6", f"revision mismatch: {report['revision']}")
    for case in report["cases"]:
        name = case["workload"]["name"]
        for mode in ("uncached", "capture", "hit"):
            samples, summary = case[mode + "Ms"], case[mode]
            check(len(samples) == summary["count"] == case["workload"]["samples"], f"{name}/{mode}: sample count mismatch")
            check(all(math.isfinite(n) and n > 0 for n in samples), f"{name}/{mode}: non-positive sample")
            check(math.isclose(sum(samples)/len(samples), summary["meanMs"], rel_tol=1e-9), f"{name}/{mode}: mean mismatch")
            check(summary["minMs"] <= summary["p50Ms"] <= summary["p95Ms"] <= summary["p99Ms"] <= summary["maxMs"], f"{name}/{mode}: percentile order violated")
        check(0 <= case["maximumProbabilityDrift"] <= 1e-3, f"{name}: max drift out of range")
        check(0 <= case["meanProbabilityDrift"] <= 1e-4, f"{name}: mean drift out of range")
        check(case["meanProbabilityDrift"] <= case["maximumProbabilityDrift"], f"{name}: mean drift exceeds max drift")
        check(case["reusedTokens"] == case["stats"]["reusedTokens"], f"{name}: reused token count mismatch")
        check(case["stats"]["bytes"] <= 512 * 1024 * 1024, f"{name}: cache bytes exceed budget")
        if case["workload"]["tokens"] >= 1024:
            check(case["stats"]["hits"] == 1 and case["stats"]["captures"] == 1, f"{name}: expected one hit and one capture")
            check(512 <= case["reusedTokens"] < case["workload"]["tokens"], f"{name}: reused tokens out of range")
        else:
            check(case["reusedTokens"] == 0, f"{name}: short input should bypass caching")


def render(directory, metadata):
    lines = ["# Exact prefix reuse: normal-path measurements", "", f"Measured {metadata['startedAt']}.", "",
             "Model loading, JSON parsing, HTTP/authentication, and hit preparation are excluded from each latency sample. "
             "Every sample includes encoding, inference, unrounded answer conversion and device completion. Full/capture/hit order "
             "rotates; capture runs one full prefill and saves state, while hit computes the suffix over retained complete context. "
             "Short inputs bypass caching. No input compression, distillation or precision change is used.", "",
             "| Profile | Tokens / fields | n per mode | Reused tokens | Old cold ms | New cold ms | Capture ms | Hit ms | Cold→hit speedup |", 
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    details = []
    for profile in metadata["profiles"]:
        name = profile.replace(":", "-")
        current = json.loads((directory / f"{name}.json").read_text())
        validate_reuse(current, profile)
        baseline_path = directory / f"{name}-baseline.json"
        old = {}
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            validate(baseline, profile)
            check(baseline["manifestDigest"] == current["manifestDigest"], "baseline manifest digest mismatch")
            old = {c["workload"]["name"]: c for c in baseline["cases"]}
        for c in current["cases"]:
            workload = c["workload"]
            before = old.get(workload["name"])
            if before:
                check(before["workload"] == workload, "baseline workload mismatch")
            old_ms = f"{before['latency']['meanMs']:.3f}" if before else "—"
            cold, capture, hit = (c[m]["meanMs"] for m in ("uncached", "capture", "hit"))
            lines.append(f"| {profile} | {workload['tokens']} / {workload['fields']} | {workload['samples']} | {c['reusedTokens']} | "
                         f"{old_ms} | {cold:.3f} | {capture:.3f} | {hit:.3f} | {cold/hit:.2f}× |")
            details.append(f"| {profile} | {workload['name']} | {c['hit']['p50Ms']:.3f} / {c['hit']['p95Ms']:.3f} / "
                           f"{c['hit']['p99Ms']:.3f} | {c['hit']['standardDeviationMs']:.3f} | {100*(capture/cold-1):.2f}% | "
                           f"{c['maximumProbabilityDrift']:.8g} / {c['meanProbabilityDrift']:.8g} | "
                           f"{c['stats']['bytes']/1024**2:.2f} |")
        lines += ["", f"[{profile} raw samples]({name}.json); " +
                  (f"[matched old cold samples]({name}-baseline.json)." if old else "No old executable comparison was requested.")]
    lines += ["", "| Profile | Case | Hit p50 / p95 / p99 ms | Hit SD ms | Capture overhead | Unrounded max / mean drift | Retained MiB |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |", *details, "",
              "A hit requires the same principal and an exactly matching long state prefix. Independent states pay the cold/capture cost. "
              "Library caching defaults to disabled; the MBP serving YAML explicitly enables it. Cache staging is included in the "
              "admission plan, and all head evidence is retained. Synthetic repeated-word workload timings are not application decision-quality evidence; "
              "full-weight release and changed-schema/state qualification is reported separately. Percentiles from small samples do not establish an SLO.", "",
              "Hardware, compiler, executable/workload/source hashes and per-process elapsed time are in [metadata.json](metadata.json). "
              "Processes are sequential; OS/driver caches are not cleared. Baseline and new runs are different processes, so their cold means "
              "also include normal thermal and system variation. Tensor memory is live retained storage, not total process or GPU allocator peak.", "",
              "```json", json.dumps(metadata["hardware"], indent=2), "```", ""]
    (directory / "report.md").write_text("\n".join(lines))


def run_process(binary, args, profile, directory, baseline):
    name = profile.replace(":", "-") + ("-baseline" if baseline else "")
    device, dtype = profile.split(":")
    command = [str(binary), "--cache-dir", str(args.cache), "--config", str(args.config),
               "--device", device, "--dtype", dtype, "--output", str(directory / f"{name}.json")]
    if not baseline:
        command.append("--prefix-reuse")
    env = os.environ.copy()
    env.pop("CANDLE_METAL_COMPUTE_PER_BUFFER", None)
    env.update(RAYON_NUM_THREADS=str(args.threads), CANDLE_NUM_THREADS=str(args.threads))
    started = time.monotonic()
    with (directory / f"{name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=7200)
    report = json.loads((directory / f"{name}.json").read_text())
    (validate if baseline else validate_reuse)(report, profile)
    return {"elapsedSeconds": time.monotonic() - started, "binarySha256": digest(binary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=["metal:f16", "metal:f32", "cpu:f32"], default=["metal:f16"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    if args.render_only:
        render(directory, json.loads(metadata_path.read_text()))
        return
    if not args.binary or not args.cache or not args.config or not 1 <= args.threads <= 32:
        parser.error("binary, cache, config and 1..32 threads are required")
    for binary in [args.binary, args.baseline_binary]:
        if binary and not binary.is_file():
            parser.error(f"executable missing: {binary}")
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    sources = {p: digest(ROOT / p) for p in paths if Path(p).suffix in [".rs", ".metal", ".toml", ".lock", ".py"] or p == "Makefile"}
    metadata = {"startedAt": datetime.now(timezone.utc).isoformat(), "hardware": hardware(), "profiles": args.profiles,
                "threads": args.threads, "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                "gitHead": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                "workloadSha256": digest(args.config), "sources": sources, "processes": {}}
    (directory / "workloads.yaml").write_bytes(args.config.read_bytes())
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for profile in args.profiles:
        if args.baseline_binary:
            metadata["processes"][profile + "-baseline"] = run_process(args.baseline_binary.resolve(), args, profile, directory, True)
        metadata["processes"][profile] = run_process(args.binary.resolve(), args, profile, directory, False)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    render(directory, metadata)


if __name__ == "__main__":
    main()
