"""Sequential full-weight CPU/Metal measurement; launched through Makefile targets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=30).stdout.strip()


def hardware():
    result = {"system": platform.system(), "release": platform.release(), "machine": platform.machine()}
    if sys.platform == "darwin":
        result.update(cpu=command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                      ramBytes=int(command(["sysctl", "-n", "hw.memsize"])),
                      cpuCores=int(command(["sysctl", "-n", "hw.ncpu"])),
                      osVersion=command(["sw_vers", "-productVersion"]))
        displays = json.loads(command(["system_profiler", "SPDisplaysDataType", "-json"]))["SPDisplaysDataType"]
        result["gpus"] = [{"model": gpu.get("sppci_model"), "cores": gpu.get("sppci_cores"),
                           "metal": gpu.get("spdisplays_mtlgpufamilysupport")} for gpu in displays]
    return result



def validate(report, profile):
    device, dtype = profile.split(":")
    assert report["complete"] and report["schemaVersion"] == 1
    assert report["profile"] == f"{device}:0:{dtype}:text:4096"
    assert report["revision"] == "17f0b0ad64efb65d273590632833508766b2aae6"
    assert report["cases"] and report["concurrency"]
    for case in report["cases"]:
        samples, summary = case["samplesMs"], case["latency"]
        assert len(samples) == summary["count"] == case["workload"]["samples"]
        assert all(math.isfinite(value) and value > 0 for value in samples)
        assert case["actualTokens"] == case["workload"]["tokens"]
        assert math.isclose(sum(samples)/len(samples), summary["meanMs"], rel_tol=1e-9)
        assert summary["minMs"] <= summary["p50Ms"] <= summary["p95Ms"] <= summary["p99Ms"] <= summary["maxMs"]
        assert case["wallMs"] + .01 >= sum(samples)
        assert math.isclose(case["requestsPerSecond"], len(samples)*1000/case["wallMs"], rel_tol=1e-9)
        assert math.isclose(case["inputTokensPerSecond"], case["actualTokens"]*case["requestsPerSecond"], rel_tol=1e-9)
        assert all(math.isfinite(value) and value >= 0 for value in case["diagnostic"].values())
    for group in report["concurrency"]:
        assert group["succeeded"] == len(group["samplesMs"]) == group["latency"]["count"]
        assert group["succeeded"] + sum(group["errors"].values()) == group["attempted"]
        assert all(math.isfinite(value) and value > 0 for value in group["samplesMs"])
        assert math.isclose(group["successfulRequestsPerSecond"], group["succeeded"]*1000/group["wallMs"], rel_tol=1e-9)


def render(directory, metadata):
    rows = []
    for profile in metadata["profiles"]:
        path = directory / f"{profile.replace(':','-')}.json"
        if path.exists() and profile in metadata["processes"]:
            report = json.loads(path.read_text())
            if report.get("complete"):
                validate(report, profile)
                rows.append((profile, report))
    lines = ["# CPU / Metal Flash performance measurements", "", f"Measured {metadata['startedAt']}.", "",
             "```json", json.dumps(metadata["hardware"], indent=2), "```", "",
             "All backends ran in separate sequential processes using the same YAML workload and fixed thread counts. Normal-path timings include encoding, model execution, answer conversion and GPU completion. Cold process load uses the existing OS file cache; no privileged cache flush was performed.", "",
             "| Profile | Snapshot verify s | Model load s | First decision s | Managed load/warmup s | Process peak RSS GiB | Post-load direct GPU peak GiB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for profile, report in rows:
        process = metadata["processes"][profile]
        peak = f"{report['gpuSampledPeakBytes']/2**30:.3f}" if "gpuSampledPeakBytes" in report else "—"
        lines.append(f"| {profile} | {report['verificationMs']/1000:.3f} | {report['loadMs']/1000:.3f} | {report['firstDecisionMs']/1000:.3f} | {report['managedStartupMs']/1000:.3f} | {process['peakRssBytes']/2**30:.3f} | {peak} |")
    lines += ["", "| Profile | Workload | Tokens | Fields | Options/field | n | Mean ms | p50 ms | p95 ms | p99 ms | req/s | input tokens/s |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for profile, report in rows:
        for case in report["cases"]:
            config, latency = case["workload"], case["latency"]
            lines.append(f"| {profile} | {config['name']} | {case['actualTokens']} | {config['fields']} | {config['options']} | {latency['count']} | {latency['meanMs']:.3f} | {latency['p50Ms']:.3f} | {latency['p95Ms']:.3f} | {latency['p99Ms']:.3f} | {case['requestsPerSecond']:.4f} | {case['inputTokensPerSecond']:.2f} |")
    baseline = next((report for profile, report in rows if profile == "cpu:f32"), None)
    if baseline:
        lines += ["", "| Workload | Comparison | CPU mean / Metal mean |", "| --- | --- | ---: |"]
        for profile, report in rows:
            if profile.startswith("metal:"):
                for cpu, gpu in zip(baseline["cases"], report["cases"]):
                    assert cpu["workload"] == gpu["workload"]
                    lines.append(f"| {cpu['workload']['name']} | cpu:f32 / {profile} | {cpu['latency']['meanMs']/gpu['latency']['meanMs']:.2f}× |")
    lines += ["", "Diagnostic stages add explicit barriers and are measured separately from the latency samples above.", "",
              "| Profile | Workload | Encoding ms | Backbone/vision ms | Head/readback ms | Conversion ms | Final sync ms | Diagnostic total ms | Metal allocated GiB |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for profile, report in rows:
        for case in report["cases"]:
            stage = case["diagnostic"]
            memory = case.get("deviceMemory", {})
            allocated = f"{memory['allocatedBytes']/2**30:.3f}" if memory else "—"
            lines.append(f"| {profile} | {case['workload']['name']} | {stage['encodingMs']:.3f} | {stage['backboneMs']:.3f} | {stage['headMs']:.3f} | {stage['conversionMs']:.3f} | {stage['synchronizationMs']:.3f} | {stage['totalMs']:.3f} | {allocated} |")
    lines += ["", "Managed-runtime concurrency uses the same fixed short workload, FIFO within a principal and round-robin across principals. Admission errors are counted separately.", "",
              "| Profile | Concurrency | Attempts | Successes | Errors | Successful req/s | Successful p50 ms | Successful p95 ms |", "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for profile, report in rows:
        for group in report["concurrency"]:
            latency = group["latency"]
            lines.append(f"| {profile} | {group['concurrency']} | {group['attempted']} | {group['succeeded']} | {json.dumps(group['errors'],sort_keys=True)} | {group['successfulRequestsPerSecond']:.4f} | {latency['p50Ms']:.3f} | {latency['p95Ms']:.3f} |")
    lines += ["", "Percentiles use nearest-rank order statistics. With small n, p95/p99 often equal the maximum observation; these are descriptive samples, not production-tail estimates or SLOs. The report preserves every raw timing, error count and environment setting. Metal allocator snapshots include retained buffers. The direct phase also samples allocator usage every 100 ms; its reported peak can miss shorter spikes and excludes initial loading and managed-runtime reload. Metal private allocations can appear as wired system memory outside process RSS, so RSS alone is not a physical-memory comparison. OS process RSS high-water includes both direct and managed phases; unified CPU/GPU allocations must not be summed as independent physical-memory pools.", "",
              "CPU F32 and Metal F32 are a same-precision comparison. Metal F16 uses F16 text backbone with an F32 classifier head/vision tower, residuals/convolution, norms/attention accumulation and recurrent state; compare its numerical qualification separately. CPU uses the default Candle pure-Rust backend, without Accelerate/MKL. Thread settings and the actual executable SHA-256 are recorded in metadata.json."]
    if (directory / "domain.md").exists():
        lines += ["", "See [the domain microbenchmarks](domain.md) for parsing/rendering estimates and [numerical qualification](../../clef-flash-verification.md) for the precision gates."]
    (directory / "report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/clef.benchmark.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=["cpu:f32", "metal:f32", "metal:f16"], default=["cpu:f32","metal:f32","metal:f16"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.output, json.loads((args.output / "metadata.json").read_text())); return
    if not 1 <= args.threads <= 64: raise ValueError("thread limit")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"startedAt": datetime.now(timezone.utc).isoformat(), "hardware": hardware(),
                "rust": command(["rustc","--version"]), "gitCommit": command(["git","rev-parse","HEAD"]),
                "gitDirty": bool(command(["git","status","--porcelain"])),
                "binarySha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "configSha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "profiles": args.profiles, "threadSettings": {"RAYON_NUM_THREADS":str(args.threads), "CANDLE_NUM_THREADS":str(args.threads)}, "processes": {}}
    (args.output / "workloads.yaml").write_bytes(args.config.read_bytes())
    (args.output / "metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    for profile in args.profiles:
        device,dtype = profile.split(":")
        name = profile.replace(":","-")
        environment = dict(os.environ, LC_ALL="C", **metadata["threadSettings"])
        environment.pop("CANDLE_METAL_COMPUTE_PER_BUFFER", None)
        invoke = [str(args.binary.resolve()),"--cache-dir",str(args.cache.resolve()),"--config",str(args.config.resolve()),
                  "--device",device,"--dtype",dtype,"--output",str((args.output / f"{name}.json").resolve())]
        timed = ["/usr/bin/time", "-l" if sys.platform=="darwin" else "-v", *invoke]
        started = time.monotonic()
        print(f"Measuring {profile}; model processes run sequentially", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(timed,env=environment,stdout=log,stderr=log,check=True,timeout=43200,cwd=ROOT)
        text = (args.output / f"{name}.log").read_text()
        if sys.platform=="darwin":
            peak = int(re.search(r"(\d+)\s+maximum resident set size",text).group(1))
        else:
            peak = int(re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)",text).group(1))*1024
        metadata["processes"][profile] = {"wallSeconds":time.monotonic()-started,"peakRssBytes":peak,"command":invoke}
        (args.output / "metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
        render(args.output,metadata)
        print(f"Completed {profile}",flush=True)


if __name__ == "__main__":
    main()
