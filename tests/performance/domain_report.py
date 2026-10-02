"""Export Criterion estimates and raw samples for the domain benchmark report."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ["request_parse/256_bytes", "request_parse/4096_bytes", "request_parse/65536_bytes",
             "bounded_json", "reference_json_render"]
    records = []
    for name in names:
        root = args.target / "criterion" / name / "new"
        records.append({"name": name, "estimatesNanoseconds": json.loads((root / "estimates.json").read_text()),
                        "samples": json.loads((root / "sample.json").read_text()),
                        "benchmark": json.loads((root / "benchmark.json").read_text())})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "domain.json").write_text(json.dumps(records, indent=2)+"\n")
    lines = ["# Domain microbenchmarks", "", "Criterion 0.8.2, release build, 50 samples, one-second warmup and three-second measurement per case. These single-thread CPU measurements do not load model weights. Confidence intervals are Criterion bootstrap estimates, not inference latency percentiles.", "",
             "| Operation | Mean µs | 95% CI lower µs | 95% CI upper µs |", "| --- | ---: | ---: | ---: |"]
    for record in records:
        mean = record["estimatesNanoseconds"]["mean"]
        interval = mean["confidence_interval"]
        lines.append(f"| {record['name']} | {mean['point_estimate']/1000:.3f} | {interval['lower_bound']/1000:.3f} | {interval['upper_bound']/1000:.3f} |")
    lines += ["", "Raw iteration counts, observed nanoseconds, estimates and workload byte throughput are preserved in [domain.json](domain.json). Hardware for the full measurement campaign is recorded in [metadata.json](metadata.json)."]
    (args.output / "domain.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
