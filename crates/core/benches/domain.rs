//! Model-independent Criterion regression benchmarks.
use std::{hint::black_box, time::Duration};

use anyhow::Result;
use clef_rs_core::{DecisionRequest, encoding::render, types::BoundedJson};
use criterion::{Criterion, Throughput};
use serde_json::json;

fn main() -> Result<()> {
    let mut criterion = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();
    let mut parsing = criterion.benchmark_group("request_parse");
    for bytes in [256, 4096, 65536] {
        let input = serde_json::to_vec(&json!({
            "state":"x".repeat(bytes),
            "questions":{"urgent":{"type":"noul"},"team":{"type":"choice","criteria":{"a":null,"b":null}},"severity":{"type":"score","criteria":[0,1]}}
        }))?;
        DecisionRequest::from_json(&input)?;
        parsing.throughput(Throughput::Bytes(input.len() as u64));
        parsing.bench_function(format!("{bytes}_bytes"), |bencher| {
            bencher.iter(|| black_box(DecisionRequest::from_json(black_box(&input))));
        });
    }
    parsing.finish();
    let value =
        json!({"message":"中文 café", "value":1.234e-7, "nested":[1,null,true,{"b":2,"a":1}]});
    let input = serde_json::to_vec(&value)?;
    BoundedJson::parse(&input)?;
    render(&value)?;
    criterion.bench_function("bounded_json", |bencher| {
        bencher.iter(|| black_box(BoundedJson::parse(black_box(&input))));
    });
    criterion.bench_function("reference_json_render", |bencher| {
        bencher.iter(|| black_box(render(black_box(&value))));
    });
    criterion.final_summary();
    Ok(())
}
