"""Pinned verification oracle, never imported by the production runtime.

Generate small deterministic F32 tensors for backbone/head/encoder parity.
Run via `make reference-fixtures`; network sources are checked against SHA-256.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request

import torch
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5TextModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/core/fixtures/synthetic"
CACHE = Path("/tmp/clef-reference")
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
SOURCE_HASH = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"


def checked_source():
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "joint_schema_model.py"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(
            f"https://huggingface.co/Cloudflare/clef-flash/resolve/{REVISION}/joint_schema_model.py", timeout=30
        ).read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_HASH
    spec = importlib.util.spec_from_file_location("clef_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def vision_fixtures():
    import numpy as np
    from PIL import Image
    from transformers import Qwen2VLImageProcessor, Qwen3_5VisionConfig, Qwen3_5VisionModel
    processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
        size={"shortest_edge":65536,"longest_edge":16777216}, image_mean=[.5]*3,image_std=[.5]*3)
    torch.manual_seed(3391)
    vc = Qwen3_5VisionConfig(hidden_size=8,intermediate_size=16,depth=2,num_heads=2,
        num_position_embeddings=16,out_hidden_size=16,patch_size=16,temporal_patch_size=2,spatial_merge_size=2)
    vision = Qwen3_5VisionModel(vc).eval()
    save_file({"model.visual."+n:t for n,t in vision.state_dict().items()}, OUT / "vision.safetensors")
    (OUT / "vision-config.json").write_text(json.dumps(vc.to_dict(),indent=2)+"\n")
    for height,width in [(256,256),(37,61)]:
        rgb = (np.arange(height*width*3,dtype=np.uint32)*37%256).astype(np.uint8).reshape(height,width,3)
        image = Image.fromarray(rgb)
        prepared = processor(images=[image],return_tensors="pt")
        with torch.no_grad():
            output=vision(prepared.pixel_values,grid_thw=prepared.image_grid_thw).pooler_output
        name=f"image-{height}-{width}"
        (OUT / (name+".rgb")).write_bytes(rgb.tobytes())
        save_file({"patches":prepared.pixel_values,"output":output},OUT/(name+".safetensors"))
        (OUT/(name+".json")).write_text(json.dumps({"height":height,"width":width,"grid":prepared.image_grid_thw[0].tolist()})+"\n")


def jpeg_and_position_fixtures():
    from PIL import Image
    from types import SimpleNamespace
    from transformers import Qwen3_5Model
    width,height=37,61
    pixels=bytes((x*17+y*31+c*71)%256 for y in range(height) for x in range(width) for c in range(3))
    image=Image.frombytes("RGB",(width,height),pixels)
    image.save(OUT / "image-37-61.jpg",quality=95,subsampling=0)
    (OUT / "image-37-61-jpeg.rgb").write_bytes(Image.open(OUT / "image-37-61.jpg").convert("RGB").tobytes())
    class RopeOracle:
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids=Qwen3_5Model.get_vision_position_ids
        get_rope_index=Qwen3_5Model.get_rope_index
    grids=[[1,4,4],[1,4,8],[1,8,4],[1,4,4]]
    spans=[[5,9],[12,20],[23,31],[35,39]]
    tokens=49; types=torch.zeros((1,tokens),dtype=torch.int32)
    for start,end in spans: types[0,start:end]=1
    positions,_=RopeOracle().get_rope_index(torch.zeros((1,tokens),dtype=torch.long),types,image_grid_thw=torch.tensor(grids))
    (OUT / "media-positions.json").write_text(json.dumps(dict(tokens=tokens,grids=grids,spans=spans,positions=positions[:,0].T.tolist()),indent=2)+"\n")


def main():
    assert torch.__version__.split("+")[0] == "2.11.0"
    import transformers
    assert transformers.__version__ == "5.10.2"
    torch.manual_seed(7729)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    reference = checked_source()
    head_config = dict(hidden_size=16, width=8, routing_layers=2, layers=4, heads=2, feedforward=32)
    head = reference.JointSchemaHead(**head_config).eval()
    hidden = torch.randn(1, 20, 16)
    ids = torch.arange(20).unsqueeze(0)
    output_embeddings = torch.randn(64, 16)
    fields = [reference.EncodedQuestion("n", 0, (1, 3), ((3, 5), (5, 7)), ("true", "false")),
              reference.EncodedQuestion("c", 1, (7, 9), ((9, 11), (11, 13), (13, 15)), ("a", "b", "c")),
              reference.EncodedQuestion("s", 2, (15, 17), ((17, 18), (18, 19)), ("0", "1"))]
    record = reference.EncodedRecord(tuple(range(20)), tuple(fields), "fixture")
    with torch.no_grad():
        logits = head(hidden, ids, torch.ones_like(ids), [record], output_embeddings)[0]
    save_file(head.state_dict(), OUT / "head.safetensors")
    save_file({"hidden": hidden[0], "output_embeddings": output_embeddings,
               **{f"logits.{i}": l for i, l in enumerate(logits)}}, OUT / "head-input.safetensors")
    (OUT / "head-config.json").write_text(json.dumps(head_config, indent=2) + "\n")

    config = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=64,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.,
            "partial_rotary_factor": .5, "mrope_section": [1, 1, 0]},
        rms_norm_eps=1e-6, attn_implementation="eager")
    model = Qwen3_5TextModel(config).eval()
    # Nontrivial normalization weights/gates exercise offset RMSNorm, not just defaults.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "layernorm.weight" in name or "self_attn.q_norm" in name or "self_attn.k_norm" in name:
                parameter.copy_(torch.randn_like(parameter) * .15)
        states = model(input_ids=ids, use_cache=False).last_hidden_state[0]
    weights = {"model.language_model." + n: t for n, t in model.state_dict().items()}
    weights["lm_head.weight"] = output_embeddings
    save_file(weights, OUT / "backbone.safetensors")
    save_file({"hidden": states}, OUT / "backbone-output.safetensors")
    (OUT / "backbone-config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")

    tokenizer_path = CACHE / "tokenizer.json"
    if not tokenizer_path.exists():
        data = urllib.request.urlopen(f"https://huggingface.co/Cloudflare/clef-flash/resolve/{REVISION}/tokenizer.json", timeout=60).read()
        assert hashlib.sha256(data).hexdigest() == "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
        tokenizer_path.write_bytes(data)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    records = [dict(state="Checkout failed 中文", questions={"urgent": {"type": "noul"},
        "team": {"type": "choice", "instructions": {"z": 1e-5, "a": "中文"}, "criteria": {"z": None, "a": {"x": 1.0}}},
        "severity": {"type": "score", "criteria": ["Low", "High"]}}),
        dict(state={"floats": [1e-5, 1e16, -0.0, 1.0], "string": "é\n\t"}, questions={"q": {"type":"noul","instructions":None}})]
    golden = []
    for record in records:
        encoded = reference.encode_record(tokenizer, record)
        golden.append(dict(request=record, ids=encoded.input_ids,
            questions=[dict(instruction=q.question_span, options=q.option_spans, kind=q.question_type) for q in encoded.questions]))
    (OUT / "encoding.json").write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
        source_sha256=SOURCE_HASH, seed=7729, dtype="float32", device="cpu", tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()), indent=2) + "\n")
    vision_fixtures()
    jpeg_and_position_fixtures()
    print("Generated deterministic backbone, head, encoder and vision fixtures.")


if __name__ == "__main__":
    main()
