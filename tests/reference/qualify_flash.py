"""Generate unrounded full-release Flash F32 probabilities on an offline CPU oracle.

Run via `make reference-release`; this is verification-only Python.
"""
import argparse
import base64
import io
import tempfile
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from generate import checked_source, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true", help="full-context and still-image parity fixtures")
    parser.add_argument("--jpeg", action="store_true", help="JPEG baseline and progressive decoder parity")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "crates/core/src/artifacts/clef-flash-catalog.json").read_text())
    temporary = tempfile.TemporaryDirectory(prefix="clef-oracle-")
    snapshot = Path(temporary.name)
    for item in manifest["files"]:
        source = args.cache / "blobs/sha256" / item["sha256"]
        assert source.is_file() and source.stat().st_size == item["size"]
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == item["sha256"]
        target = snapshot / item["name"]
        if not target.exists():
            os.link(source, target)
    reference = checked_source()
    torch.set_num_threads(8)
    start = time.monotonic()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float32, local_files_only=True).eval()
    config = json.loads((snapshot / "joint_head_config.json").read_text())
    head = reference.JointSchemaHead(**config)
    head.load_state_dict(load_file(snapshot / "joint_head.safetensors"))
    model = reference.ClefModel(backbone, head.float()).eval()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    results = []
    for index in range(2 if args.jpeg else 3 if args.extended else 100):
        state = ["Checkout failed for all customers.", "An invoice needs a spelling correction.", "The service is working normally.", "The payment gateway is unavailable.", "A customer wants a different plan."][index % 5]
        questions = {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"technical": "outages", "billing": "invoices", "sales": "plans"}},
            "severity": {"type": "score", "instructions": "How severe?", "criteria": ["None", "Minor", "Critical"]},
        }
        if index % 2:
            questions = dict(reversed(list(questions.items())))
        if index % 3 == 0:
            questions["urgent"]["instructions"] = None
        request = {"model": "clef-flash", "state": state if index % 4 else {"message": state, "attempts": index, "locale": "中文"}, "questions": questions}
        input_record = request
        if args.jpeg:
            from PIL import Image
            width,height = [(37,61),(63,37)][index]
            pixels = bytes((x*17+y*31+c*71)%256 for y in range(height) for x in range(width) for c in range(3))
            image = Image.frombytes("RGB",(width,height),pixels)
            stream = io.BytesIO(); image.save(stream,format="JPEG",quality=95 if index==0 else 85,subsampling=0 if index==0 else 2,progressive=bool(index))
            request["images"] = [{"mediaType":"image/jpeg","data":base64.b64encode(stream.getvalue()).decode()}]
            decoded = Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
            input_record = dict(request,images=[decoded])
        elif args.extended:
            if index == 0:
                low, high = 1, 4096
                while low < high:
                    middle = (low + high + 1) // 2
                    request["state"] = "normal " * middle
                    count = len(reference.encode_record(processor.tokenizer, request, max_length=16384).input_ids)
                    if count <= 4096:
                        low = middle
                    else:
                        high = middle - 1
                request["state"] = "normal " * low
                assert len(reference.encode_record(processor.tokenizer, request, max_length=16384).input_ids) == 4096
            else:
                from PIL import Image
                # Deterministic RGB patterns also exercise the asymmetric resize path.
                width, height = [(256, 256), (37, 61)][index-1]
                pixels = bytes((x * 17 + y * 31 + c * 71) % 256 for y in range(height) for x in range(width) for c in range(3))
                image = Image.frombytes("RGB", (width, height), pixels)
                stream = io.BytesIO(); image.save(stream, format="PNG")
                request["images"] = [{"mediaType":"image/png","data":base64.b64encode(stream.getvalue()).decode()}]
                input_record = dict(request, images=[image])
        encoded = reference.encode_record(processor.tokenizer, input_record, max_length=4096, processor=processor)
        with torch.inference_mode():
            logits = model(reference.collate_records([encoded], 0, torch.device("cpu")))[0]
        probabilities = {question.question_id: dict(zip(question.option_ids, field.float().softmax(-1).tolist())) for question, field in zip(encoded.questions, logits)}
        results.append({"request": request, "inputTokens": len(encoded.input_ids), "probabilities": probabilities})
        if index % 10 == 0:
            print(f"Oracle {index+1}, elapsed {time.monotonic()-start:.1f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"revision": manifest["revision"], "dtype": "f32", "sourceSha256": hashlib.sha256((Path('/tmp/clef-reference/joint_schema_model.py')).read_bytes()).hexdigest(), "torch": torch.__version__, "modality": "image" if args.extended or args.jpeg else "text", "records": results}, ensure_ascii=False, indent=2)+"\n")
    print(f"Wrote {len(results)} unrounded records in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
