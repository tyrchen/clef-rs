"""Exercise the release CLI/server with public test-only keys and offline weights.

No downloaded Python is executed; use `make verify-serving` on a large-RAM runner.
"""
import argparse
import base64
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_MODULUS = "yRE6rHuNR0QbHO3H3Kt2pOKGVhQqGZXInOduQNxXzuKlvQTLUTv4l4sggh5_CYYi_cvI-SXVT9kPWSKXxJXBXd_4LkvcPuUakBoAkfh-eiFVMh2VrUyWyj3MFl0HTVF9KwRXLAcwkREiS3npThHRyIxuy0ZMeZfxVL5arMhw1SRELB8HoGfG_AtH89BIE9jDBHZ9dLelK9a184zAf8LwoPLxvJb3Il5nncqPcSfKDDodMFBIMc4lQzDKL5gvmiXLXB1AGLm8KBjfE8s3L5xqi-yUod-j8MtvIj812dkS4QMiRVN_by2h3ZY8LYVGrqZXZTcgn2ujn8uKjXLZVD5TdQ"


def encoded(value):
    return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()


def access_token():
    header = encoded({"alg":"RS256", "typ":"at+jwt", "kid":"test-key"})
    claims = encoded({"iss":"https://issuer.example", "aud":"clef-rs", "sub":"test-principal", "exp":int(time.time())+600,
                      "scope":"clef:decide clef:observe", "models":["clef-flash"]})
    unsigned = f"{header}.{claims}"
    signature = subprocess.run(["openssl", "dgst", "-sha256", "-keyform", "DER", "-sign",
                                str(ROOT / "apps/server/fixtures/test-rsa-private.der")],
                               input=unsigned.encode(), capture_output=True, check=True, timeout=10).stdout
    return unsigned + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="clef-serving-") as directory:
        directory = Path(directory)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        settings = yaml.safe_load((ROOT / "examples/clef.cpu.yaml").read_text())
        settings["cache"]["root"] = str(args.cache.resolve())
        settings["http"].update(bind=f"127.0.0.1:{port}", maxBodyBytes=16*1024*1024)
        settings["models"][0]["execution"]["modality"] = "image"
        settings["runtime"]["maxPayloadBytes"] = 512*1024*1024
        jwks = directory / "jwks.json"
        jwks.write_text(json.dumps({"keys":[{"kty":"RSA", "alg":"RS256", "use":"sig", "kid":"test-key", "n":PUBLIC_MODULUS, "e":"AQAB"}]}))
        settings["auth"].update(jwksFile=str(jwks), keysValidUntilUnixSeconds=int(time.time())+3600)
        config = directory / "config.yaml"
        config.write_text(yaml.safe_dump(settings,sort_keys=False))
        token = access_token()
        environment = dict(os.environ, RAYON_NUM_THREADS="8")
        environment.pop("CLEF_HTTP_BIND",None); environment.pop("CLEF_CACHE_ROOT",None)
        with (directory / "server.log").open("w") as log:
            process = subprocess.Popen([str(args.binary.resolve()), "--json-logs", "serve", "--config", str(config), "--offline"],
                                       stdout=log, stderr=log, env=environment)
            def request(method,path,body=None,authenticated=True):
                connection = http.client.HTTPConnection("127.0.0.1",port,timeout=300)
                try:
                    headers = {"Content-Type":"application/json"}
                    if authenticated: headers["Authorization"] = "Bearer " + token
                    connection.request(method,path,body,headers)
                    response = connection.getresponse()
                    return response.status,dict(response.getheaders()),response.read(1024*1024)
                finally:
                    connection.close()
            try:
                deadline = time.monotonic()+180
                while True:
                    if process.poll() is not None: raise RuntimeError((directory / "server.log").read_text())
                    try:
                        status,_,body = request("GET","/readyz")
                        if status == 200 and json.loads(body)["ready"]: break
                    except (ConnectionError,OSError): pass
                    if time.monotonic() >= deadline: raise TimeoutError("server readiness")
                    time.sleep(.5)
                assert request("GET","/livez",authenticated=False)[0] == 401
                status,_,body = request("GET","/v1/models")
                assert status == 200
                advertised = json.loads(body)["data"]
                assert len(advertised) == 1
                assert advertised[0]["qualification"] == "flash-cpu-f32-v1"
                assert advertised[0]["modality"] == "image"
                records = [
                    json.loads((ROOT / "crates/core/fixtures/release/flash-extended-f32.json").read_text())["records"][1],
                    json.loads((ROOT / "crates/core/fixtures/release/flash-jpeg-f32.json").read_text())["records"][0],
                ]
                for record in records:
                    raw = json.dumps(record["request"],ensure_ascii=False).encode()
                    status,headers,body = request("POST","/v1/systemone",raw)
                    assert status == 200,(status,body)
                    answer = json.loads(body)
                    assert answer["usage"]["input_tokens"] == record["inputTokens"]
                    assert headers["x-clef-revision"] == "17f0b0ad64efb65d273590632833508766b2aae6"
                    for key,value in answer["answers"].items():
                        expected = record["probabilities"][key]
                        if value["type"] == "noul":
                            assert abs(value["noul"]-expected["true"]) <= .000101
                        else:
                            for option,probability in value["probabilities"].items():
                                assert abs(probability-expected[option]) <= .000101
                assert request("GET","/metrics")[0] == 200
                process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=35) == 0,(directory / "server.log").read_text()
                payload = directory / "request.json"; payload.write_bytes(raw)
                result = subprocess.run([str(args.binary.resolve()),"decide","--config",str(config),"--request",str(payload)],
                                        capture_output=True,check=True,timeout=180,env=environment)
                assert json.loads(result.stdout) == answer
                print("PASS: offline CLI, authenticated TCP PNG/JPEG decisions, reference probabilities, provenance, metrics, SIGTERM drain, and exact CLI/HTTP response equality")
            finally:
                if process.poll() is None:
                    process.kill();process.wait(timeout=10)


if __name__ == "__main__":
    main()
