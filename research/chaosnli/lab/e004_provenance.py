"""E004 Model Provenance Capture Script.

Queries local Ollama REST endpoints (/api/tags and /api/show) for gemma3:12b,
records system metadata, model digests, parameters, template, and hardware info,
and writes provenance files to research/chaosnli/artifacts/E004/manifests/.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
from pathlib import Path
import urllib.request
import urllib.error

OLLAMA_BASE_URL = "http://localhost:11434"
MANIFEST_DIR = Path("research/chaosnli/artifacts/E004/manifests")

def http_get(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_post(url: str, data: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def capture_provenance(model_tag: str = "gemma3:12b") -> dict:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Querying Ollama server at {OLLAMA_BASE_URL}...")
    try:
        tags_response = http_get(f"{OLLAMA_BASE_URL}/api/tags")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}/api/tags: {e}. Is Ollama running?")
        
    tags_path = MANIFEST_DIR / "ollama_tags_response.json"
    with open(tags_path, "w", encoding="utf-8") as f:
        json.dump(tags_response, f, indent=2)
        
    try:
        show_response = http_post(f"{OLLAMA_BASE_URL}/api/show", {"name": model_tag})
    except Exception as e:
        raise RuntimeError(f"Failed to call /api/show for model '{model_tag}': {e}")

    show_path = MANIFEST_DIR / f"{model_tag.replace(':', '_')}_show_response.json"
    with open(show_path, "w", encoding="utf-8") as f:
        json.dump(show_response, f, indent=2)
        
    # Extract model digest from tags response if available
    models_list = tags_response.get("models", [])
    model_digest = "unknown"
    for m in models_list:
        if m.get("name") == model_tag or m.get("model") == model_tag:
            model_digest = m.get("digest", "unknown")
            break
            
    details = show_response.get("details", {})
    
    provenance = {
        "ollama_version": tags_response.get("version", "unknown"),
        "model_tag": model_tag,
        "model_digest": model_digest,
        "parameter_size": details.get("parameter_size", "unknown"),
        "quantization_level": details.get("quantization_level", "unknown"),
        "family": details.get("family", "unknown"),
        "context_length": show_response.get("model_info", {}).get("gemma3.context_length", 131072),
        "modelfile": show_response.get("modelfile", ""),
        "template": show_response.get("template", ""),
        "parameters": show_response.get("parameters", ""),
        "hardware_os": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    prov_bytes = json.dumps(provenance, sort_keys=True).encode("utf-8")
    provenance["sha256"] = hashlib.sha256(prov_bytes).hexdigest()
    
    prov_path = MANIFEST_DIR / "model_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
        
    print("=========================================================================")
    print("   MODEL PROVENANCE CAPTURED SUCCESSFULLY")
    print("=========================================================================")
    print(f"  Model Tag:           {provenance['model_tag']}")
    print(f"  Model Digest:        {provenance['model_digest'][:16]}...")
    print(f"  Parameter Size:      {provenance['parameter_size']}")
    print(f"  Quantization Level:  {provenance['quantization_level']}")
    print(f"  Ollama Version:      {provenance['ollama_version']}")
    print(f"  Hardware / OS:       {provenance['hardware_os']}")
    print(f"  Provenance SHA-256:  {provenance['sha256'][:16]}...")
    print("=========================================================================")

    return provenance

if __name__ == "__main__":
    capture_provenance()
