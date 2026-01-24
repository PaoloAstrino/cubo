"""Simple Ollama diagnostic utility.

Usage:
  python tools/check_ollama.py --model <model_name>

Returns exit code 0 if Ollama responds and the model can generate, non-zero otherwise.
"""
import argparse
import sys

try:
    import ollama
except Exception as e:
    print("Ollama python package is not installed or not importable:", e)
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Check Ollama availability and model responsiveness")
    parser.add_argument("--model", type=str, default=None, help="Model name to test (optional)")
    args = parser.parse_args()

    import os
    print("OLLAMA_BASE_URL:", os.environ.get("OLLAMA_BASE_URL"))
    print("OLLAMA_HOST:", os.environ.get("OLLAMA_HOST"))
    print("OLLAMA_PORT:", os.environ.get("OLLAMA_PORT"))
    print("Python 'ollama' module info: version=", getattr(ollama, '__version__', 'n/a'))
    try:
        print('ollama dir:', [k for k in dir(ollama) if not k.startswith('_')][:60])
    except Exception:
        pass

    # Try a direct chat call to test model responsiveness.
    model_candidates = [args.model] if args.model else ["llama3", "llama3.2:latest", "llama3.2"]

    for model_name in model_candidates:
        if not model_name:
            continue
        try:
            print(f"Testing Ollama model '{model_name}' with a simple chat call...")
            try:
                resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": "Hello"}], timeout=5)
            except TypeError:
                # Some versions may not accept timeout kwarg
                resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": "Hello"}])

            # Handle either dict or iterator results
            if isinstance(resp, dict):
                msg = resp.get("message", {}).get("content", "")
            else:
                # Streaming-like iterator: collect a few deltas
                try:
                    parts = []
                    for chunk in resp:
                        if isinstance(chunk, dict):
                            delta = chunk.get("message", {}).get("content", "")
                        else:
                            delta = str(chunk)
                        if delta:
                            parts.append(delta)
                            if len(parts) >= 3:
                                break
                    msg = "".join(parts)
                except Exception:
                    msg = str(resp)

            print("Model response snippet:", (msg or "(empty)")[:200])
            print("Ollama check passed for model:", model_name)
            sys.exit(0)
        except Exception as e:
            print(f"Model test failed for '{model_name}':", e)
            # try next candidate

    print("Model chat probes failed; attempting raw HTTP health checks on common endpoints...")
    # If chat probes failed, try basic HTTP endpoints
    try:
        import requests
    except Exception as e:
        print("requests library not available for HTTP probes:", e)
        print("Ollama check failed for all probes. Ensure Ollama is running and the model exists.")
        sys.exit(6)

    endpoints = [
        "http://127.0.0.1:11434/",
        "http://127.0.0.1:11434/health",
        "http://127.0.0.1:11434/v1/models",
        "http://127.0.0.1:11434/models",
        "http://localhost:11434/",
        "http://localhost:11434/health",
    ]
    for ep in endpoints:
        try:
            r = requests.get(ep, timeout=2)
            print(f"{ep} -> HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"{ep} -> failed: {e}")

    print("Ollama check failed for all probes. Ensure Ollama daemon is running and accessible to this process (same user/session).")
    sys.exit(6)

    print("Ollama check passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
