# halo-json-eval

Reproducible evaluation harness and programmatic reward verifier for strict JSON extraction post-training (SFT / GRPO) with [White Circle Halo](https://github.com/whitecircle/halo).

## Metrics Tracked
- **Valid JSON:** Output parses cleanly via `json.loads`.
- **Schema Pass:** Output conforms strictly to the `BookingSchema` Pydantic model (types + required keys).
- **Exact Match:** Extracted fields match ground truth 100%.

## Quickstart

### Local (Dry Run)
```bash
pip install -r requirements.txt
python eval.py --dry-run