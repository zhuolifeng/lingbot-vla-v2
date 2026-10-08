import argparse
import os

from huggingface_hub import constants, snapshot_download


"""
python3 scripts/download_hf_model.py --repo_id deepseek-ai/Janus-1.3B --local_dir Janus-1.3B
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="deepseek-ai/Janus-1.3B")
    parser.add_argument("--local_dir", type=str, default="./Janus-1.3B")
    parser.add_argument(
        "--endpoint", default=None,
        help="Override HF_ENDPOINT, e.g. https://huggingface.co",
    )
    parser.add_argument(
        "--include", nargs="+", default=None,
        help="Optional file patterns; use --include config.json for a small connectivity check.",
    )
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir
    destination = os.path.join(local_dir, repo_id.rsplit("/", 1)[-1])
    print(f"Endpoint: {args.endpoint or constants.ENDPOINT}", flush=True)
    print(f"Destination: {destination}", flush=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        endpoint=args.endpoint,
        allow_patterns=args.include,
    )
