"""Downloading the original PubMed 20k RCT data and record its provenance."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pubmed20k"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT"
)


def validate_file(path):
    """Check that the downloaded file has the expected corpus format."""
    text = path.read_text(encoding="utf-8")

    if not text.startswith("###") or "\t" not in text:
        raise ValueError(f"{path.name} is not a valid PubMed RCT data file.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_records = []

    for filename in ["train.txt", "test.txt"]:
        target = DATA_DIR / filename
        url = f"{BASE_URL}/{filename}"

        if target.exists():
            validate_file(target)
            print(f"Using existing {filename}", flush=True)
        else:
            print(f"Downloading {filename} ...", flush=True)

            request = Request(
                url,
                headers={"User-Agent": "Clinical-SVM-Research/1.0"},
            )

            with urlopen(request, timeout=120) as response:
                content = response.read()

            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(content)

            try:
                validate_file(temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

            print(f"Saved {filename}", flush=True)

        content = target.read_bytes()
        file_records.append({
            "filename": filename,
            "source_url": url,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })

    manifest = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Franck-Dernoncourt/pubmed-rct",
        "dataset": "PubMed_20k_RCT",
        "files": file_records,
    }

    manifest_path = DATA_DIR / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nDataset ready in: {DATA_DIR}")
    print("Source URLs and file checksums saved in source_manifest.json")


if __name__ == "__main__":
    main()