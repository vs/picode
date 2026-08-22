"""Export the production LDPC code (and Swift test vectors) for the iOS app.

Writes the parity-check structure of LDPC(n, seed=42), the same code the `picode`
CLI and the decode pipeline use, plus the codeword position of every message bit.
With --vectors it also writes noisy decoder outputs with the Python decoding result,
so the Swift decoder can be tested against this reference implementation.

Usage:
    python scripts/export_ldpc.py --bits 72 \
        --output ../picode-ios/picode-ios/Decoder/PicodeLDPC.json \
        --vectors ../picode-ios/picode-iosTests/Fixtures/ldpc_vectors.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from picode.cli import LDPC_SEED, LDPC_SNRS, ldpc_decode
from picode.ecc.ldpc import LDPC
from picode.ecc.payload import decode_text, encode_text


def dense(m: object) -> np.ndarray:
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits", type=int, default=72)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vectors", type=Path)
    args = parser.parse_args()

    ldpc = LDPC(n=args.bits, seed=LDPC_SEED)
    H, G = dense(ldpc.H), dense(ldpc.G)
    n, k = G.shape
    positions = []
    for j in range(k):
        unit = np.zeros(k, dtype=G.dtype)
        unit[j] = 1
        rows = [i for i in range(n) if np.array_equal(G[i], unit)]
        assert len(rows) == 1, f"message bit {j} is not systematic"
        positions.append(rows[0])
    spec = {
        "n": int(n),
        "k": int(k),
        "seed": LDPC_SEED,
        "snrs": list(LDPC_SNRS),
        "checks": [[int(v) for v in np.flatnonzero(row)] for row in H],
        "messagePositions": positions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, separators=(",", ":")) + "\n")
    print(f"Wrote LDPC({n},{k}) spec to {args.output}")

    if args.vectors:
        rng = np.random.default_rng(7)
        cases = []
        for text, flips in [("hello", 0), ("pc-Q7x", 3), ("A1b2", 5), ("hi!", 2), ("zz", 14)]:
            payload, stored = encode_text(text, k)
            codeword = ldpc.encode(torch.tensor([payload], dtype=torch.float32))[0].numpy()
            logits = (codeword * 2 - 1) * rng.uniform(2.0, 6.0, size=n)
            idx = rng.choice(n, size=flips, replace=False)
            logits[idx] *= -rng.uniform(0.1, 0.6, size=flips)  # weakly wrong bits
            bits, ok, _ = ldpc_decode(ldpc, torch.from_numpy(logits[None].astype(np.float32)))
            cases.append({
                "logits": [round(float(x), 5) for x in logits],
                "decodes": bool(ok),
                "text": decode_text(bits) if ok else None,
                "stored": stored,
            })
        args.vectors.parent.mkdir(parents=True, exist_ok=True)
        args.vectors.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
        print(f"Wrote {len(cases)} test vectors to {args.vectors} "
              f"({sum(c['decodes'] for c in cases)} decodable)")


if __name__ == "__main__":
    main()
