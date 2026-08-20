"""CLI entrypoint: export the deployment CNNGroupNorm model to TorchScript +
ONNX + normalization stats, under models/export/. See src/export.py.
"""

import os

from src.export import export_all


def main():
    paths = export_all()
    print("Exported deployment model:")
    for kind, path in paths.items():
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  {kind:12s} {path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
