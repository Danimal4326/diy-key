#!/bin/bash
# Generate stage2_grpc/generated/ from the real AI Key .proto files.
# Only the unifi.protect.v1 import closure is compiled.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=stage2_grpc/generated
mkdir -p "$OUT"

# console/v1/api.proto pulls in protos we do not ship; skip them.
python3 -m grpc_tools.protoc \
  -I proto \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  proto/unifi/protect/v1/api.proto \
  proto/unifi/protect/v1/types.proto \
  proto/unifi/core/console/v1/types.proto \
  proto/google/type/latlng.proto

# Add __init__.py for imports, but never under google/: it must stay a namespace package
# merged with google.protobuf.
find "$OUT" -type d -not -path "$OUT/google*" -exec touch {}/__init__.py \;
rm -f "$OUT/google/__init__.py" "$OUT/google/type/__init__.py"

echo "Generated python bindings under $OUT"
