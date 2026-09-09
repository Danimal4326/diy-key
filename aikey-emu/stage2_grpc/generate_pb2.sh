#!/bin/bash
# Generates python bindings from the real AI Key .proto files into
# stage2_grpc/generated/, matching the package layout the real firmware's
# uaibd/grpcserver/grpc_server.py imports from (unifi.protect.v1.*,
# unifi.core.console.v1.*).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=stage2_grpc/generated
mkdir -p "$OUT"

# Only compile what unifi.protect.v1 actually imports. console/v1/api.proto
# and device.proto pull in unifi.core.application/speed_test protos we didn't
# copy from the firmware (not needed for the AI Key's ProtectAPI surface).
python3 -m grpc_tools.protoc \
  -I proto \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  proto/unifi/protect/v1/api.proto \
  proto/unifi/protect/v1/types.proto \
  proto/unifi/core/console/v1/types.proto \
  proto/google/type/latlng.proto

# grpc_tools.protoc doesn't create __init__.py for the generated package
# tree; add them so `from unifi.protect.v1 import api_pb2` works. Skip the
# google/ tree deliberately: "google" and "google.protobuf" are namespace
# packages provided by the real protobuf library, and dropping an
# __init__.py under our generated/google/ would turn it into a *regular*
# package that shadows and breaks that namespace merge (ModuleNotFoundError:
# No module named 'google.protobuf').
find "$OUT" -type d -not -path "$OUT/google*" -exec touch {}/__init__.py \;
rm -f "$OUT/google/__init__.py" "$OUT/google/type/__init__.py"

echo "Generated python bindings under $OUT"
