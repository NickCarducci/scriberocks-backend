#!/bin/sh
# Builds ScribeCore.xcframework (iOS device, iOS simulator, macOS) and the
# generated Swift bindings into swift/ScribeCore.
set -eu
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
export IPHONEOS_DEPLOYMENT_TARGET=15.0 MACOSX_DEPLOYMENT_TARGET=12.0
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

for target in aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin; do
  cargo build --release --lib --target "$target"
done

# Swift bindings from the compiled library's metadata.
cargo build --release --lib
rm -rf target/bindings && mkdir -p target/bindings/headers
cargo run --quiet --release --features bindgen --bin uniffi-bindgen -- \
  generate --library target/release/libscribe.dylib --language swift --out-dir target/bindings
mv target/bindings/scribeFFI.h target/bindings/headers/
mv target/bindings/scribeFFI.modulemap target/bindings/headers/module.modulemap

out=swift/ScribeCore
rm -rf "$out/ScribeFFI.xcframework"
xcodebuild -create-xcframework \
  -library target/aarch64-apple-ios/release/libscribe.a -headers target/bindings/headers \
  -library target/aarch64-apple-ios-sim/release/libscribe.a -headers target/bindings/headers \
  -library target/aarch64-apple-darwin/release/libscribe.a -headers target/bindings/headers \
  -output "$out/ScribeFFI.xcframework" >/dev/null
cp target/bindings/scribe.swift "$out/Sources/ScribeCore/scribe.swift"
echo "built $out/ScribeFFI.xcframework"
