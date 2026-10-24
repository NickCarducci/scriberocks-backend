// swift-tools-version: 6.2
// ScribeCore for apps: Swift bindings over the prebuilt Rust engine.
// Local development uses swift/ScribeCore (path-based xcframework); this
// manifest points at the release asset so apps can depend on the repo URL.
import PackageDescription

let package = Package(
    name: "ScribeCore",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [.library(name: "ScribeCore", targets: ["ScribeCore"])],
    targets: [
        .binaryTarget(
            name: "ScribeFFI",
            url: "https://github.com/NickCarducci/scriberocks-backend/releases/download/v0.1.6/ScribeFFI.xcframework.zip",
            checksum: "876fbfb96b0ea0fd6684948e8a3ae7e6cdaa5fc36d258ef67ff597128599d35e"
        ),
        .target(
            name: "ScribeCore",
            dependencies: ["ScribeFFI"],
            path: "swift/ScribeCore/Sources/ScribeCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
