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
            url: "https://github.com/NickCarducci/scriberocks-backend/releases/download/v0.1.4/ScribeFFI.xcframework.zip",
            checksum: "e6ff86686adf1a7c439a85b1340f4579732d357d1f84c20d409c88eba5944737"
        ),
        .target(
            name: "ScribeCore",
            dependencies: ["ScribeFFI"],
            path: "swift/ScribeCore/Sources/ScribeCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
