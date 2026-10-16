// swift-tools-version: 6.2
// ScribeCore for apps: Swift bindings over the prebuilt Rust engine.
// Local development uses swift/ScribeCore (path-based xcframework); this
// manifest points at the release asset so apps can depend on the repo URL.
import PackageDescription

let package = Package(
    name: "ScribeCore",
    platforms: [.iOS(.v26), .macOS(.v15)],
    products: [.library(name: "ScribeCore", targets: ["ScribeCore"])],
    targets: [
        .binaryTarget(
            name: "ScribeFFI",
            url: "https://github.com/NickCarducci/scriberocks-backend/releases/download/v0.1.1/ScribeFFI.xcframework.zip",
            checksum: "18ee8b58f754830724784d65cb04e6ef2f47ee17cee468a66970b71c0e637f24"
        ),
        .target(
            name: "ScribeCore",
            dependencies: ["ScribeFFI"],
            path: "swift/ScribeCore/Sources/ScribeCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
