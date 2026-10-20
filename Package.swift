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
            url: "https://github.com/NickCarducci/scriberocks-backend/releases/download/v0.1.3/ScribeFFI.xcframework.zip",
            checksum: "2bea3a371dc789127b7e175c2f9ed2ab1d7c7bc5492662bd3a963d03264f1977"
        ),
        .target(
            name: "ScribeCore",
            dependencies: ["ScribeFFI"],
            path: "swift/ScribeCore/Sources/ScribeCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
