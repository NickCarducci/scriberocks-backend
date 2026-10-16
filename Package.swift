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
            url: "https://github.com/NickCarducci/scriberocks-backend/releases/download/v0.1.2/ScribeFFI.xcframework.zip",
            checksum: "bc2cb447d64884e9468995c25cdaee311bfad1c1dffc6042824625e25b2dfa80"
        ),
        .target(
            name: "ScribeCore",
            dependencies: ["ScribeFFI"],
            path: "swift/ScribeCore/Sources/ScribeCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
