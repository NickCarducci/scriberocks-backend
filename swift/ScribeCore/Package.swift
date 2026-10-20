// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ScribeCore",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [.library(name: "ScribeCore", targets: ["ScribeCore"])],
    targets: [
        .binaryTarget(name: "ScribeFFI", path: "ScribeFFI.xcframework"),
        .target(
            name: "ScribeCore",
            dependencies: ["ScribeFFI"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(name: "ScribeCoreTests", dependencies: ["ScribeCore"]),
    ]
)
