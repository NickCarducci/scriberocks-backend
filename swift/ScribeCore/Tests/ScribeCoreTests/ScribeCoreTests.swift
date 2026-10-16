import Foundation
import Testing
@testable import ScribeCore

/// Runs against the real models in core/models (build them first: see README).
struct ScribeCoreTests {
    static let models = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "models").path()

    @Test func variantsComeBackWithReadableRanges() throws {
        let engine = try ScribeEngine(modelsDir: Self.models)
        let variants = engine.variants(rock: "I should go running more but I'm really too tired after work.", limit: 20)
        #expect(variants.count >= 5)
        #expect(variants.contains { $0.text == "I'm too tired after work." })
        let reframed = try #require(variants.first { $0.ops == ["reframe"] })
        #expect(reframed.changedRanges.map { String(reframed.text[$0]) }.allSatisfy { !$0.isEmpty })
    }

    @Test func framesCarryModalityAndLinks() throws {
        let engine = try ScribeEngine(modelsDir: Self.models)
        let frames = engine.frames(rock: "I want to move back home because I miss my mom.")
        #expect(frames.first?.modality == "want")
        #expect(frames.contains { $0.link == "because" && $0.verb == "miss" })
    }

    @Test func missingModelsThrow() {
        #expect(throws: ScribeError.self) { try ScribeEngine(modelsDir: "/nonexistent") }
    }
}
