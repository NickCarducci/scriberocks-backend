import Foundation

extension RockVariant {
    /// `changed` as Swift string ranges (the engine reports UTF-8 byte offsets).
    public var changedRanges: [Range<String.Index>] {
        let utf8 = text.utf8
        return changed.compactMap { range in
            guard let lower = utf8.index(utf8.startIndex, offsetBy: Int(range.start), limitedBy: utf8.endIndex),
                  let upper = utf8.index(utf8.startIndex, offsetBy: Int(range.end), limitedBy: utf8.endIndex),
                  lower <= upper else { return nil }
            return lower..<upper
        }
    }
}
