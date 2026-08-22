import Foundation

/// Text packing shared with `picode.ecc.payload` (Python). The first payload bit selects
/// a compact 6-bit alphabet (0) or UTF-8 bytes (1); a zero symbol ends the text.
enum PayloadText {
    static let alphabet: [Character] = Array(
        "\0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    )

    static func decode(_ bits: [Int]) -> String {
        guard let mode = bits.first else { return "" }
        let width = mode == 0 ? 6 : 8
        var values: [Int] = []
        var i = 1
        while i + width <= bits.count {
            let value = bits[i..<(i + width)].reduce(0) { ($0 << 1) | $1 }
            if value == 0 { break }
            values.append(value)
            i += width
        }
        if mode == 0 {
            return String(values.map { alphabet[$0] })
        }
        return String(decoding: values.map { UInt8($0) }, as: UTF8.self)
    }
}
