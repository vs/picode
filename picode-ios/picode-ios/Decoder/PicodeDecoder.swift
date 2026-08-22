import UIKit

/// Result of decoding a picode watermark from an image.
struct DecodeResult: Identifiable {
    let id = UUID()

    /// Decoded message text (converted from bits).
    let message: String

    /// Overall confidence score (0.0 to 1.0).
    let confidence: Double

    /// Number of bits decoded with high confidence.
    let bitAccuracy: Int

    /// Total number of bits in the watermark.
    let totalBits: Int

    /// Time taken to decode in milliseconds.
    let decodeTimeMs: Double

    /// Region where watermark was detected in the image (normalized 0-1 coordinates).
    let detectedRegion: CGRect?

    /// Per-bit confidence scores (for debug/visualization).
    let rawBits: [Float]
}

/// Protocol for picode watermark decoders.
///
/// Implementations can be swapped between stub (for development),
/// Core ML (for production), or mock (for testing).
protocol PicodeDecoder {
    /// Decode a picode watermark from an image.
    ///
    /// - Parameter image: The image to decode.
    /// - Returns: The decoded result containing message and metadata.
    /// - Throws: If decoding fails.
    func decode(image: UIImage) async throws -> DecodeResult
}

/// Errors that can occur during decoding.
enum DecodeError: LocalizedError {
    case noWatermarkFound
    case invalidImage
    case errorCorrectionFailed
    case modelError(String)

    var errorDescription: String? {
        switch self {
        case .noWatermarkFound:
            return "No picode watermark found in image"
        case .invalidImage:
            return "Could not process the image"
        case .errorCorrectionFailed:
            return "Found a picode but could not read it reliably. Hold steadier or move closer."
        case .modelError(let message):
            return "Decoder error: \(message)"
        }
    }
}
