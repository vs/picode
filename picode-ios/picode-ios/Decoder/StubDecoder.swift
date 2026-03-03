import UIKit

/// Stub decoder for development and testing.
///
/// Returns realistic but fake results with simulated processing time.
/// Replace with CoreMLDecoder for production use.
final class StubDecoder: PicodeDecoder {

    /// Simulated messages that the stub can "decode".
    private let stubMessages = [
        "Hello Picode!",
        "https://example.com/abc123",
        "Product: SKU-78432",
        "Scan successful",
        "picode://verify?id=demo",
    ]

    func decode(image: UIImage) async throws -> DecodeResult {
        // Simulate processing time (50-150ms)
        let processingTime = Int.random(in: 50...150)
        try await Task.sleep(for: .milliseconds(processingTime))

        // Generate realistic fake results
        let confidence = Double.random(in: 0.85...0.98)
        let bitAccuracy = Int.random(in: 92...100)
        let message = stubMessages.randomElement() ?? "STUB"

        // Fake per-bit confidences (clustered around high values)
        let rawBits = (0..<100).map { _ in
            Float.random(in: 0.6...0.99)
        }

        // Fake detection region (center of image)
        let region = CGRect(
            x: Double.random(in: 0.1...0.3),
            y: Double.random(in: 0.1...0.3),
            width: Double.random(in: 0.4...0.6),
            height: Double.random(in: 0.4...0.6)
        )

        return DecodeResult(
            message: message,
            confidence: confidence,
            bitAccuracy: bitAccuracy,
            totalBits: 100,
            decodeTimeMs: Double(processingTime),
            detectedRegion: region,
            rawBits: rawBits
        )
    }
}
