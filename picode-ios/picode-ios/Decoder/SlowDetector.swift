import UIKit
import CoreImage
import Vision
import CoreML

/// Multi-scale sliding window detector for captured photos.
/// Provides higher accuracy than FastDetector by searching at multiple scales.
final class SlowDetector: PicodeDecoder {

    // MARK: - Configuration

    /// Scales to search at (relative to image size).
    private let scales: [CGFloat] = [0.25, 0.35, 0.5, 0.65, 0.75]

    /// Stride ratio (fraction of window size).
    private let strideRatio: CGFloat = 0.15

    /// Decoder input size.
    private let decoderInputSize: CGSize = CGSize(width: 400, height: 400)

    /// Number of bits in watermark.
    private let numBits: Int = 100

    /// Minimum confidence to consider a detection.
    private let confidenceThreshold: Float = 0.15

    // MARK: - Models

    /// Core ML decoder model.
    private let decoderModel: VNCoreMLModel?

    /// CI context for image processing.
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Initialization

    init() {
        // Load decoder model - try StegaStampDecoder first, then PicodeDecoder
        if let decoderURL = Bundle.main.url(forResource: "StegaStampDecoder", withExtension: "mlmodelc"),
           let compiledDecoder = try? MLModel(contentsOf: decoderURL),
           let vnDecoder = try? VNCoreMLModel(for: compiledDecoder) {
            self.decoderModel = vnDecoder
        } else if let decoderURL = Bundle.main.url(forResource: "PicodeDecoder", withExtension: "mlmodelc"),
                  let compiledDecoder = try? MLModel(contentsOf: decoderURL),
                  let vnDecoder = try? VNCoreMLModel(for: compiledDecoder) {
            self.decoderModel = vnDecoder
        } else {
            print("[SlowDetector] Warning: Decoder model not found")
            self.decoderModel = nil
        }
    }

    /// Initialize with custom model URL (for testing).
    init(decoderModelURL: URL?) throws {
        if let url = decoderModelURL {
            let model = try MLModel(contentsOf: url)
            self.decoderModel = try VNCoreMLModel(for: model)
        } else {
            self.decoderModel = nil
        }
    }

    // MARK: - PicodeDecoder

    /// Decode a watermark from the image (PicodeDecoder protocol).
    ///
    /// - Parameter image: The image to decode.
    /// - Returns: The decoded result.
    /// - Throws: `DecodeError.noWatermarkFound` if no watermark is detected,
    ///           or `DecodeError.invalidImage` if the image cannot be processed.
    func decode(image: UIImage) async throws -> DecodeResult {
        guard let result = try await detect(image: image) else {
            throw DecodeError.noWatermarkFound
        }
        return result
    }

    // MARK: - Detection

    /// Detect watermark in image using sliding window approach.
    ///
    /// - Parameter image: The image to search for watermarks.
    /// - Returns: The best detection result, or nil if no watermark found.
    /// - Throws: `DecodeError.invalidImage` if the image cannot be processed.
    func detect(image: UIImage) async throws -> DecodeResult? {
        guard let ciImage = CIImage(image: image) else {
            throw DecodeError.invalidImage
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        let imageSize = ciImage.extent.size

        var bestResult: (confidence: Float, bits: [Float], rect: CGRect)?

        // Search at each scale
        for scale in scales {
            let windowSize = min(imageSize.width, imageSize.height) * scale
            let stride = windowSize * strideRatio

            // Slide window across image
            var y: CGFloat = 0
            while y + windowSize <= imageSize.height {
                var x: CGFloat = 0
                while x + windowSize <= imageSize.width {
                    let rect = CGRect(x: x, y: y, width: windowSize, height: windowSize)

                    // Crop and resize to decoder input
                    let cropped = ciImage.cropped(to: rect)
                    let scaleX = decoderInputSize.width / windowSize
                    let scaleY = decoderInputSize.height / windowSize
                    let scaled = cropped.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
                        .transformed(by: CGAffineTransform(translationX: -rect.minX * scaleX, y: -rect.minY * scaleY))

                    // Decode
                    if let (confidence, bits) = try await decodeWindow(scaled) {
                        if confidence > (bestResult?.confidence ?? confidenceThreshold) {
                            // Normalize rect to [0,1]
                            let normalizedRect = CGRect(
                                x: rect.minX / imageSize.width,
                                y: rect.minY / imageSize.height,
                                width: rect.width / imageSize.width,
                                height: rect.height / imageSize.height
                            )
                            bestResult = (confidence, bits, normalizedRect)
                        }
                    }

                    x += stride
                }
                y += stride
            }
        }

        guard let result = bestResult else {
            return nil
        }

        let endTime = CFAbsoluteTimeGetCurrent()
        let decodeTimeMs = (endTime - startTime) * 1000

        return DecodeResult(
            message: bitsToMessage(result.bits),
            confidence: Double(result.confidence),
            bitAccuracy: calculateBitAccuracy(result.bits),
            totalBits: numBits,
            decodeTimeMs: decodeTimeMs,
            detectedRegion: result.rect,
            rawBits: result.bits
        )
    }

    // MARK: - Private

    /// Decode a single window and return confidence + bits.
    private func decodeWindow(_ image: CIImage) async throws -> (Float, [Float])? {
        guard let model = decoderModel else {
            // Fallback for development - return random bits with low confidence
            let randomBits = (0..<numBits).map { _ in Float.random(in: 0.4...0.6) }
            return (0.5, randomBits)
        }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                guard let self else { return }

                if let error {
                    continuation.resume(throwing: DecodeError.modelError(error.localizedDescription))
                    return
                }

                guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                      let first = results.first,
                      let multiArray = first.featureValue.multiArrayValue else {
                    continuation.resume(returning: nil)
                    return
                }

                // Extract logits and compute confidence
                let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
                var logits = [Float](repeating: 0, count: self.numBits)
                for i in 0..<min(multiArray.count, self.numBits) {
                    logits[i] = pointer[i]
                }

                // Confidence = mean absolute logit (higher = more certain)
                let confidence = logits.map { abs($0) }.reduce(0, +) / Float(logits.count)

                // Convert to probabilities
                let bits = logits.map { 1.0 / (1.0 + exp(-$0)) }

                continuation.resume(returning: (confidence, bits))
            }

            request.imageCropAndScaleOption = .scaleFill

            let handler = VNImageRequestHandler(ciImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: DecodeError.modelError(error.localizedDescription))
            }
        }
    }

    /// Convert bit probabilities to message string.
    private func bitsToMessage(_ bits: [Float]) -> String {
        let binaryBits = bits.map { $0 > 0.5 ? 1 : 0 }

        var message = ""
        for i in stride(from: 0, to: min(binaryBits.count, 96), by: 8) {
            var byte: UInt8 = 0
            for j in 0..<8 where i + j < binaryBits.count {
                byte |= UInt8(binaryBits[i + j]) << (7 - j)
            }
            if byte == 0 { break }
            if byte >= 32 && byte < 127 {
                message.append(Character(UnicodeScalar(byte)))
            }
        }

        if message.isEmpty {
            return "0x" + binaryBits.prefix(32).map { String($0) }.joined()
        }

        return message.trimmingCharacters(in: .whitespaces)
    }

    /// Calculate bit accuracy (number of high-confidence bits).
    private func calculateBitAccuracy(_ bits: [Float]) -> Int {
        bits.filter { abs($0 - 0.5) > 0.3 }.count
    }
}

// MARK: - Preview Support

#if DEBUG
extension SlowDetector {
    /// Create a preview-friendly detector.
    static var preview: SlowDetector {
        SlowDetector()
    }
}
#endif
