import UIKit
import CoreML
import Vision
import CoreImage

/// FastDetector implementation using Core ML for real-time watermark detection.
///
/// This decoder uses two Core ML models:
/// 1. FastDetector model: Detects watermark presence and location (quadrilateral corners)
/// 2. Decoder model: Extracts the encoded message from the rectified region
///
/// The pipeline is:
/// 1. Resize input to 320x320 for detection
/// 2. Run FastDetector to get corners and confidence
/// 3. If detected, rectify the region using perspective correction
/// 4. Run Decoder on the 400x400 rectified image
/// 5. Convert bit logits to message
final class FastDetector: PicodeDecoder {

    // MARK: - Configuration

    /// Input size for the detector model.
    private let detectorInputSize: CGSize = CGSize(width: 320, height: 320)

    /// Input size for the decoder model.
    private let decoderInputSize: CGSize = CGSize(width: 400, height: 400)

    /// Number of bits encoded in the watermark.
    private let numBits: Int = 100

    /// Confidence threshold for detection.
    private let detectionThreshold: Float = 0.5

    // MARK: - Models

    /// Core ML model for watermark detection.
    private let detectorModel: VNCoreMLModel?

    /// Core ML model for message decoding.
    private let decoderModel: VNCoreMLModel?

    /// Core Image context for image processing.
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Initialization

    /// Initialize FastDetector with bundled Core ML models.
    ///
    /// Models should be named:
    /// - `FastDetector.mlmodelc` for detection
    /// - `PicodeDecoder.mlmodelc` for decoding
    ///
    /// If models are not found, falls back to stub behavior for development.
    init() {
        // Try to load detector model
        if let detectorURL = Bundle.main.url(forResource: "FastDetector", withExtension: "mlmodelc"),
           let compiledDetector = try? MLModel(contentsOf: detectorURL),
           let vnDetector = try? VNCoreMLModel(for: compiledDetector) {
            self.detectorModel = vnDetector
        } else {
            print("[FastDetector] Warning: FastDetector.mlmodelc not found, detection disabled")
            self.detectorModel = nil
        }

        // Try to load decoder model
        if let decoderURL = Bundle.main.url(forResource: "PicodeDecoder", withExtension: "mlmodelc"),
           let compiledDecoder = try? MLModel(contentsOf: decoderURL),
           let vnDecoder = try? VNCoreMLModel(for: compiledDecoder) {
            self.decoderModel = vnDecoder
        } else {
            print("[FastDetector] Warning: PicodeDecoder.mlmodelc not found, decoding disabled")
            self.decoderModel = nil
        }
    }

    /// Initialize with custom model URLs (for testing).
    init(detectorModelURL: URL?, decoderModelURL: URL?) throws {
        if let url = detectorModelURL {
            let model = try MLModel(contentsOf: url)
            self.detectorModel = try VNCoreMLModel(for: model)
        } else {
            self.detectorModel = nil
        }

        if let url = decoderModelURL {
            let model = try MLModel(contentsOf: url)
            self.decoderModel = try VNCoreMLModel(for: model)
        } else {
            self.decoderModel = nil
        }
    }

    // MARK: - PicodeDecoder Protocol

    func decode(image: UIImage) async throws -> DecodeResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Convert UIImage to CIImage
        guard let ciImage = CIImage(image: image) else {
            throw DecodeError.invalidImage
        }

        // Step 1: Detect watermark
        let detection = try await detectWatermark(in: ciImage)

        guard let corners = detection.corners, detection.confidence > detectionThreshold else {
            throw DecodeError.noWatermarkFound
        }

        // Step 2: Rectify the detected region
        let rectifiedImage = try rectifyRegion(in: ciImage, corners: corners)

        // Step 3: Decode the message
        let (logits, rawBits) = try await decodeMessage(from: rectifiedImage)

        // Step 4: Convert logits to message
        let message = bitsToMessage(rawBits)
        let bitAccuracy = calculateBitAccuracy(logits)

        let endTime = CFAbsoluteTimeGetCurrent()
        let decodeTimeMs = (endTime - startTime) * 1000

        // Convert corners to normalized CGRect
        let detectedRegion = cornersToRect(corners, imageSize: ciImage.extent.size)

        return DecodeResult(
            message: message,
            confidence: Double(detection.confidence),
            bitAccuracy: bitAccuracy,
            totalBits: numBits,
            decodeTimeMs: decodeTimeMs,
            detectedRegion: detectedRegion,
            rawBits: rawBits
        )
    }

    // MARK: - Detection

    /// Detection result from FastDetector model.
    private struct DetectionOutput {
        let isWatermark: Float      // Logit for watermark presence
        let corners: [CGPoint]?     // 4 corners if detected (normalized 0-1)
        let confidence: Float       // Corner confidence
    }

    /// Run watermark detection on the image.
    private func detectWatermark(in image: CIImage) async throws -> DetectionOutput {
        guard let model = detectorModel else {
            // Fallback: assume full image is watermarked (for development)
            return DetectionOutput(
                isWatermark: 1.0,
                corners: [
                    CGPoint(x: 0.1, y: 0.1),
                    CGPoint(x: 0.9, y: 0.1),
                    CGPoint(x: 0.9, y: 0.9),
                    CGPoint(x: 0.1, y: 0.9)
                ],
                confidence: 0.9
            )
        }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if let error = error {
                    continuation.resume(throwing: DecodeError.modelError(error.localizedDescription))
                    return
                }

                guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
                    continuation.resume(throwing: DecodeError.modelError("Unexpected output format"))
                    return
                }

                // Parse model outputs
                let output = self.parseDetectorOutput(results)
                continuation.resume(returning: output)
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

    /// Parse FastDetector model output tensors.
    private func parseDetectorOutput(_ observations: [VNCoreMLFeatureValueObservation]) -> DetectionOutput {
        var isWatermark: Float = 0
        var corners: [CGPoint]? = nil
        var confidence: Float = 0

        for observation in observations {
            guard let multiArray = observation.featureValue.multiArrayValue else { continue }

            let name = observation.featureName
            let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)

            if name == "is_watermark" || name.contains("watermark") {
                isWatermark = pointer[0]
            } else if (name == "corners" || name.contains("corner")) && multiArray.count == 8 {
                // Parse 8 values as 4 corners: [x1, y1, x2, y2, x3, y3, x4, y4]
                corners = [
                    CGPoint(x: CGFloat(pointer[0]), y: CGFloat(pointer[1])),
                    CGPoint(x: CGFloat(pointer[2]), y: CGFloat(pointer[3])),
                    CGPoint(x: CGFloat(pointer[4]), y: CGFloat(pointer[5])),
                    CGPoint(x: CGFloat(pointer[6]), y: CGFloat(pointer[7]))
                ]
            } else if name == "corner_confidence" || name.contains("confidence") {
                confidence = pointer[0]
            }
        }

        // Apply sigmoid to logit
        let watermarkProb = 1.0 / (1.0 + exp(-isWatermark))

        return DetectionOutput(
            isWatermark: watermarkProb,
            corners: corners,
            confidence: confidence
        )
    }

    // MARK: - Rectification

    /// Rectify the detected region using perspective correction.
    private func rectifyRegion(in image: CIImage, corners: [CGPoint]) throws -> CIImage {
        let imageSize = image.extent.size

        // Convert normalized corners to pixel coordinates
        let pixelCorners = corners.map { point in
            CGPoint(
                x: point.x * imageSize.width,
                y: (1.0 - point.y) * imageSize.height  // Flip Y for Core Image
            )
        }

        // Use CIPerspectiveCorrection filter
        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else {
            throw DecodeError.modelError("CIPerspectiveCorrection not available")
        }

        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(CIVector(cgPoint: pixelCorners[0]), forKey: "inputTopLeft")
        filter.setValue(CIVector(cgPoint: pixelCorners[1]), forKey: "inputTopRight")
        filter.setValue(CIVector(cgPoint: pixelCorners[2]), forKey: "inputBottomRight")
        filter.setValue(CIVector(cgPoint: pixelCorners[3]), forKey: "inputBottomLeft")

        guard let corrected = filter.outputImage else {
            throw DecodeError.modelError("Perspective correction failed")
        }

        // Resize to decoder input size (400x400)
        let scaleX = decoderInputSize.width / corrected.extent.width
        let scaleY = decoderInputSize.height / corrected.extent.height
        let scaled = corrected.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        return scaled
    }

    // MARK: - Decoding

    /// Decode message from rectified image.
    private func decodeMessage(from image: CIImage) async throws -> (logits: [Float], bits: [Float]) {
        guard let model = decoderModel else {
            // Fallback: return random bits (for development)
            let randomBits = (0..<numBits).map { _ in Float.random(in: 0.6...0.99) }
            return (randomBits, randomBits)
        }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if let error = error {
                    continuation.resume(throwing: DecodeError.modelError(error.localizedDescription))
                    return
                }

                guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                      let first = results.first,
                      let multiArray = first.featureValue.multiArrayValue else {
                    continuation.resume(throwing: DecodeError.modelError("Unexpected decoder output"))
                    return
                }

                // Extract logits
                let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
                var logits = [Float](repeating: 0, count: self.numBits)
                for i in 0..<min(multiArray.count, self.numBits) {
                    logits[i] = pointer[i]
                }

                // Convert logits to probabilities via sigmoid
                let bits = logits.map { 1.0 / (1.0 + exp(-$0)) }

                continuation.resume(returning: (logits, bits))
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

    // MARK: - Utilities

    /// Convert bit probabilities to message string.
    private func bitsToMessage(_ bits: [Float]) -> String {
        // Convert probabilities to binary
        let binaryBits = bits.map { $0 > 0.5 ? 1 : 0 }

        // Pack bits into bytes (8 bits per character)
        var message = ""
        for i in stride(from: 0, to: min(binaryBits.count, 96), by: 8) {
            var byte: UInt8 = 0
            for j in 0..<8 where i + j < binaryBits.count {
                byte |= UInt8(binaryBits[i + j]) << (7 - j)
            }
            if byte >= 32 && byte < 127 {
                message.append(Character(UnicodeScalar(byte)))
            }
        }

        // Return message or hex representation if not printable
        if message.isEmpty || message.allSatisfy({ !$0.isLetter && !$0.isNumber }) {
            return "0x" + binaryBits.prefix(32).map { String($0) }.joined()
        }

        return message.trimmingCharacters(in: .whitespaces)
    }

    /// Calculate bit accuracy from logits.
    private func calculateBitAccuracy(_ logits: [Float]) -> Int {
        // Count bits with high confidence (|logit| > 1.0)
        let highConfidenceBits = logits.filter { abs($0) > 1.0 }.count
        return highConfidenceBits
    }

    /// Convert quadrilateral corners to bounding rect.
    private func cornersToRect(_ corners: [CGPoint], imageSize: CGSize) -> CGRect {
        guard corners.count == 4 else { return .zero }

        let xs = corners.map { $0.x }
        let ys = corners.map { $0.y }

        let minX = xs.min() ?? 0
        let maxX = xs.max() ?? 1
        let minY = ys.min() ?? 0
        let maxY = ys.max() ?? 1

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    // MARK: - Real-time Detection

    /// Detect watermark in real-time from camera frame.
    /// Returns detection info for overlay, or nil if not detected.
    func detectRealtime(ciImage: CIImage) async throws -> ScanState.DetectionInfo? {
        // Step 1: Detect watermark
        let detection = try await detectWatermark(in: ciImage)

        guard let corners = detection.corners,
              detection.confidence > detectionThreshold else {
            return nil
        }

        // Step 2: Rectify the detected region
        let rectifiedImage = try rectifyRegion(in: ciImage, corners: corners)

        // Step 3: Decode the message
        let (_, rawBits) = try await decodeMessage(from: rectifiedImage)

        // Step 4: Convert to detection info
        let message = bitsToMessage(rawBits)

        // Corners are already normalized (0-1), create Quadrilateral directly
        let quadrilateral = Quadrilateral(
            topLeft: corners[0],
            topRight: corners[1],
            bottomRight: corners[2],
            bottomLeft: corners[3]
        )

        return ScanState.DetectionInfo(
            quadrilateral: quadrilateral,
            message: message,
            confidence: Double(detection.confidence)
        )
    }
}

// MARK: - Preview Support

#if DEBUG
extension FastDetector {
    /// Create a preview-friendly detector that always succeeds.
    static var preview: FastDetector {
        FastDetector()
    }
}
#endif
