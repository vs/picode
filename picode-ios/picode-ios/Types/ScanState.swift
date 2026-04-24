import Foundation

/// State of the scanning process.
enum ScanState: Equatable {
    /// Actively scanning, no watermark detected yet.
    case scanning

    /// Watermark detected in viewfinder with live decode result.
    case detected(DetectionInfo)

    /// Photo captured, running SlowDetector analysis.
    case analyzing

    /// Analysis complete, showing result.
    case result(DecodeResult)

    /// Analysis complete, no watermark found.
    case noResult

    // MARK: - Equatable

    static func == (lhs: ScanState, rhs: ScanState) -> Bool {
        switch (lhs, rhs) {
        case (.scanning, .scanning):
            return true
        case (.detected(let lhsInfo), .detected(let rhsInfo)):
            return lhsInfo == rhsInfo
        case (.analyzing, .analyzing):
            return true
        case (.result(let lhsResult), .result(let rhsResult)):
            return lhsResult.id == rhsResult.id
        case (.noResult, .noResult):
            return true
        default:
            return false
        }
    }

    /// Info about a real-time detection.
    struct DetectionInfo: Equatable {
        /// The detected quadrilateral corners.
        let quadrilateral: Quadrilateral

        /// Decoded message from fast detection.
        let message: String

        /// Confidence score (0.0 to 1.0).
        let confidence: Double
    }
}
