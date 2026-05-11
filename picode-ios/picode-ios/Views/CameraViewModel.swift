import SwiftUI
import AVFoundation
import CoreImage

/// View model for the camera view, managing capture and decode state.
@MainActor
final class CameraViewModel: NSObject, ObservableObject {

    // MARK: - Published State

    /// Current scanning state.
    @Published var state: ScanState = .scanning

    /// Detected quadrilateral for overlay display.
    @Published var detectedQuadrilateral: Quadrilateral?

    /// Current decoded message (for real-time display).
    @Published var currentMessage: String?

    /// Current confidence score (for real-time display).
    @Published var currentConfidence: Double?

    /// Captured image for display in results.
    @Published var capturedImage: UIImage?

    /// Whether to show the error alert.
    @Published var showError = false

    /// Error message to display.
    @Published private(set) var errorMessage = ""

    // MARK: - Services

    /// Camera controller for capturing photos.
    let cameraController = CameraController()

    /// Fast detector for real-time watermark detection.
    private let fastDetector: FastDetector

    /// Slow detector for high-quality analysis (placeholder for now).
    private let slowDetector: PicodeDecoder

    // MARK: - Processing State

    /// Whether a frame is currently being processed.
    private var isProcessingFrame = false

    /// Core Image context for efficient image processing.
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Throttle interval for frame processing (seconds).
    private let frameProcessingInterval: TimeInterval = 0.1  // 10 FPS max

    /// Minimum confidence threshold for detection.
    private let detectionConfidenceThreshold: Double = 0.5

    /// Last time a frame was processed.
    private var lastFrameTime: CFAbsoluteTime = 0

    // MARK: - Legacy Properties (for backward compatibility)

    /// Whether a decode operation is in progress.
    var isDecoding: Bool {
        if case .analyzing = state {
            return true
        }
        return false
    }

    /// The most recent decode result (for backward compatibility).
    var decodeResult: DecodeResult? {
        get {
            if case .result(let result) = state {
                return result
            }
            return nil
        }
        set {
            if let result = newValue {
                state = .result(result)
            } else if case .result = state {
                state = .scanning
            }
        }
    }

    // MARK: - Initialization

    /// Create a view model with the specified detectors.
    ///
    /// - Parameters:
    ///   - fastDetector: Detector for real-time frame analysis.
    ///   - slowDetector: Decoder for high-quality analysis. Defaults to FastDetector.
    init(fastDetector: FastDetector = FastDetector(), slowDetector: PicodeDecoder? = nil) {
        self.fastDetector = fastDetector
        self.slowDetector = slowDetector ?? fastDetector
        super.init()
    }

    /// Create a view model with a custom decoder (for testing/backward compatibility).
    ///
    /// - Parameter decoder: The decoder to use for both fast and slow detection.
    convenience init(decoder: PicodeDecoder) {
        if let fastDetector = decoder as? FastDetector {
            self.init(fastDetector: fastDetector, slowDetector: decoder)
        } else {
            self.init(fastDetector: FastDetector(), slowDetector: decoder)
        }
    }

    // MARK: - Scanning Control

    /// Start scanning for watermarks.
    func startScanning() {
        state = .scanning
        detectedQuadrilateral = nil
        currentMessage = nil
        currentConfidence = nil
        capturedImage = nil

        // Set up video delegate for real-time processing
        cameraController.videoDelegate = self
        cameraController.setupCamera()
    }

    /// Stop scanning.
    func stopScanning() {
        cameraController.videoDelegate = nil
        cameraController.stopCamera()
    }

    /// Reset state to scanning mode.
    func backToScanning() {
        state = .scanning
        detectedQuadrilateral = nil
        currentMessage = nil
        currentConfidence = nil
        capturedImage = nil

        // Re-enable video processing
        cameraController.videoDelegate = self
    }

    // MARK: - Capture and Analyze

    /// Capture a photo and analyze it.
    func capture() {
        guard !isDecoding else { return }

        // Disable video processing during capture
        cameraController.videoDelegate = nil
        state = .analyzing

        cameraController.capturePhoto { [weak self] image in
            guard let self else { return }

            guard let image else {
                Task { @MainActor in
                    self.handleError("Failed to capture photo")
                }
                return
            }

            Task { @MainActor in
                self.capturedImage = image
                await self.analyzeCapture(image)
            }
        }
    }

    /// Capture and decode (backward compatibility alias for capture).
    func captureAndDecode() {
        capture()
    }

    /// Dismiss the results and return to scanning.
    func dismissResults() {
        backToScanning()
    }

    // MARK: - Detection Handling

    /// Handle a detection result from real-time processing.
    func handleDetection(_ result: DecodeResult, quadrilateral: Quadrilateral) {
        detectedQuadrilateral = quadrilateral
        currentMessage = result.message
        currentConfidence = result.confidence

        let detectionInfo = ScanState.DetectionInfo(
            quadrilateral: quadrilateral,
            message: result.message,
            confidence: result.confidence
        )
        state = .detected(detectionInfo)
    }

    /// Clear detection (watermark lost from view).
    func clearDetection() {
        guard case .detected = state else { return }

        state = .scanning
        detectedQuadrilateral = nil
        currentMessage = nil
        currentConfidence = nil
    }

    // MARK: - Private Methods

    /// Analyze a captured image using the slow detector.
    private func analyzeCapture(_ image: UIImage) async {
        do {
            let result = try await slowDetector.decode(image: image)
            state = .result(result)
        } catch {
            if case DecodeError.noWatermarkFound = error {
                state = .noResult
            } else {
                handleError(error.localizedDescription)
            }
        }
    }

    /// Handle an error.
    func handleError(_ message: String) {
        errorMessage = message
        showError = true
        state = .scanning
        capturedImage = nil

        // Re-enable video processing
        cameraController.videoDelegate = self
    }

    /// Process a video frame for detection.
    ///
    /// - Parameter ciImage: The CIImage extracted from the video frame.
    private func processFrame(_ ciImage: CIImage) {
        // Convert to UIImage for decoder
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage)

        // Run detection
        Task {
            do {
                let result = try await fastDetector.decode(image: image)

                // Check if we got a valid detection
                if result.confidence > detectionConfidenceThreshold, let region = result.detectedRegion {
                    // Create quadrilateral from detected region
                    let quad = Quadrilateral(
                        topLeft: CGPoint(x: region.minX, y: region.minY),
                        topRight: CGPoint(x: region.maxX, y: region.minY),
                        bottomRight: CGPoint(x: region.maxX, y: region.maxY),
                        bottomLeft: CGPoint(x: region.minX, y: region.maxY)
                    )

                    await MainActor.run {
                        self.handleDetection(result, quadrilateral: quad)
                        self.isProcessingFrame = false
                    }
                } else {
                    await MainActor.run {
                        self.clearDetection()
                        self.isProcessingFrame = false
                    }
                }
            } catch {
                await MainActor.run {
                    self.clearDetection()
                    self.isProcessingFrame = false
                }
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {

    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let currentTime = CFAbsoluteTimeGetCurrent()

        // Extract pixel buffer synchronously before it's recycled by AVFoundation
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        Task { @MainActor in
            // Check throttling and state on main actor
            guard !self.isProcessingFrame else { return }
            guard currentTime - self.lastFrameTime >= self.frameProcessingInterval else { return }

            // Only process if we're in scanning or detected state
            switch self.state {
            case .scanning, .detected:
                self.isProcessingFrame = true
                self.lastFrameTime = currentTime
                self.processFrame(ciImage)
            default:
                break
            }
        }
    }

    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Frames are being dropped - processing is too slow
        // This is expected when running on simulator or during heavy load
    }
}
