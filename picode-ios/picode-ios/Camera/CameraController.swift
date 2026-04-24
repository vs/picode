import AVFoundation
import UIKit

/// Manages the camera capture session and photo capture.
///
/// This is a UIKit controller that wraps AVFoundation for use with SwiftUI.
final class CameraController: NSObject, ObservableObject {

    /// The capture session for camera input/output.
    let captureSession = AVCaptureSession()

    /// Photo output for capturing still images.
    private let photoOutput = AVCapturePhotoOutput()

    /// Video output for real-time frame processing.
    private let videoOutput = AVCaptureVideoDataOutput()

    /// Queue for video frame processing.
    private let videoQueue = DispatchQueue(label: "com.picode.videoQueue", qos: .userInteractive)

    /// Delegate for receiving video frames.
    weak var videoDelegate: AVCaptureVideoDataOutputSampleBufferDelegate? {
        didSet {
            videoOutput.setSampleBufferDelegate(videoDelegate, queue: videoQueue)
        }
    }

    /// Completion handler for photo capture.
    private var photoCaptureCompletion: ((UIImage?) -> Void)?

    /// Whether the camera is currently configured and running.
    @Published private(set) var isReady = false

    /// Error message if setup fails.
    @Published private(set) var error: String?

    override init() {
        super.init()
    }

    /// Configure and start the camera session.
    ///
    /// Call this when the camera view appears.
    func setupCamera() {
        // Run setup on background queue to avoid blocking UI
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.configureSession()
        }
    }

    /// Stop the camera session.
    ///
    /// Call this when the camera view disappears.
    func stopCamera() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    /// Capture a still photo.
    ///
    /// - Parameter completion: Called with the captured image, or nil if capture failed.
    func capturePhoto(completion: @escaping (UIImage?) -> Void) {
        guard captureSession.isRunning else {
            completion(nil)
            return
        }

        photoCaptureCompletion = completion

        let settings = AVCapturePhotoSettings()
        settings.flashMode = .off

        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    // MARK: - Private

    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .photo

        // Add camera input
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            handleSetupError("Could not access camera")
            return
        }

        guard captureSession.canAddInput(input) else {
            handleSetupError("Could not add camera input")
            return
        }
        captureSession.addInput(input)

        // Add photo output
        guard captureSession.canAddOutput(photoOutput) else {
            handleSetupError("Could not add photo output")
            return
        }
        captureSession.addOutput(photoOutput)

        // Add video output for real-time processing
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard captureSession.canAddOutput(videoOutput) else {
            handleSetupError("Could not add video output")
            return
        }
        captureSession.addOutput(videoOutput)

        // Set video orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90  // Portrait
        }

        captureSession.commitConfiguration()
        captureSession.startRunning()

        DispatchQueue.main.async { [weak self] in
            self?.isReady = true
        }
    }

    private func handleSetupError(_ message: String) {
        captureSession.commitConfiguration()
        DispatchQueue.main.async { [weak self] in
            self?.error = message
        }
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraController: AVCapturePhotoCaptureDelegate {

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        defer { photoCaptureCompletion = nil }

        if let error = error {
            print("Photo capture error: \(error.localizedDescription)")
            photoCaptureCompletion?(nil)
            return
        }

        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            photoCaptureCompletion?(nil)
            return
        }

        photoCaptureCompletion?(image)
    }
}
