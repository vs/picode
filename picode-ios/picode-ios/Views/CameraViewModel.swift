import SwiftUI

/// View model for the camera view, managing capture and decode state.
@MainActor
final class CameraViewModel: ObservableObject {

    /// Camera controller for capturing photos.
    let cameraController = CameraController()

    /// Decoder for extracting watermarks (injectable for testing).
    private let decoder: PicodeDecoder

    /// Whether a decode operation is in progress.
    @Published var isDecoding = false

    /// The most recent decode result (triggers results sheet).
    @Published var decodeResult: DecodeResult?

    /// Whether to show the error alert.
    @Published var showError = false

    /// Error message to display.
    @Published private(set) var errorMessage = ""

    /// Create a view model with the specified decoder.
    ///
    /// - Parameter decoder: The decoder to use. Defaults to StubDecoder.
    init(decoder: PicodeDecoder = StubDecoder()) {
        self.decoder = decoder
    }

    /// Capture a photo and decode it.
    func captureAndDecode() {
        guard !isDecoding else { return }
        isDecoding = true

        cameraController.capturePhoto { [weak self] image in
            guard let self else { return }

            guard let image else {
                Task { @MainActor in
                    self.handleError("Failed to capture photo")
                }
                return
            }

            Task {
                await self.decode(image: image)
            }
        }
    }

    /// Dismiss the results sheet.
    func dismissResults() {
        decodeResult = nil
    }

    // MARK: - Private

    private func decode(image: UIImage) async {
        do {
            let result = try await decoder.decode(image: image)
            decodeResult = result
        } catch {
            handleError(error.localizedDescription)
        }
        isDecoding = false
    }

    private func handleError(_ message: String) {
        errorMessage = message
        showError = true
        isDecoding = false
    }
}
