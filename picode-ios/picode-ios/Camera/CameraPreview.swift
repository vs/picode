import AVFoundation
import SwiftUI

/// SwiftUI wrapper for AVCaptureVideoPreviewLayer.
///
/// Displays the live camera feed in a SwiftUI view.
struct CameraPreview: UIViewRepresentable {

    /// The camera controller providing the capture session.
    let cameraController: CameraController

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = cameraController.captureSession
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // Session is already connected in makeUIView
    }
}

/// UIView subclass that uses AVCaptureVideoPreviewLayer as its backing layer.
final class PreviewView: UIView {

    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
        super.layoutSubviews()

        // Update preview orientation based on device orientation
        if let connection = previewLayer.connection {
            let orientation = UIDevice.current.orientation

            if connection.isVideoRotationAngleSupported(0) {
                switch orientation {
                case .portrait:
                    connection.videoRotationAngle = 90
                case .landscapeLeft:
                    connection.videoRotationAngle = 180
                case .landscapeRight:
                    connection.videoRotationAngle = 0
                case .portraitUpsideDown:
                    connection.videoRotationAngle = 270
                default:
                    connection.videoRotationAngle = 90
                }
            }
        }
    }
}
