import SwiftUI

/// Camera capture button with processing state.
struct CaptureButton: View {

    /// Whether a capture/decode operation is in progress.
    let isProcessing: Bool

    /// Action to perform when tapped.
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                // Outer ring
                Circle()
                    .stroke(Color.white, lineWidth: 4)
                    .frame(width: 72, height: 72)

                // Inner circle or spinner
                if isProcessing {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.white)
                        .scaleEffect(1.2)
                } else {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 60, height: 60)
                }
            }
        }
        .disabled(isProcessing)
        .animation(.easeInOut(duration: 0.2), value: isProcessing)
    }
}

#Preview("Ready") {
    ZStack {
        Color.black
        CaptureButton(isProcessing: false) { }
    }
}

#Preview("Processing") {
    ZStack {
        Color.black
        CaptureButton(isProcessing: true) { }
    }
}
