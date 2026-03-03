import SwiftUI

/// Main camera view for capturing and decoding picode watermarks.
struct CameraView: View {

    @StateObject private var viewModel = CameraViewModel()

    var body: some View {
        ZStack {
            // Camera preview fills screen
            CameraPreview(cameraController: viewModel.cameraController)
                .ignoresSafeArea()

            // Dark overlay at top for status
            VStack {
                if let error = viewModel.cameraController.error {
                    errorBanner(error)
                }
                Spacer()
            }

            // Capture controls at bottom
            VStack {
                Spacer()

                CaptureButton(isProcessing: viewModel.isDecoding) {
                    viewModel.captureAndDecode()
                }
                .padding(.bottom, 40)
            }
        }
        .onAppear {
            viewModel.cameraController.setupCamera()
        }
        .onDisappear {
            viewModel.cameraController.stopCamera()
        }
        .sheet(item: $viewModel.decodeResult) { result in
            ResultsView(result: result) {
                viewModel.dismissResults()
            }
        }
        .alert("Decode Failed", isPresented: $viewModel.showError) {
            Button("OK") { }
        } message: {
            Text(viewModel.errorMessage)
        }
    }

    @ViewBuilder
    private func errorBanner(_ message: String) -> some View {
        Text(message)
            .font(.subheadline)
            .foregroundStyle(.white)
            .padding()
            .frame(maxWidth: .infinity)
            .background(.red.opacity(0.8))
    }
}

#Preview {
    CameraView()
}
