import SwiftUI

/// Main camera view for real-time scanning and capture.
struct CameraView: View {

    @StateObject private var viewModel = CameraViewModel()

    var body: some View {
        ZStack {
            // Camera preview
            CameraPreview(cameraController: viewModel.cameraController)
                .ignoresSafeArea()

            // Quadrilateral overlay
            QuadrilateralOverlay(
                quadrilateral: viewModel.detectedQuadrilateral,
                color: overlayColor
            )
            .ignoresSafeArea()

            // UI overlays
            VStack {
                // Error banner
                if let error = viewModel.cameraController.error {
                    errorBanner(error)
                }

                Spacer()

                // Message overlay
                MessageOverlay(
                    message: viewModel.currentMessage,
                    confidence: viewModel.currentConfidence,
                    isAnalyzing: isAnalyzing
                )
                .padding(.bottom, 20)

                // Capture button
                CaptureButton(isProcessing: isAnalyzing) {
                    viewModel.capture()
                }
                .disabled(isAnalyzing)
                .padding(.bottom, 40)
            }
        }
        .onAppear {
            viewModel.startScanning()
        }
        .onDisappear {
            viewModel.stopScanning()
        }
        .sheet(item: resultBinding) { result in
            ResultsView(
                result: result,
                capturedImage: viewModel.capturedImage
            ) {
                viewModel.backToScanning()
            }
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK") { }
        } message: {
            Text(viewModel.errorMessage)
        }
    }

    // MARK: - Computed Properties

    private var isAnalyzing: Bool {
        if case .analyzing = viewModel.state { return true }
        return false
    }

    private var overlayColor: Color {
        switch viewModel.state {
        case .scanning:
            return .clear
        case .detected:
            return .green
        case .analyzing:
            return .yellow
        case .result, .noResult:
            return .blue
        }
    }

    private var resultBinding: Binding<DecodeResult?> {
        Binding(
            get: {
                if case .result(let result) = viewModel.state {
                    return result
                }
                if case .noResult = viewModel.state {
                    return DecodeResult(
                        message: "No watermark found",
                        confidence: 0,
                        bitAccuracy: 0,
                        totalBits: 100,
                        decodeTimeMs: 0,
                        detectedRegion: nil,
                        rawBits: []
                    )
                }
                return nil
            },
            set: { _ in
                viewModel.backToScanning()
            }
        )
    }

    // MARK: - Views

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
