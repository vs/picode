import SwiftUI

/// Overlay showing decoded message and confidence.
struct MessageOverlay: View {
    let message: String?
    let confidence: Double?
    let isAnalyzing: Bool

    var body: some View {
        VStack(spacing: 4) {
            if isAnalyzing {
                HStack(spacing: 8) {
                    ProgressView()
                        .tint(.white)
                    Text("Analyzing...")
                        .foregroundStyle(.white)
                }
            } else if let message, let confidence {
                Text(message)
                    .font(.headline.monospaced())
                    .foregroundStyle(.white)

                Text(String(format: "%.0f%% confidence", confidence * 100))
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.8))
            } else {
                Text("Point at encoded image")
                    .foregroundStyle(.white.opacity(0.6))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.black.opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

#Preview {
    ZStack {
        Color.gray
        VStack(spacing: 20) {
            MessageOverlay(message: nil, confidence: nil, isAnalyzing: false)
            MessageOverlay(message: "Hello World", confidence: 0.94, isAnalyzing: false)
            MessageOverlay(message: nil, confidence: nil, isAnalyzing: true)
        }
    }
}
