import SwiftUI

/// View displaying the decode results with captured image.
struct ResultsView: View {

    let result: DecodeResult
    let capturedImage: UIImage?
    let onDismiss: () -> Void

    init(result: DecodeResult, capturedImage: UIImage? = nil, onDismiss: @escaping () -> Void) {
        self.result = result
        self.capturedImage = capturedImage
        self.onDismiss = onDismiss
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Captured image with detection overlay
                    if let image = capturedImage {
                        ZStack {
                            Image(uiImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)

                            if let region = result.detectedRegion {
                                GeometryReader { geometry in
                                    Rectangle()
                                        .stroke(Color.blue, lineWidth: 3)
                                        .frame(
                                            width: region.width * geometry.size.width,
                                            height: region.height * geometry.size.height
                                        )
                                        .position(
                                            x: (region.minX + region.width / 2) * geometry.size.width,
                                            y: (region.minY + region.height / 2) * geometry.size.height
                                        )
                                }
                            }
                        }
                        .frame(maxHeight: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .padding(.horizontal)
                    }

                    // Message card
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Decoded Message")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Text(result.message)
                            .font(.title2.monospaced())
                            .textSelection(.enabled)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(.systemGray6))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)

                    // Technical details
                    VStack(spacing: 0) {
                        detailRow("Confidence", formatPercent(result.confidence), color: confidenceColor)
                        Divider()
                        detailRow("Bit Accuracy", "\(result.bitAccuracy)/\(result.totalBits)")
                        Divider()
                        detailRow("Decode Time", "\(Int(result.decodeTimeMs))ms")
                    }
                    .background(Color(.systemGray6))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)
                }
                .padding(.vertical)
            }
            .navigationTitle("Result")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done", action: onDismiss)
                }
            }
        }
    }

    // MARK: - Components

    @ViewBuilder
    private func detailRow(_ label: String, _ value: String, color: Color = .primary) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .foregroundStyle(color)
        }
        .padding()
    }

    // MARK: - Formatting

    private func formatPercent(_ value: Double) -> String {
        String(format: "%.1f%%", value * 100)
    }

    private var confidenceColor: Color {
        if result.confidence >= 0.95 {
            return .green
        } else if result.confidence >= 0.85 {
            return .orange
        } else {
            return .red
        }
    }
}

#Preview {
    ResultsView(
        result: DecodeResult(
            message: "Hello Picode!",
            confidence: 0.942,
            bitAccuracy: 98,
            totalBits: 100,
            decodeTimeMs: 847,
            detectedRegion: CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.5),
            rawBits: []
        ),
        onDismiss: { }
    )
}
