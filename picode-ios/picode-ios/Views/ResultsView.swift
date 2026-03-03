import SwiftUI

/// View displaying the decode results.
struct ResultsView: View {

    /// The decode result to display.
    let result: DecodeResult

    /// Action to dismiss the view.
    let onDismiss: () -> Void

    var body: some View {
        NavigationStack {
            List {
                Section("Decoded Message") {
                    Text(result.message)
                        .font(.title3.monospaced())
                        .textSelection(.enabled)
                        .padding(.vertical, 4)
                }

                Section("Technical Data") {
                    LabeledContent("Confidence") {
                        Text(formatPercent(result.confidence))
                            .foregroundStyle(confidenceColor(result.confidence))
                    }

                    LabeledContent("Bit Accuracy") {
                        Text("\(result.bitAccuracy)/\(result.totalBits)")
                    }

                    LabeledContent("Decode Time") {
                        Text("\(Int(result.decodeTimeMs))ms")
                    }

                    if let region = result.detectedRegion {
                        LabeledContent("Region") {
                            Text(formatRegion(region))
                                .font(.caption.monospaced())
                        }
                    }
                }
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

    // MARK: - Formatting

    private func formatPercent(_ value: Double) -> String {
        String(format: "%.1f%%", value * 100)
    }

    private func formatRegion(_ rect: CGRect) -> String {
        String(format: "%.0f%%x%.0f%% @ (%.0f%%, %.0f%%)",
               rect.width * 100, rect.height * 100,
               rect.minX * 100, rect.minY * 100)
    }

    private func confidenceColor(_ value: Double) -> Color {
        if value >= 0.95 {
            return .green
        } else if value >= 0.85 {
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
            decodeTimeMs: 47,
            detectedRegion: CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.5),
            rawBits: []
        ),
        onDismiss: { }
    )
}
