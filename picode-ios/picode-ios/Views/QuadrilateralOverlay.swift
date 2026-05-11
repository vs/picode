import SwiftUI

/// Draws a quadrilateral overlay on the camera preview.
struct QuadrilateralOverlay: View {
    let quadrilateral: Quadrilateral?
    let color: Color
    let lineWidth: CGFloat

    init(
        quadrilateral: Quadrilateral?,
        color: Color = .green,
        lineWidth: CGFloat = 3
    ) {
        self.quadrilateral = quadrilateral
        self.color = color
        self.lineWidth = lineWidth
    }

    var body: some View {
        GeometryReader { geometry in
            if let quad = quadrilateral {
                let pixelQuad = quad.toPixelCoordinates(imageSize: geometry.size)

                Path { path in
                    path.move(to: pixelQuad.topLeft)
                    path.addLine(to: pixelQuad.topRight)
                    path.addLine(to: pixelQuad.bottomRight)
                    path.addLine(to: pixelQuad.bottomLeft)
                    path.closeSubpath()
                }
                .stroke(color, lineWidth: lineWidth)

                // Corner markers
                let pixelQuadCorners = pixelQuad.corners
                ForEach(0..<pixelQuadCorners.count, id: \.self) { index in
                    Circle()
                        .fill(color)
                        .frame(width: 12, height: 12)
                        .position(pixelQuadCorners[index])
                }
            }
        }
    }
}

#Preview {
    ZStack {
        Color.black
        QuadrilateralOverlay(
            quadrilateral: Quadrilateral(
                topLeft: CGPoint(x: 0.2, y: 0.2),
                topRight: CGPoint(x: 0.8, y: 0.25),
                bottomRight: CGPoint(x: 0.75, y: 0.8),
                bottomLeft: CGPoint(x: 0.15, y: 0.75)
            )
        )
    }
}
