import CoreGraphics

/// A quadrilateral defined by four corner points.
struct Quadrilateral: Equatable, Sendable {
    let topLeft: CGPoint
    let topRight: CGPoint
    let bottomRight: CGPoint
    let bottomLeft: CGPoint

    /// Create from array of 8 normalized coordinates [x1,y1,x2,y2,x3,y3,x4,y4].
    init?(from corners: [Float]) {
        guard corners.count == 8 else { return nil }

        self.topLeft = CGPoint(x: CGFloat(corners[0]), y: CGFloat(corners[1]))
        self.topRight = CGPoint(x: CGFloat(corners[2]), y: CGFloat(corners[3]))
        self.bottomRight = CGPoint(x: CGFloat(corners[4]), y: CGFloat(corners[5]))
        self.bottomLeft = CGPoint(x: CGFloat(corners[6]), y: CGFloat(corners[7]))
    }

    init(topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint) {
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft
    }

    /// Convert to pixel coordinates given image size.
    func toPixelCoordinates(imageSize: CGSize) -> Quadrilateral {
        Quadrilateral(
            topLeft: CGPoint(x: topLeft.x * imageSize.width, y: topLeft.y * imageSize.height),
            topRight: CGPoint(x: topRight.x * imageSize.width, y: topRight.y * imageSize.height),
            bottomRight: CGPoint(x: bottomRight.x * imageSize.width, y: bottomRight.y * imageSize.height),
            bottomLeft: CGPoint(x: bottomLeft.x * imageSize.width, y: bottomLeft.y * imageSize.height)
        )
    }

    /// Get bounding rect.
    var boundingRect: CGRect {
        let xs = [topLeft.x, topRight.x, bottomRight.x, bottomLeft.x]
        let ys = [topLeft.y, topRight.y, bottomRight.y, bottomLeft.y]

        let minX = xs.min() ?? 0
        let maxX = xs.max() ?? 1
        let minY = ys.min() ?? 0
        let maxY = ys.max() ?? 1

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    /// All four corners as array.
    var corners: [CGPoint] {
        [topLeft, topRight, bottomRight, bottomLeft]
    }
}
