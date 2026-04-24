import XCTest
@testable import picode_ios

final class QuadrilateralTests: XCTestCase {

    func testInit_fromArray_validInput() {
        let corners: [Float] = [0.1, 0.2, 0.8, 0.2, 0.8, 0.9, 0.1, 0.9]

        let quad = Quadrilateral(from: corners)

        XCTAssertNotNil(quad)
        XCTAssertEqual(Double(quad!.topLeft.x), 0.1, accuracy: 0.001)
        XCTAssertEqual(Double(quad!.topLeft.y), 0.2, accuracy: 0.001)
    }

    func testInit_fromArray_invalidInput() {
        let corners: [Float] = [0.1, 0.2, 0.8]  // Too few

        let quad = Quadrilateral(from: corners)

        XCTAssertNil(quad)
    }

    func testToPixelCoordinates() {
        let quad = Quadrilateral(
            topLeft: CGPoint(x: 0.1, y: 0.2),
            topRight: CGPoint(x: 0.9, y: 0.2),
            bottomRight: CGPoint(x: 0.9, y: 0.8),
            bottomLeft: CGPoint(x: 0.1, y: 0.8)
        )

        let pixelQuad = quad.toPixelCoordinates(imageSize: CGSize(width: 100, height: 200))

        XCTAssertEqual(Double(pixelQuad.topLeft.x), 10, accuracy: 0.001)
        XCTAssertEqual(Double(pixelQuad.topLeft.y), 40, accuracy: 0.001)
    }

    func testBoundingRect() {
        let quad = Quadrilateral(
            topLeft: CGPoint(x: 0.1, y: 0.2),
            topRight: CGPoint(x: 0.9, y: 0.2),
            bottomRight: CGPoint(x: 0.9, y: 0.8),
            bottomLeft: CGPoint(x: 0.1, y: 0.8)
        )

        let rect = quad.boundingRect

        XCTAssertEqual(Double(rect.minX), 0.1, accuracy: 0.001)
        XCTAssertEqual(Double(rect.minY), 0.2, accuracy: 0.001)
        XCTAssertEqual(Double(rect.width), 0.8, accuracy: 0.001)
        XCTAssertEqual(Double(rect.height), 0.6, accuracy: 0.001)
    }
}
