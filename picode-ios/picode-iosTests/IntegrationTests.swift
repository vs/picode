import XCTest
@testable import picode_ios

final class IntegrationTests: XCTestCase {

    func testFastDetector_init_succeeds() {
        let detector = FastDetector()
        XCTAssertNotNil(detector)
    }

    func testSlowDetector_init_succeeds() {
        let detector = SlowDetector()
        XCTAssertNotNil(detector)
    }

    @MainActor
    func testCameraViewModel_stateTransitions() async {
        let vm = CameraViewModel()

        // Initial state
        if case .scanning = vm.state { } else {
            XCTFail("Should start in scanning state")
        }

        // Back to scanning
        vm.backToScanning()
        if case .scanning = vm.state { } else {
            XCTFail("Should be in scanning state after back")
        }
    }

    @MainActor
    func testCameraViewModel_detectThenCapture() async {
        let mockDecoder = MockDecoder()
        let vm = CameraViewModel(decoder: mockDecoder)

        // Simulate detection
        let quad = Quadrilateral(
            topLeft: CGPoint(x: 0.1, y: 0.1),
            topRight: CGPoint(x: 0.9, y: 0.1),
            bottomRight: CGPoint(x: 0.9, y: 0.9),
            bottomLeft: CGPoint(x: 0.1, y: 0.9)
        )

        let result = DecodeResult(
            message: "Integration test",
            confidence: 0.92,
            bitAccuracy: 92,
            totalBits: 100,
            decodeTimeMs: 25,
            detectedRegion: quad.boundingRect,
            rawBits: []
        )

        vm.handleDetection(result, quadrilateral: quad)

        if case .detected(let info) = vm.state {
            XCTAssertEqual(info.message, "Integration test")
            XCTAssertEqual(info.confidence, 0.92)
        } else {
            XCTFail("Should be in detected state")
        }

        XCTAssertNotNil(vm.detectedQuadrilateral)
        XCTAssertEqual(vm.currentMessage, "Integration test")

        // Return to scanning
        vm.backToScanning()
        XCTAssertNil(vm.detectedQuadrilateral)
        XCTAssertNil(vm.currentMessage)
    }
}
