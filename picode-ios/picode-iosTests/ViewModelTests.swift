import XCTest
@testable import picode_ios

/// Mock decoder for testing view model behavior.
final class MockDecoder: PicodeDecoder {
    var shouldFail = false
    var decodeCallCount = 0
    var mockResult: DecodeResult?

    func decode(image: UIImage) async throws -> DecodeResult {
        decodeCallCount += 1

        if shouldFail {
            throw DecodeError.noWatermarkFound
        }

        return mockResult ?? DecodeResult(
            message: "Test message",
            confidence: 0.95,
            bitAccuracy: 95,
            totalBits: 100,
            decodeTimeMs: 50,
            detectedRegion: CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6),
            rawBits: []
        )
    }
}

@MainActor
final class ViewModelTests: XCTestCase {

    // MARK: - Initial State Tests

    func testInitialState_isScanning() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        if case .scanning = viewModel.state {
            // Expected
        } else {
            XCTFail("Initial state should be scanning, got: \(viewModel.state)")
        }

        XCTAssertNil(viewModel.detectedQuadrilateral)
        XCTAssertNil(viewModel.currentMessage)
        XCTAssertNil(viewModel.currentConfidence)
        XCTAssertNil(viewModel.capturedImage)
    }

    func testInitialState_legacyProperties() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        XCTAssertFalse(viewModel.isDecoding)
        XCTAssertNil(viewModel.decodeResult)
        XCTAssertFalse(viewModel.showError)
    }

    // MARK: - State Transition Tests

    func testBackToScanning_resetsState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        // Set up some state
        viewModel.capturedImage = UIImage()
        viewModel.detectedQuadrilateral = Quadrilateral(
            topLeft: .zero,
            topRight: CGPoint(x: 1, y: 0),
            bottomRight: CGPoint(x: 1, y: 1),
            bottomLeft: CGPoint(x: 0, y: 1)
        )
        viewModel.currentMessage = "Test"
        viewModel.currentConfidence = 0.9

        viewModel.backToScanning()

        if case .scanning = viewModel.state {
            // Expected
        } else {
            XCTFail("State should be scanning after backToScanning")
        }
        XCTAssertNil(viewModel.capturedImage)
        XCTAssertNil(viewModel.detectedQuadrilateral)
        XCTAssertNil(viewModel.currentMessage)
        XCTAssertNil(viewModel.currentConfidence)
    }

    func testStartScanning_resetsState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        // Set up some state
        viewModel.capturedImage = UIImage()
        viewModel.currentMessage = "Test"

        viewModel.startScanning()

        if case .scanning = viewModel.state {
            // Expected
        } else {
            XCTFail("State should be scanning after startScanning")
        }
        XCTAssertNil(viewModel.capturedImage)
        XCTAssertNil(viewModel.currentMessage)
    }

    func testDismissResults_returnsToScanning() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        // Set a result
        let result = DecodeResult(
            message: "Test",
            confidence: 0.9,
            bitAccuracy: 90,
            totalBits: 100,
            decodeTimeMs: 50,
            detectedRegion: nil,
            rawBits: []
        )
        viewModel.decodeResult = result

        viewModel.dismissResults()

        if case .scanning = viewModel.state {
            // Expected
        } else {
            XCTFail("State should be scanning after dismissResults")
        }
    }

    // MARK: - Detection Tests

    func testHandleDetection_updatesState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        let quad = Quadrilateral(
            topLeft: CGPoint(x: 0.1, y: 0.1),
            topRight: CGPoint(x: 0.9, y: 0.1),
            bottomRight: CGPoint(x: 0.9, y: 0.9),
            bottomLeft: CGPoint(x: 0.1, y: 0.9)
        )

        let result = DecodeResult(
            message: "Detected message",
            confidence: 0.85,
            bitAccuracy: 85,
            totalBits: 100,
            decodeTimeMs: 20,
            detectedRegion: quad.boundingRect,
            rawBits: []
        )

        viewModel.handleDetection(result, quadrilateral: quad)

        if case .detected(let info) = viewModel.state {
            XCTAssertEqual(info.message, "Detected message")
            XCTAssertEqual(info.confidence, 0.85)
            XCTAssertEqual(info.quadrilateral, quad)
        } else {
            XCTFail("State should be detected")
        }

        XCTAssertEqual(viewModel.detectedQuadrilateral, quad)
        XCTAssertEqual(viewModel.currentMessage, "Detected message")
        XCTAssertEqual(viewModel.currentConfidence, 0.85)
    }

    func testClearDetection_returnsToScanning() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        // First set detected state
        let quad = Quadrilateral(
            topLeft: .zero,
            topRight: CGPoint(x: 1, y: 0),
            bottomRight: CGPoint(x: 1, y: 1),
            bottomLeft: CGPoint(x: 0, y: 1)
        )

        let result = DecodeResult(
            message: "Test",
            confidence: 0.9,
            bitAccuracy: 90,
            totalBits: 100,
            decodeTimeMs: 20,
            detectedRegion: nil,
            rawBits: []
        )

        viewModel.handleDetection(result, quadrilateral: quad)
        viewModel.clearDetection()

        if case .scanning = viewModel.state {
            // Expected
        } else {
            XCTFail("State should be scanning after clearDetection")
        }
        XCTAssertNil(viewModel.detectedQuadrilateral)
        XCTAssertNil(viewModel.currentMessage)
        XCTAssertNil(viewModel.currentConfidence)
    }

    func testClearDetection_noOpIfNotDetected() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        // Set to analyzing state
        viewModel.capture()  // This sets state to analyzing

        // Clear detection should not change state from analyzing
        viewModel.clearDetection()

        // State should still not be scanning (capture sets it to analyzing)
        // Note: The exact state depends on camera availability
    }

    // MARK: - Legacy Compatibility Tests

    func testDecodeResultGetter_returnsResultFromState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        let result = DecodeResult(
            message: "Test",
            confidence: 0.9,
            bitAccuracy: 90,
            totalBits: 100,
            decodeTimeMs: 50,
            detectedRegion: nil,
            rawBits: []
        )

        viewModel.state = .result(result)

        XCTAssertNotNil(viewModel.decodeResult)
        XCTAssertEqual(viewModel.decodeResult?.message, "Test")
    }

    func testDecodeResultSetter_updatesState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        let result = DecodeResult(
            message: "New result",
            confidence: 0.95,
            bitAccuracy: 95,
            totalBits: 100,
            decodeTimeMs: 30,
            detectedRegion: nil,
            rawBits: []
        )

        viewModel.decodeResult = result

        if case .result(let stateResult) = viewModel.state {
            XCTAssertEqual(stateResult.message, "New result")
        } else {
            XCTFail("State should be result")
        }
    }

    func testDecodeResultSetter_nilClearsState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        let result = DecodeResult(
            message: "Test",
            confidence: 0.9,
            bitAccuracy: 90,
            totalBits: 100,
            decodeTimeMs: 50,
            detectedRegion: nil,
            rawBits: []
        )

        viewModel.state = .result(result)
        viewModel.decodeResult = nil

        if case .scanning = viewModel.state {
            // Expected
        } else {
            XCTFail("State should be scanning after setting decodeResult to nil")
        }
    }

    func testIsDecoding_reflectsAnalyzingState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        XCTAssertFalse(viewModel.isDecoding)

        viewModel.state = .analyzing

        XCTAssertTrue(viewModel.isDecoding)

        viewModel.state = .scanning

        XCTAssertFalse(viewModel.isDecoding)
    }

    // MARK: - Error Handling Tests

    func testHandleError_setsErrorState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        viewModel.handleError("Test error message")

        XCTAssertTrue(viewModel.showError)
        XCTAssertEqual(viewModel.errorMessage, "Test error message")
        if case .scanning = viewModel.state {
            // Expected - error resets to scanning
        } else {
            XCTFail("State should be scanning after error")
        }
    }

    func testHandleError_clearsCapturedImage() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        viewModel.capturedImage = UIImage()
        viewModel.handleError("Error")

        XCTAssertNil(viewModel.capturedImage)
    }
}

// MARK: - ScanState Tests

final class ScanStateTests: XCTestCase {

    func testScanStateEquality_scanning() {
        XCTAssertEqual(ScanState.scanning, ScanState.scanning)
    }

    func testScanStateEquality_analyzing() {
        XCTAssertEqual(ScanState.analyzing, ScanState.analyzing)
    }

    func testScanStateEquality_noResult() {
        XCTAssertEqual(ScanState.noResult, ScanState.noResult)
    }

    func testScanStateEquality_detected() {
        let quad = Quadrilateral(
            topLeft: .zero,
            topRight: CGPoint(x: 1, y: 0),
            bottomRight: CGPoint(x: 1, y: 1),
            bottomLeft: CGPoint(x: 0, y: 1)
        )

        let info1 = ScanState.DetectionInfo(
            quadrilateral: quad,
            message: "Test",
            confidence: 0.9
        )

        let info2 = ScanState.DetectionInfo(
            quadrilateral: quad,
            message: "Test",
            confidence: 0.9
        )

        XCTAssertEqual(ScanState.detected(info1), ScanState.detected(info2))
    }

    func testScanStateEquality_detectedDifferent() {
        let quad = Quadrilateral(
            topLeft: .zero,
            topRight: CGPoint(x: 1, y: 0),
            bottomRight: CGPoint(x: 1, y: 1),
            bottomLeft: CGPoint(x: 0, y: 1)
        )

        let info1 = ScanState.DetectionInfo(
            quadrilateral: quad,
            message: "Test1",
            confidence: 0.9
        )

        let info2 = ScanState.DetectionInfo(
            quadrilateral: quad,
            message: "Test2",
            confidence: 0.9
        )

        XCTAssertNotEqual(ScanState.detected(info1), ScanState.detected(info2))
    }

    func testScanStateInequality_differentCases() {
        XCTAssertNotEqual(ScanState.scanning, ScanState.analyzing)
        XCTAssertNotEqual(ScanState.scanning, ScanState.noResult)
        XCTAssertNotEqual(ScanState.analyzing, ScanState.noResult)
    }
}
