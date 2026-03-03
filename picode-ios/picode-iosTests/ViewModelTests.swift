import XCTest
@testable import picode_ios

/// Mock decoder for testing view model behavior.
final class MockDecoder: PicodeDecoder {
    var shouldFail = false
    var decodeCallCount = 0

    func decode(image: UIImage) async throws -> DecodeResult {
        decodeCallCount += 1

        if shouldFail {
            throw DecodeError.noWatermarkFound
        }

        return DecodeResult(
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

    func testInitialState() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        XCTAssertFalse(viewModel.isDecoding)
        XCTAssertNil(viewModel.decodeResult)
        XCTAssertFalse(viewModel.showError)
    }

    func testDismissResultsClearsResult() {
        let viewModel = CameraViewModel(decoder: MockDecoder())

        // Manually set a result
        viewModel.decodeResult = DecodeResult(
            message: "Test",
            confidence: 0.9,
            bitAccuracy: 90,
            totalBits: 100,
            decodeTimeMs: 50,
            detectedRegion: nil,
            rawBits: []
        )

        viewModel.dismissResults()

        XCTAssertNil(viewModel.decodeResult)
    }
}
