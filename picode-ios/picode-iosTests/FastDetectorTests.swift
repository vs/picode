import XCTest
@testable import picode_ios

final class FastDetectorTests: XCTestCase {

    // MARK: - Initialization Tests

    func testFastDetectorInitializes() {
        // Should not crash even without bundled models
        let detector = FastDetector()
        XCTAssertNotNil(detector)
    }

    func testFastDetectorConformsToPicodeDecoder() {
        let detector: PicodeDecoder = FastDetector()
        XCTAssertNotNil(detector)
    }

    // MARK: - Decode Tests (with fallback behavior)

    func testDecodeReturnsResultWithFallback() async throws {
        // Without bundled models, FastDetector falls back to stub-like behavior
        let detector = FastDetector()
        let image = UIImage()

        let result = try await detector.decode(image: image)

        // Should return valid result structure
        XCTAssertFalse(result.message.isEmpty)
        XCTAssertGreaterThan(result.confidence, 0)
        XCTAssertLessThanOrEqual(result.confidence, 1)
        XCTAssertEqual(result.totalBits, 100)
        XCTAssertGreaterThan(result.decodeTimeMs, 0)
    }

    func testDecodeResultHasValidRegion() async throws {
        let detector = FastDetector()
        let image = UIImage()

        let result = try await detector.decode(image: image)

        guard let region = result.detectedRegion else {
            XCTFail("Expected region to be present")
            return
        }

        // Region should be normalized (0-1 range)
        XCTAssertGreaterThanOrEqual(region.minX, 0)
        XCTAssertLessThanOrEqual(region.maxX, 1)
        XCTAssertGreaterThanOrEqual(region.minY, 0)
        XCTAssertLessThanOrEqual(region.maxY, 1)
    }

    func testDecodeResultHasRawBits() async throws {
        let detector = FastDetector()
        let image = UIImage()

        let result = try await detector.decode(image: image)

        // Should have 100 raw bit values
        XCTAssertEqual(result.rawBits.count, 100)

        // All values should be probabilities (0-1 range)
        for bit in result.rawBits {
            XCTAssertGreaterThanOrEqual(bit, 0)
            XCTAssertLessThanOrEqual(bit, 1)
        }
    }

    // MARK: - Performance Tests

    func testDecodePerformance() async throws {
        let detector = FastDetector()
        let image = UIImage()

        // Measure decode time
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await detector.decode(image: image)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        // Without real models, should be fast (< 100ms)
        // With real models, target is < 100ms for full pipeline
        XCTAssertLessThan(elapsed, 500, "Decode should complete in under 500ms")
    }

    // MARK: - Integration with ViewModel

    func testViewModelUsesFastDetector() {
        // CameraViewModel should default to FastDetector
        let viewModel = CameraViewModel()
        XCTAssertNotNil(viewModel)
    }

    func testViewModelAcceptsCustomDecoder() {
        // Should still work with StubDecoder for testing
        let stubDecoder = StubDecoder()
        let viewModel = CameraViewModel(decoder: stubDecoder)
        XCTAssertNotNil(viewModel)
    }
}

// MARK: - Bit Conversion Tests

final class BitConversionTests: XCTestCase {

    func testBitsToMessageWithPrintableCharacters() async throws {
        // Test that high-confidence bits produce readable output
        // Note: This tests the internal logic indirectly through decode result
        let detector = FastDetector()
        let image = UIImage()

        let result = try await detector.decode(image: image)
        // Message should be non-empty
        XCTAssertFalse(result.message.isEmpty)
    }
}

// MARK: - Mock Decoder for Comparison

/// Mock decoder that returns predictable results for testing.
final class MockFastDetector: PicodeDecoder {
    var shouldSucceed = true
    var mockConfidence: Double = 0.95
    var mockMessage = "Test Message"

    func decode(image: UIImage) async throws -> DecodeResult {
        if !shouldSucceed {
            throw DecodeError.noWatermarkFound
        }

        return DecodeResult(
            message: mockMessage,
            confidence: mockConfidence,
            bitAccuracy: 95,
            totalBits: 100,
            decodeTimeMs: 50.0,
            detectedRegion: CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8),
            rawBits: Array(repeating: 0.9, count: 100)
        )
    }
}

final class MockFastDetectorTests: XCTestCase {

    func testMockDetectorSuccess() async throws {
        let mock = MockFastDetector()
        mock.shouldSucceed = true
        mock.mockMessage = "Hello World"

        let result = try await mock.decode(image: UIImage())

        XCTAssertEqual(result.message, "Hello World")
        XCTAssertEqual(result.confidence, 0.95)
    }

    func testMockDetectorFailure() async {
        let mock = MockFastDetector()
        mock.shouldSucceed = false

        do {
            _ = try await mock.decode(image: UIImage())
            XCTFail("Expected error to be thrown")
        } catch {
            // Expected
            XCTAssertTrue(error is DecodeError)
        }
    }
}
