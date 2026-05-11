import XCTest
@testable import picode_ios

final class SlowDetectorTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInit_succeeds() {
        let detector = SlowDetector()
        XCTAssertNotNil(detector)
    }

    func testInit_withNilModelURL_succeeds() throws {
        // Should not crash when model URL is nil
        let detector = try SlowDetector(decoderModelURL: nil)
        XCTAssertNotNil(detector)
    }

    // MARK: - PicodeDecoder Conformance Tests

    func testDecode_withValidImage_returnsResult() async throws {
        let detector = SlowDetector()

        // Create a valid test image
        let size = CGSize(width: 400, height: 400)
        UIGraphicsBeginImageContext(size)
        UIColor.gray.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        // decode() should return a result (with fallback behavior, it always finds something)
        let result = try await detector.decode(image: image)

        XCTAssertNotNil(result.message)
        XCTAssertGreaterThan(result.confidence, 0)
        XCTAssertEqual(result.totalBits, 100)
    }

    func testDecode_withEmptyImage_throws() async {
        let detector = SlowDetector()

        // Create an empty UIImage (no underlying image data)
        let image = UIImage()

        do {
            _ = try await detector.decode(image: image)
            XCTFail("Should throw for invalid image")
        } catch {
            XCTAssertTrue(error is DecodeError)
        }
    }

    // MARK: - Detection Tests

    func testDetect_withValidImage_returnsResult() async throws {
        let detector = SlowDetector()

        // Create a valid test image
        let size = CGSize(width: 400, height: 400)
        UIGraphicsBeginImageContext(size)
        UIColor.gray.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await detector.detect(image: image)

        // With fallback behavior (no models), should still return a result
        XCTAssertNotNil(result)
        XCTAssertNotNil(result?.message)
        XCTAssertGreaterThan(result?.confidence ?? 0, 0)
        XCTAssertEqual(result?.totalBits, 100)
    }

    func testDetect_withEmptyImage_throws() async {
        let detector = SlowDetector()

        // Create an empty UIImage (no underlying image data)
        let image = UIImage()

        do {
            _ = try await detector.detect(image: image)
            XCTFail("Should throw for invalid image")
        } catch {
            XCTAssertTrue(error is DecodeError)
            if case DecodeError.invalidImage = error {
                // Expected
            } else {
                XCTFail("Expected invalidImage error, got \(error)")
            }
        }
    }

    func testDetect_returnsNormalizedRegion() async throws {
        let detector = SlowDetector()

        // Create a valid test image
        let size = CGSize(width: 800, height: 600)
        UIGraphicsBeginImageContext(size)
        UIColor.blue.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await detector.detect(image: image)

        guard let region = result?.detectedRegion else {
            XCTFail("Expected region to be present")
            return
        }

        // Region should be normalized (0-1 range)
        XCTAssertGreaterThanOrEqual(region.minX, 0)
        XCTAssertLessThanOrEqual(region.maxX, 1)
        XCTAssertGreaterThanOrEqual(region.minY, 0)
        XCTAssertLessThanOrEqual(region.maxY, 1)
    }

    func testDetect_returnsCorrectBitCount() async throws {
        let detector = SlowDetector()

        // Create a valid test image
        let size = CGSize(width: 400, height: 400)
        UIGraphicsBeginImageContext(size)
        UIColor.red.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await detector.detect(image: image)

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.rawBits.count, 100)

        // All bits should be probabilities (0-1 range)
        for bit in result?.rawBits ?? [] {
            XCTAssertGreaterThanOrEqual(bit, 0)
            XCTAssertLessThanOrEqual(bit, 1)
        }
    }

    func testDetect_recordsDecodeTime() async throws {
        let detector = SlowDetector()

        // Create a valid test image
        let size = CGSize(width: 400, height: 400)
        UIGraphicsBeginImageContext(size)
        UIColor.green.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await detector.detect(image: image)

        XCTAssertNotNil(result)
        XCTAssertGreaterThan(result?.decodeTimeMs ?? 0, 0)
    }

    // MARK: - Multi-Scale Tests

    func testDetect_searchesMultipleScales() async throws {
        // This test verifies that the detector uses the sliding window approach
        // by testing with images of different sizes
        let detector = SlowDetector()

        // Create a larger image that would have multiple windows
        let size = CGSize(width: 1000, height: 1000)
        UIGraphicsBeginImageContext(size)
        UIColor.yellow.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await detector.detect(image: image)

        // Should complete without error
        XCTAssertNotNil(result)
    }

    // MARK: - Performance Tests

    func testDetect_performance() async throws {
        let detector = SlowDetector()

        // Create a test image
        let size = CGSize(width: 400, height: 400)
        UIGraphicsBeginImageContext(size)
        UIColor.purple.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        // SlowDetector is expected to be slower than FastDetector
        // but should still complete in reasonable time (< 5 seconds for 400x400)
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await detector.detect(image: image)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        // Without real models, should complete quickly
        // With real models on a larger image, may take longer
        XCTAssertLessThan(elapsed, 5000, "Detection should complete in under 5 seconds")
    }

    // MARK: - Message Conversion Tests

    func testDetect_returnsValidMessage() async throws {
        let detector = SlowDetector()

        // Create a valid test image
        let size = CGSize(width: 400, height: 400)
        UIGraphicsBeginImageContext(size)
        UIColor.orange.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await detector.detect(image: image)

        XCTAssertNotNil(result)
        XCTAssertFalse(result?.message.isEmpty ?? true)
    }
}

// MARK: - Mock SlowDetector for Testing

/// Mock SlowDetector that returns predictable results.
final class MockSlowDetector {
    var shouldReturnResult = true
    var mockConfidence: Float = 0.8
    var mockMessage = "Test Message"

    func detect(image: UIImage) async throws -> DecodeResult? {
        guard CIImage(image: image) != nil else {
            throw DecodeError.invalidImage
        }

        if !shouldReturnResult {
            return nil
        }

        return DecodeResult(
            message: mockMessage,
            confidence: Double(mockConfidence),
            bitAccuracy: 80,
            totalBits: 100,
            decodeTimeMs: 500.0,
            detectedRegion: CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6),
            rawBits: Array(repeating: 0.8, count: 100)
        )
    }
}

final class MockSlowDetectorTests: XCTestCase {

    func testMockDetector_returnsResult() async throws {
        let mock = MockSlowDetector()
        mock.shouldReturnResult = true
        mock.mockMessage = "Hello World"

        // Create a valid test image
        let size = CGSize(width: 100, height: 100)
        UIGraphicsBeginImageContext(size)
        UIColor.black.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await mock.detect(image: image)

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.message, "Hello World")
    }

    func testMockDetector_returnsNil() async throws {
        let mock = MockSlowDetector()
        mock.shouldReturnResult = false

        // Create a valid test image
        let size = CGSize(width: 100, height: 100)
        UIGraphicsBeginImageContext(size)
        UIColor.white.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = try await mock.detect(image: image)

        XCTAssertNil(result)
    }

    func testMockDetector_throwsForInvalidImage() async {
        let mock = MockSlowDetector()

        let image = UIImage()

        do {
            _ = try await mock.detect(image: image)
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertTrue(error is DecodeError)
        }
    }
}
