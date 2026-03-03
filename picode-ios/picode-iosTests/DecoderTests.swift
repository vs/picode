import XCTest
@testable import picode_ios

final class DecoderTests: XCTestCase {

    func testStubDecoderReturnsResult() async throws {
        let decoder = StubDecoder()
        let image = UIImage()

        let result = try await decoder.decode(image: image)

        XCTAssertFalse(result.message.isEmpty)
        XCTAssertGreaterThan(result.confidence, 0)
        XCTAssertLessThanOrEqual(result.confidence, 1)
        XCTAssertEqual(result.totalBits, 100)
        XCTAssertGreaterThan(result.bitAccuracy, 0)
        XCTAssertLessThanOrEqual(result.bitAccuracy, result.totalBits)
    }

    func testStubDecoderSimulatesProcessingTime() async throws {
        let decoder = StubDecoder()
        let image = UIImage()

        let start = Date()
        _ = try await decoder.decode(image: image)
        let elapsed = Date().timeIntervalSince(start)

        // Should take at least 50ms (simulated delay)
        XCTAssertGreaterThan(elapsed, 0.05)
    }

    func testDecodeResultHasValidRegion() async throws {
        let decoder = StubDecoder()
        let image = UIImage()

        let result = try await decoder.decode(image: image)

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
}
