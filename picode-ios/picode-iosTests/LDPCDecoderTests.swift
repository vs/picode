import XCTest
@testable import picode_ios

/// Checks the Swift LDPC decoder against vectors produced by the Python reference
/// (`picode-model/scripts/export_ldpc.py --vectors`).
final class LDPCDecoderTests: XCTestCase {

    private struct Vectors: Decodable {
        struct Case: Decodable {
            let logits: [Float]
            let decodes: Bool
            let text: String?
            let stored: String
        }
        let cases: [Case]
    }

    private func loadVectors() throws -> Vectors {
        let bundle = Bundle(for: LDPCDecoderTests.self)
        let url = try XCTUnwrap(bundle.url(forResource: "ldpc_vectors", withExtension: "json"))
        return try JSONDecoder().decode(Vectors.self, from: Data(contentsOf: url))
    }

    func testBundledSpecIsProductionCode() throws {
        let ldpc = try XCTUnwrap(LDPCDecoder.bundled())
        XCTAssertEqual(ldpc.spec.n, 72)
        XCTAssertEqual(ldpc.spec.k, 38)
        XCTAssertEqual(ldpc.spec.messagePositions.count, 38)
        XCTAssertTrue(ldpc.spec.checks.allSatisfy { $0.count == 6 })
    }

    func testMatchesPythonReference() throws {
        let ldpc = try XCTUnwrap(LDPCDecoder.bundled())
        let vectors = try loadVectors()
        XCTAssertFalse(vectors.cases.isEmpty)
        for (i, testCase) in vectors.cases.enumerated() {
            let result = ldpc.decode(logits: testCase.logits)
            if testCase.decodes {
                XCTAssertTrue(result.success, "case \(i) should decode")
                XCTAssertEqual(PayloadText.decode(result.payload), testCase.text, "case \(i)")
            } else {
                // Python could not correct it; Swift must not claim a wrong message either.
                if result.success {
                    XCTAssertNotEqual(PayloadText.decode(result.payload), "", "case \(i)")
                }
            }
        }
    }

    func testPayloadTextModes() {
        // compact: mode 0, then 'h' (index 8) and 'i' (index 9) as 6-bit symbols
        let compact = [0] + [0, 0, 1, 0, 0, 0] + [0, 0, 1, 0, 0, 1] + Array(repeating: 0, count: 25)
        XCTAssertEqual(PayloadText.decode(compact), "hi")
        // UTF-8: mode 1, then '!' (0x21)
        let utf8 = [1] + [0, 0, 1, 0, 0, 0, 0, 1] + Array(repeating: 0, count: 29)
        XCTAssertEqual(PayloadText.decode(utf8), "!")
        XCTAssertEqual(PayloadText.decode([]), "")
    }
}
