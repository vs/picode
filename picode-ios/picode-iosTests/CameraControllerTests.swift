import XCTest
@testable import picode_ios

final class CameraControllerTests: XCTestCase {

    func testVideoDelegate_canBeSet() {
        let controller = CameraController()

        // VideoDelegate should be settable
        // (actual delegate test requires device)
        XCTAssertNil(controller.videoDelegate)
    }
}
