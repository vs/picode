# Picode iOS

> Part of the [Picode monorepo](../README.md). See also [picode-model](../picode-model/), the
> PyTorch framework that trains and exports the models this app runs.

A SwiftUI camera app that finds a Picode-encoded image in the viewfinder and decodes its
message on-device with Core ML.

## Features

- Live viewfinder scanning with a fast detector that outlines the encoded region (quadrilateral
  overlay)
- Tap-to-capture for a slower, higher-quality decode
- Perspective rectification of the detected region before decoding
- Result sheet showing the message, confidence, bit accuracy, decode time and detected region
- Protocol-based decoders (`PicodeDecoder`): real Core ML pipelines (`FastDetector`,
  `SlowDetector`) and a `StubDecoder` for UI work without models

## Requirements

- Xcode 15+ and [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`)
- iOS 16.0+, iPhone only. The camera needs a physical device; the simulator is enough for tests.

## Setup

```bash
cd picode-ios
xcodegen generate            # regenerates picode-ios.xcodeproj from project.yml
open picode-ios.xcodeproj
```

Set your development team under *Signing & Capabilities* before running on a device.

### Models

Put the exported `.mlpackage` files in `picode-ios/Models/`, and Xcode compiles them to
`.mlmodelc` in the app bundle. The directory is optional in `project.yml`, and model files are
git-ignored. Without models, the detectors fall back to placeholder output so the UI still
works.

| Model name | Produced by |
|-------------|-------------|
| `FastDetector` | `detect-export coreml …` (from `picode-model`) |
| `PicodeDecoder` or `StegaStampDecoder` | `python scripts/export_decoder.py …` (from `picode-model`) |

The on-device pipeline is currently wired for the 100-bit, 400×400 StegaStamp-style decoder
(`FastDetector.swift`: `numBits`, `decoderInputSize`). Running the 72-bit, 512×512 PicoTrust
production decoder means updating those constants and adding LDPC decoding.

## Project structure

```
picode-ios/
├── App/                     # PicodeApp entry point, assets
├── Camera/                  # AVFoundation controller + SwiftUI preview
├── Decoder/
│   ├── PicodeDecoder.swift  # Protocol, DecodeResult, DecodeError
│   ├── FastDetector.swift   # Real-time detector + decoder (Core ML / Vision)
│   ├── SlowDetector.swift   # Higher-quality decode for captured frames
│   └── StubDecoder.swift    # Fake decoder for UI development
├── Types/                   # Quadrilateral, ScanState
├── Views/                   # CameraView(+Model), ResultsView, overlays, capture button
└── Info.plist
picode-iosTests/             # XCTest: detectors, view model, geometry, integration
```

## Architecture

- **SwiftUI + MVVM**: `CameraViewModel` drives a scan state machine (`ScanState`).
- **UIKit camera bridge**: `CameraController` wraps `AVCaptureSession`, and `CameraPreview` is
  a `UIViewRepresentable`.
- **async/await** for the capture and decode flow.
- **Vision + Core ML** for inference, and **Core Image** for perspective correction.

## Testing

```bash
xcodebuild test -scheme picode-ios \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' CODE_SIGNING_ALLOWED=NO
```

## License

Apache License 2.0. See [LICENSE](../LICENSE).
