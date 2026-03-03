# Picode iOS

iPhone camera app for decoding picode watermarks.

## Features

- Point camera at an encoded image
- Tap to capture and decode
- View decoded message with technical metadata:
  - Confidence score
  - Bit accuracy
  - Decode time
  - Detection region

## Requirements

- iOS 16.0+
- Xcode 15.0+

## Setup

1. Open Xcode and create a new project:
   - Select **App** template
   - Product Name: `picode-ios`
   - Interface: **SwiftUI**
   - Language: **Swift**

2. Replace the generated files with the source files in `picode-ios/`:
   - Delete the auto-generated `ContentView.swift`
   - Add all files from `picode-ios/` maintaining the folder structure
   - Add `Info.plist` to the project

3. Configure the project:
   - Set minimum deployment target to **iOS 16.0**
   - Add **Camera** capability in Signing & Capabilities

4. Build and run on a physical device (camera not available in simulator)

## Project Structure

```
picode-ios/
├── App/
│   └── PicodeApp.swift          # App entry point
├── Camera/
│   ├── CameraController.swift   # AVFoundation wrapper
│   └── CameraPreview.swift      # UIViewRepresentable
├── Decoder/
│   ├── PicodeDecoder.swift      # Protocol + types
│   └── StubDecoder.swift        # Fake decoder for dev
├── Views/
│   ├── CameraView.swift         # Main screen
│   ├── CameraViewModel.swift    # State management
│   ├── ResultsView.swift        # Results sheet
│   └── Components/
│       └── CaptureButton.swift  # Capture button
└── Info.plist                   # Permissions
```

## Architecture

- **SwiftUI** for UI with **UIKit** camera integration
- **MVVM** pattern with `@StateObject` view models
- **Protocol-based** decoder for easy swapping (stub → Core ML)
- **async/await** for capture and decode flow

## Testing

Run tests in Xcode:
- `DecoderTests` - Decoder protocol conformance
- `ViewModelTests` - State transitions

## Next Steps

1. **Core ML Integration** - Replace `StubDecoder` with real model
2. **Live Scanning** - Real-time viewfinder detection
3. **Photo Library** - Import images to decode
4. **History** - Save decoded messages

## License

MIT
