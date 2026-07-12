# Hero Vision Analysis (Mock 11) - Updated

Below is the updated vision debug overlay showing all bounding boxes, anchors, and standard Tesseract OCR read values related to our Hero on the mock board **11_flop_7c6s_check.png**.

![Hero Vision Debug Overlay](file:///C:/Users/zwonk/.gemini/antigravity-ide/brain/c68a647c-2540-4757-8bda-15d48c1088de/hero_boxes_mock_11.png)

---

### Hero Telemetry Readings & OCR Approaches:

| Feature/Region | Coordinates (x, y, w, h) | Detected Value | OCR / Matching Approach |
| --- | --- | --- | --- |
| **Hero Anchor** | `(767, 837)` | *Hexagon Anchor Match* | Template matching with hexagon mask |
| **Hero Cards** | `(640, 725, 230, 95)` | `['7c', '6s']` | Card template matching against standard deck |
| **Hero Name** | `(692, 859, 150, 26)` | `Zw` | Dual-pass `ocr_roi()` (Tesseract OCR) |
| **Hero Stack** | `(702, 887, 130, 34)` | `4293` | Dual-pass `ocr_roi()` (Tesseract OCR) |
| **Hero VPIP** | `(667, 825, 20, 24)` | `None` | Average BGR color classification |
| **Hero AGG** | `(847, 825, 20, 24)` | `None` | Average BGR color classification |

---

### Key Updates:
- **Unified Stack Crop & OCR**: Hero's stack is now parsed using the exact same bounding box layout (`cy+50` to `cy+84`) as opponent seats. The ML template classifier was removed, and parsing has been migrated to standard Tesseract OCR.
- **Robust `ocr_roi` Strategy**: For both Name and Stack, we run the dual-pass `ocr_roi()` method. This executes standard grayscale resizing followed by an Otsu threshold/inversion fallback. This ensures high-contrast binarization of text on complex background gradients, leading to accurate digit reads.
