# confedge
A computer vision pipeline built with OpenCV to automatically detect and count cakes placed on an industrial tray along an assembly line. Designed to handle real-time or batch image processing in environments with gray/steel trays and cakes of varying colors and patterns.

## ⚙️ How It Works

1. **Preprocessing**
   - Convert input image to grayscale and HSV
   - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Gaussian blur to reduce high-frequency noise

2. **Tray Segmentation**
   - HSV thresholding to detect and mask out steel/black tray regions
   - Inversion to isolate cake candidates

3. **Edge Detection + Cleanup**
   - Canny edge detection on cleaned mask
   - Morphological dilation and closing to restore object continuity

4. **Contour Extraction & Filtering**
   - Detect bounding boxes using `cv2.findContours`
   - Filter based on area, shape, and hierarchy to remove noise

5. **Output**
   - Count of detected cakes
   - Saved image with bounding boxes and labels (`output/annotation.png`)

## 🧪 Sample Usage

### Request

You can use `curl`, Postman, or any HTTP client to send an image.

#### Example using `curl`:

```bash
curl -X POST "https://productcount.azurewebsites.net/api/count_items" \
  -H "Content-Type: multipart/form-data" \
  -F "image_upload=@sample_images/tray1.jpg"
```

## Requirements

```bash
pip install opencv-contrib-python numpy matplotlib azure-functions torch torchvision 
```

### 📌 Notes
Assumes the tray is always metallic/gray (HSV-based exclusion)

Designed to work with cakes of any color or pattern

Easily extendable for other industrial object detection use-cases


