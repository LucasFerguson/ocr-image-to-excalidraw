#!/usr/bin/env python3
"""
Three-Stage Pipeline for Drawing Analysis

Stage 1: Preprocess & Vectorize
  - Loads the image and converts it to grayscale.
  - Applies binary thresholding.
  - Uses the Potrace library to generate an SVG vector file from the thresholded image.
  (Expected outputs: 'gray.png', 'threshold.png', and 'vectorized.svg' in the output folder.)

Stage 2: Shape Detection
  - Uses OpenCV to detect contours in the threshold image.
  - Approximates each contour to determine whether it is a square, rectangle, circle, or an unspecified polygon.
  - Draws these detected shapes with annotations on the original image.
  (Expected output: 'detected_shapes.png'.)

Stage 3: OCR using Keras-OCR
  - Converts the original image to RGB and processes it with a pretrained Keras-OCR pipeline.
  - Logs each detected text snippet with its bounding box.
  - Uses keras-ocr’s built-in annotation function to produce an image showing text and bounding boxes.
  (Expected output: 'ocr_output.png'.)
  
The logging output will display progress details and key values from each step.
"""

import logging
import os
import numpy as np
import cv2
from PIL import Image
# import potrace
import easyocr
import json

# Configure logging with timestamp and level.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def preprocess_image(input_path, output_dir):
	"""
	Loads an image, converts it to grayscale, and applies binary thresholding.
	
	Saves:
	  - 'gray.png': Grayscale version.
	  - 'threshold.png': Binary image where drawing strokes become white on a black background.
	
	Returns:
	  original: The loaded original BGR image.
	  gray: The grayscale image.
	  thresh: The binary image.
	"""
	logging.info("Loading image from %s", input_path)
	original = cv2.imread(input_path)
	if original is None:
		logging.error("Failed to load image at %s", input_path)
		return None, None, None

	# Convert to grayscale.
	logging.info("Converting image to grayscale")
	gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	gray_path = os.path.join(output_dir, "gray.png")
	cv2.imwrite(gray_path, gray)
	logging.info("Saved grayscale image to %s", gray_path)
	
	# Apply binary thresholding.
	logging.info("Applying binary thresholding")
	# THRESH_BINARY_INV: makes dark strokes white (foreground) and background black.
	ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
	thresh_path = os.path.join(output_dir, "threshold.png")
	cv2.imwrite(thresh_path, thresh)
	logging.info("Saved threshold image to %s", thresh_path)
	
	return original, gray, thresh

# def vectorize_image(binary_image, output_svg_path):
# 	"""
# 	Uses Potrace to convert a binary image (0/255) into an SVG vector representation.
	
# 	The binary image is converted into a 0/1 NumPy array and then traced.
	
# 	Saves:
# 	  - An SVG file at the specified path.
	
# 	Returns:
# 	  svg_content: The SVG file content as a string.
# 	"""
# 	logging.info("Vectorizing image with Potrace")
# 	# Convert 255/0 binary image to a 0/1 array.
# 	bmp_array = (binary_image > 0).astype(np.uint8)
# 	bmp = potrace.Bitmap(bmp_array)
# 	path = bmp.trace()
	
# 	svg_content = path.to_svg()
# 	with open(output_svg_path, 'w') as f:
# 		f.write(svg_content)
# 	logging.info("Saved vectorized SVG to %s", output_svg_path)
# 	return svg_content

def detect_shapes(binary_image, original_image, output_image_path):
	"""
	Detect contours in the thresholded binary image and approximate shapes.
	
	For each contour, if it has:
	  - Four vertices → classified as Square (if aspect ratio nearly 1) or Rectangle.
	  - More than 4 vertices → if circularity is high, classified as a Circle; otherwise as a Polygon.
	
	Returns:
	  shapes: List of dictionaries containing shape information (type, position, dimensions)
	  shape_image: Annotated image with drawn contours and shape labels
	"""
	logging.info("Detecting shapes from contours")
	contours, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	logging.info("Found %d contours", len(contours))
	
	# Make a copy of the original image for drawing
	shape_image = original_image.copy()
	shapes = []
	
	for idx, contour in enumerate(contours):
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
		x, y, w, h = cv2.boundingRect(approx)
		shape_type = "Unidentified"
		
		if len(approx) == 4:
			# Potential rectangle or square
			aspect_ratio = float(w) / h
			if 0.95 <= aspect_ratio <= 1.05:
				shape_type = "Square"
			else:
				shape_type = "Rectangle"
		elif len(approx) > 4:
			# Compare area of the contour to that of its minimum enclosing circle
			area = cv2.contourArea(contour)
			(cx, cy), radius = cv2.minEnclosingCircle(contour)
			circle_area = np.pi * (radius ** 2)
			if abs(1 - (area / circle_area)) < 0.2:
				shape_type = "Circle"
			else:
				shape_type = "Polygon"
		else:
			shape_type = "Polygon"

		# Store shape information
		shape_info = {
			'type': shape_type,
			'x': int(x),
			'y': int(y),
			'width': int(w),
			'height': int(h),
			'vertices': len(approx),
			'contour': approx.tolist()
		}
		shapes.append(shape_info)
		
		logging.info("Contour %d: %s found at [x=%d, y=%d, w=%d, h=%d] with %d vertices", 
					idx, shape_type, x, y, w, h, len(approx))
		
		# Draw the approximated contour and label it
		cv2.drawContours(shape_image, [approx], -1, (0, 255, 0), 2)
		cv2.putText(shape_image, shape_type, (x, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	
	cv2.imwrite(output_image_path, shape_image)
	logging.info("Saved shape detection image to %s", output_image_path)
	return shapes, shape_image


def perform_ocr_easy(original_image, output_image_path):
    """
    Uses EasyOCR to perform OCR on the image.

    Steps:
      - Optionally converts the image if needed.
      - Initializes an EasyOCR reader for the English language.
      - Processes the image to obtain bounding boxes, recognized text, and confidence values.
      - Logs each detected text string along with its bounding box and confidence.
      - Draws annotations (bounding boxes and text) on the image.
      - Saves the annotated image.

    Returns:
      results: A list of OCR predictions, where each prediction is in the form: 
               [bounding box, text string, confidence score]
    """
    logging.info("Performing OCR using EasyOCR")
    
    # Create a copy of the original image to draw annotations
    annotated_image = original_image.copy()
    
    # Initialize the EasyOCR reader for English (add more languages if needed)
    reader = easyocr.Reader(['en'], gpu=False)
    logging.info("Initialized EasyOCR reader; processing image...")
    
    # Process the image. Each result is a tuple: (bbox, text, confidence)
    results = reader.readtext(original_image)
    
    for bbox, text, conf in results:
        # Log the detection: convert each bounding box point to an integer list for clarity.
        bbox_int = [list(map(int, point)) for point in bbox]
        logging.info("Detected OCR text '%s' with box %s and confidence %.2f", text, bbox_int, conf)
        
        # Convert the bounding box list to a NumPy array for drawing
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Instead of casting the entire bbox to an int, extract the top-left coordinate.
        # bbox[0] should be the top-left corner if the points are ordered.
        x, y = int(bbox[0][0]), int(bbox[0][1])
        text_position = (x, y - 10)
        
        cv2.putText(annotated_image, text, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(output_image_path, annotated_image)
    logging.info("Saved OCR annotated image to %s", output_image_path)
    
    return results

# The JSON format for Excalidraw is a bit complex, so we will need to define the structure at some point. - lucas 2025-04-17
def generate_excalidraw_json(shapes, ocr_results):
	"""
	Generates a JSON representation for Excalidraw based on detected shapes and OCR results.
	
	Parameters:
	  - shapes: List of detected shapes with their properties.
	  - ocr_results: List of OCR results with text and bounding boxes.
	
	Returns:
	  A JSON object structured for Excalidraw.
	"""

	logging.info("Generating Excalidraw JSON")
	excalidraw_elements = []
	
	for shape in shapes:
		element = {
			"type": "rectangle" if shape['type'] == "Rectangle" else "ellipse",
			"x": shape['x'],
			"y": shape['y'],
			"width": shape['width'],
			"height": shape['height'],
			"angle": 0,
			"fillColor": "#FFFFFF",
			"strokeColor": "#000000",
			"strokeWidth": 1,
			"text": shape.get('label', '')
		}
		excalidraw_elements.append(element)
	
	for bbox, text, _ in ocr_results:
		element = {
			"type": "text",
			"x": int(bbox[0][0]),
			"y": int(bbox[0][1]),
			"text": text,
			"fontSize": 20,
			"color": "#000000"
		}
		excalidraw_elements.append(element)
	
	return excalidraw_elements

# Wrap Excalidraw JSON in markdown for Obsidian

# Example:
	# ---

	# excalidraw-plugin: parsed
	# tags: [excalidraw]

	# ---
	# ==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'


	# # Excalidraw Data

	# ## Text Elements
	# %%
	# ## Drawing
	# ```json
	# {
	# ...json data
	# }
	# ```
	# %%
	# def 
def generate_excalidraw_markdown(excalidraw_json_data):
	"""
	Generates a markdown representation for Excalidraw JSON data.
	
	Parameters:
	  - excalidraw_json_data: JSON data structured for Excalidraw.
	
	Returns:
	  A markdown string formatted for Obsidian.
	"""
	markdown = "---\n"
	markdown += "excalidraw-plugin: parsed\n"
	markdown += "tags: [excalidraw]\n"
	markdown += "---\n\n"
	markdown += "== Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. == You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'\n\n"
	markdown += "# Excalidraw Data\n\n"
	markdown += "## Text Elements\n%%\n"
	markdown += "## Drawing\n```json\n"
	markdown += "{\n"
	markdown += "\"type\": \"excalidraw\",\n"
	markdown += "\"version\": 2,\n"
	markdown += "\"source\": \"https://github.com/zsviczian/obsidian-excalidraw-plugin/releases/tag/2.10.1\",\n"
	markdown += "\"elements\": "
	markdown += json.dumps(excalidraw_json_data, indent=4)
	markdown += ",\n"
	markdown += "}\n"
	markdown += "\n```\n%%\n"
	return markdown

def main():
	# Define paths.
	input_path = "input_images/testimage1.png"  # Change this to your image file.
	output_dir = "output/" + input_path.split(".")[0].split("/")[-1]  # Create a unique output directory based on the input image name.
	os.makedirs(output_dir, exist_ok=True)
	
	# PART 1: Preprocessing and Vectorization.
	original_img, gray_img, thresh_img = preprocess_image(input_path, output_dir)
	if original_img is None:
		logging.error("Exiting: Could not load image.")
		return
	
	# svg_output_path = os.path.join(output_dir, "vectorized.svg")
	# vectorize_image(thresh_img, svg_output_path)
	
	# PART 2: Shape Detection.
	shapes_output_path = os.path.join(output_dir, "detected_shapes.png")
	shapes, image = detect_shapes(thresh_img, original_img, shapes_output_path)
	
	# PART 3: OCR using Keras-OCR.
	ocr_output_path = os.path.join(output_dir, "ocr_output.png")
	ocr_results = perform_ocr_easy(original_img, ocr_output_path)
	
	logging.info("Pipeline complete. Check the '%s' folder for output images and SVG file.", output_dir)

	# Now onto generating JSON for the Excalidraw file.
	# This will be a separate function that takes the detected shapes and OCR results to create a JSON file.
	json_output_path = os.path.join(output_dir, "excalidraw.json")
	excalidraw_data = generate_excalidraw_json(shapes, ocr_results)
	with open(json_output_path, 'w') as json_file:
		json.dump(excalidraw_data, json_file, indent=4)
	logging.info("Saved Excalidraw JSON to %s", json_output_path)

	# save markdown file
	markdown_output_path = os.path.join(output_dir, "Drawing 4 03.22.42.excalidraw.md")
	markdown_data = generate_excalidraw_markdown(excalidraw_data)
	with open(markdown_output_path, 'w') as md_file:
		md_file.write(markdown_data)
	logging.info("Saved Excalidraw markdown to %s", markdown_output_path)

if __name__ == "__main__":
	main()
