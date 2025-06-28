#!/usr/bin/env python3
"""
Pipeline for Converting Hand-Drawn Images to Excalidraw Format

Stage 1: Preprocess Image
	- Loads image and converts to grayscale
	- Applies thresholding to identify drawing strokes
	- Outputs 'gray.png' and 'threshold.png'

Stage 2: Shape Detection 
	- Detects contours and approximates basic shapes
	- Identifies rectangles, squares, circles, polygons
	- Outputs 'detected_shapes.png' with annotations

Stage 3: OCR with EasyOCR
	- Detects and recognizes text in image
	- Outputs 'ocr_output.png' with detected text and bounds

Stage 4: Excalidraw Export
	- Converts detected shapes and text to Excalidraw JSON format
	- Generates Obsidian-compatible markdown file
	- Outputs 'excalidraw.json' and '.excalidraw.md' files

Logs progress details at each stage.
"""

import logging
import os
import numpy as np
import cv2
from PIL import Image
# import potrace
import easyocr
import json

from Excalidraw_Interface import SketchBuilder

# Configure logging with timestamp and level.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def preprocess_image(input_path, output_dir):
	"""
	Loads an image, converts it to grayscale, and applies thresholding.
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
	gray_path = os.path.join(output_dir, "1-gray.png")
	cv2.imwrite(gray_path, gray)
	logging.info("Saved grayscale image to %s", gray_path)
	
	# Apply binary thresholding.
	logging.info("Applying binary thresholding")

	# method = 'HSV Color Segmentation + Morphology gray'

	# # Convert to HSV and create masks for black/blue
	# hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

	# # Blue mask (adjust hue range for your specific marker)
	# lower_blue = np.array([90, 50, 50])  # ~100-140° hue
	# upper_blue = np.array([130, 255, 255])
	# blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

	# # Black mask (low value channel)
	# lower_black = np.array([0, 0, 0])
	# upper_black = np.array([180, 255, 100])  # Max value=50 for darkness
	# black_mask = cv2.inRange(hsv, lower_black, upper_black)

	# # Combine masks and enhance
	# combined = cv2.bitwise_or(blue_mask, black_mask)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	# thresh = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)






	# # 2. Adaptive Threshold + Contrast Enhancement
	# method = 'adaptive-clahe'
	
	# # CLAHE for contrast enhancement
	# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	# enhanced = clahe.apply(gray)

	# # Adaptive threshold with different parameters
	# thresh = cv2.adaptiveThreshold(enhanced, 255, 
	# 							cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 							cv2.THRESH_BINARY_INV, 21, 5)

	# # Morphological cleanup
	# kernel = np.ones((2,2), np.uint8)
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	# thresh = cv2.dilate(thresh, kernel, iterations=1)


	# # Option 1: Larger Morphological Kernel + Opening
	
	# method = 'adaptive-clahe-reduced-contrast'

	# # CLAHE with reduced contrast amplification
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Reduced clipLimit
	# enhanced = clahe.apply(gray)

	# # Adaptive threshold with larger block size
	# thresh = cv2.adaptiveThreshold(enhanced, 255, 
	# 							cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 							cv2.THRESH_BINARY_INV, 21, 7)  # Increased C to 7

	# # Morphological opening (erode->dilate) with larger kernel
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


	# Option 2: Median Blur + Adjusted Thresholding - Best so far 2025-06-14
	method = 'median-blur-adaptive-thresh'
	# Median blur for salt-and-pepper noise removal
	blurred = cv2.medianBlur(gray, 3)

	# Adaptive threshold with different parameters
	thresh = cv2.adaptiveThreshold(blurred, 255, 
								cv2.ADAPTIVE_THRESH_MEAN_C,  # Try mean instead of Gaussian
								cv2.THRESH_BINARY_INV, 25, 5)

	# Two-step morphological cleanup
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)



	# Option 3: Noise-Aware Threshold Fusion
	# not as good as above


	# # 3. Edge-Aware Threshold Fusion
	# method = 'edge-aware-fusion-gray'

	# # Dual processing paths
	# edges = cv2.Canny(original, 50, 150)
	# adapt = cv2.adaptiveThreshold(gray, 255, 
	# 							cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 							cv2.THRESH_BINARY_INV, 15, 3)

	# # Combine edge + threshold info
	# combined = cv2.bitwise_and(edges, adapt)
	# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
	# thresh = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)






	# Choose preprocessing method: 'threshold', 'adaptive', 'canny', or 'otsu'
	# method = 'adaptive-erosion-iterations-2'

	# # Adaptive threshold - better for images with varying illumination
	# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
	# 								cv2.THRESH_BINARY_INV, 11, 2)
	# # Add mild erosion to remove noise while preserving details
	# kernel = np.ones((1,1), np.uint8)
	# thresh = cv2.erode(thresh, kernel, iterations=2)
	# # thresh = cv2.Canny(thresh, 100, 200)
	
	# if method == 'threshold':
	# 	# Basic binary threshold - makes dark strokes white (foreground) and background black
	# 	ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
	# elif method == 'adaptive-erosion-less':
	# 	# Adaptive threshold - better for images with varying illumination
	# 	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
	# 								 cv2.THRESH_BINARY_INV, 11, 2)
	# 	# Add mild erosion to remove noise while preserving details
	# 	kernel = np.ones((2,2), np.uint8)
	# 	thresh = cv2.erode(thresh, kernel, iterations=1)
	# elif method == 'canny':
	# 	# Canny edge detection - good for finding edges/contours
	# 	thresh = cv2.Canny(gray, 100, 200)
	# elif method == 'otsu':
	# 	# Otsu's method - automatically determines optimal threshold value
	# 	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	filename_with_method = f"2-threshold-{method}.png"

	thresh_path = os.path.join(output_dir, filename_with_method)
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
	Takes a binary image, detects shapes, and annotates them on the original image.
	
	Identifies:
	- Squares/Rectangles: 4 vertices
	- Circles: High circularity
	- Polygons: Other closed shapes
	
	Args:
		binary_image: Thresholded binary image
		original_image: Original color image for drawing
		output_image_path: Where to save annotated image
	
	Returns:
		shapes: List of detected shapes with properties
		shape_image: Image with shape annotations
	"""
	logging.info("Detecting shapes from contours")
	contours, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	logging.info("Found %d contours", len(contours))
	
	# Make a copy of the original image for drawing
	shape_image = original_image.copy()
	shapes = []
	
	# Pre-filter small noise contours (adjust 500 based on your image size)
	area_threshold = 5
	contours = [c for c in contours if cv2.contourArea(c) > area_threshold]
	logging.info("Filtered contours to %d based on area threshold of %d", len(contours), area_threshold)

	for idx, contour in enumerate(contours):
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.015 * peri, True)  # Reduced epsilon
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
	Performs OCR using EasyOCR to detect and recognize text in images.

	Args:
		original_image: Input image array
		output_image_path: Path to save annotated image
		
	Returns:
		List of (bbox, text, confidence) tuples for detected text
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

# New fun to create Excalidraw JSON.
def generate_excalidraw_json(shapes, ocr_results):
	sb = SketchBuilder()

	# Process shapes first
	for shape in shapes:
		x = shape['x']
		y = shape['y']
		width = shape['width']
		height = shape['height']

		# center
		center_x = int(x + width / 2)
		center_y = int(y + height / 2)

		if shape['type'] == 'Rectangle' or shape['type'] == 'Square':
			shape = sb.Rectangle(x=center_x, y=center_y, width=width, height=height)
		elif shape['type'] == 'Circle':
			shape = sb.Ellipse(x=center_x, y=center_y, width=width, height=height)

		elif shape['type'] == 'Polygon':
			# For polygons, create lines between vertices
			contour = shape['contour']
			for i in range(len(contour)):
				start = contour[i][0]
				end = contour[(i + 1) % len(contour)][0]
				sb.Line((start[0], start[1]), (end[0], end[1]))

	# Process OCR results
	for bbox, text, conf in ocr_results:
		# Calculate center coordinates and height from the bounding box
		x1, y1 = bbox[0]  # Top-left
		x2, y2 = bbox[2]  # Bottom-right
		center_x = int((x1 + x2) / 2)
		center_y = int((y1 + y2) / 2)
		height = int(y2 - y1)  # Calculate text height
		# Scale font size based on height, with some reasonable limits
		font_size = min(max(height/2, 8), 36)  # Min 8, max 36
		box = sb.Text(text, x=center_x, y=center_y, fontSize=font_size)
		# some math I think is not needed anymore
		# distance_shift_left = (center_x - box.width / 2) - box.x
		# box.x -= distance_shift_left
		box.x = int(x1)
		box.y = int(y1)

	# export data = sb.export_to_json()
	return sb.export_to_json()


# 	Creating Sketch Objects
# Rectangles, Diamonds, Ellipses can be created with a center_x and center_y position. Width and height can also be set (defaults to 100). Other params can be set in kwargs.

# from Excalidraw_Interface import SketchBuilder

# sb = SketchBuilder()
# sb.Rectangle(x = 0, y = 0)
# sb.Diamond(x = 0, y = 0, width=50, height=20)
# sb.Ellipse(x = 0, y = 0, backgroundColor='red')
# Text, Lines, and Arrows have similar functions.

# from Excalidraw_Interface import SketchBuilder

# sb = SketchBuilder()
# sb.Text('some text', x = 0, y = 0)
# sb.Line((0,0), (100,100))
# sb.Arrow((0,0), (100,100))
# sb.DoubleArrow((0,0), (100,100))




# My json creator before discovering the Excalidraw_Interface class.
# The JSON format for Excalidraw is a bit complex, so we will need to define the structure at some point. - lucas 2025-04-17
# def generate_excalidraw_json(shapes, ocr_results):
# 	"""
# 	Generates a JSON representation for Excalidraw based on detected shapes and OCR results.
	
# 	Parameters:
# 	  - shapes: List of detected shapes with their properties.
# 	  - ocr_results: List of OCR results with text and bounding boxes.
	
# 	Returns:
# 	  A JSON object structured for Excalidraw.
# 	"""

# 	logging.info("Generating Excalidraw JSON")
# 	excalidraw_elements = []
	
# 	for shape in shapes:
# 		element = {
# 			"type": "rectangle" if shape['type'] == "Rectangle" else "ellipse",
# 			"x": shape['x'],
# 			"y": shape['y'],
# 			"width": shape['width'],
# 			"height": shape['height'],
# 			"angle": 0,
# 			"fillColor": "#FFFFFF",
# 			"strokeColor": "#000000",
# 			"strokeWidth": 1,
# 			"text": shape.get('label', '')
# 		}
# 		excalidraw_elements.append(element)
	
# 	for bbox, text, _ in ocr_results:
# 		element = {
# 			"type": "text",
# 			"x": int(bbox[0][0]),
# 			"y": int(bbox[0][1]),
# 			"text": text,
# 			"fontSize": 20,
# 			"color": "#000000"
# 		}
# 		excalidraw_elements.append(element)
	
# 	return excalidraw_elements

# Old
# def generate_excalidraw_markdown(excalidraw_json_data):
# 	"""
# 	Generates a markdown representation for Excalidraw JSON data.
	
# 	Parameters:
# 	  - excalidraw_json_data: JSON data structured for Excalidraw.
	
# 	Returns:
# 	  A markdown string formatted for Obsidian.
# 	"""
# 	markdown = "---\n"
# 	markdown += "excalidraw-plugin: parsed\n"
# 	markdown += "tags: [excalidraw]\n"
# 	markdown += "---\n\n"
# 	markdown += "== Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. == You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'\n\n"
# 	markdown += "# Excalidraw Data\n\n"
# 	markdown += "## Text Elements\n%%\n"
# 	markdown += "## Drawing\n```json\n"
# 	markdown += "{\n"
# 	markdown += "\"type\": \"excalidraw\",\n"
# 	markdown += "\"version\": 2,\n"
# 	markdown += "\"source\": \"https://github.com/zsviczian/obsidian-excalidraw-plugin/releases/tag/2.10.1\",\n"
# 	markdown += "\"elements\": "
# 	markdown += json.dumps(excalidraw_json_data, indent=4)
# 	markdown += "\n"
# 	markdown += "}\n"
# 	markdown += "\n```\n%%\n"
# 	return markdown








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
	markdown += json.dumps(excalidraw_json_data, indent=4)
	markdown += "\n```\n%%\n"
	return markdown




# Function to run the entire pipeline.
def run_pipeline(input_path, output_dir):
	"""
	Runs the entire pipeline from preprocessing to OCR and Excalidraw JSON generation.
	
	Parameters:
	  - input_path: Path to the input image file.
	  - output_dir: Directory where output files will be saved.
	"""
	# PART 1: Preprocessing and Vectorization.
	original_img, gray_img, thresh_img = preprocess_image(input_path, output_dir)
	if original_img is None:
		logging.error("Exiting: Could not load image.")
		return
	
	# svg_output_path = os.path.join(output_dir, "vectorized.svg")
	# vectorize_image(thresh_img, svg_output_path)
	
	# PART 2: Shape Detection.
	shapes_output_path = os.path.join(output_dir, "3-detected_shapes Reduced epsilon.png")
	shapes, image = detect_shapes(thresh_img, original_img, shapes_output_path)

	# PART 3: OCR using Easy OCR.
	ocr_output_path = os.path.join(output_dir, "4-ocr_output.png")
	ocr_results = perform_ocr_easy(original_img, ocr_output_path)
	
	logging.info("Pipeline complete. Check the '%s' folder for output images and SVG file.", output_dir)

	# Now onto generating JSON for the Excalidraw file.
	# This will be a separate function that takes the detected shapes and OCR results to create a JSON file.
	json_output_path = os.path.join(output_dir, "5-excalidraw.json")
	excalidraw_data = generate_excalidraw_json(shapes, ocr_results)
	with open(json_output_path, 'w') as json_file:
		json.dump(excalidraw_data, json_file, indent=4)
	logging.info("Saved Excalidraw JSON to %s", json_output_path)

	# save markdown file
	markdown_output_path = os.path.join(output_dir, "6-Drawing-for-Obsidian.excalidraw.md")
	markdown_data = generate_excalidraw_markdown(excalidraw_data)
	with open(markdown_output_path, 'w') as md_file:
		md_file.write(markdown_data)
	logging.info("Saved Excalidraw markdown to %s", markdown_output_path)



def test_single_image(input_path, output_dir):
	logging.info("Running test on single image: %s", input_path)
	run_pipeline(input_path, output_dir)
	logging.info("Test completed successfully.")


def test_multiple_images(input_dir_batch, output_dir_batch):
	logging.info("Running batch test on images in directory: %s", input_dir_batch)
	for filename in os.listdir(input_dir_batch):
		if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
			input_path = os.path.join(input_dir_batch, filename)
			logging.info("Processing image: %s", input_path)
			output_dir = os.path.join(output_dir_batch, filename.split(".")[0])
			os.makedirs(output_dir, exist_ok=True)
			run_pipeline(input_path, output_dir)
	logging.info("Batch test completed successfully.")


def main():
	# Define paths.
	# input_path = "input_images/Fledge_test2_20250613_111426.jpg"  # Change this to your image file.
	# input_path = "input_images/testimage1.png"  # Change this to your image file.
	# input_path = "input_images/Netflix-High-Level-System-Architecture.png"  # Change this to your image file.
	# input_path = "input_images/Screenshot 2025-04-17 042813.png"  # Change this to your image file.

	# output_dir = "output/" + input_path.split(".")[0].split("/")[-1]  # Create a unique output directory based on the input image name.
	# os.makedirs(output_dir, exist_ok=True)
	# test_single_image(input_path, output_dir)

	# For batch processing, uncomment the following lines:
	input_dir_batch = "input_images"  # Directory containing multiple images.
	output_dir_batch = "output/batch_test_output"  # Directory to save batch outputs.
	os.makedirs(output_dir_batch, exist_ok=True)
	test_multiple_images(input_dir_batch, output_dir_batch)


if __name__ == "__main__":
	main()
