
# Python Script to convert images into Excalidraw diagrams

I created this project to be able to take photos of system architectural diagrams I've drawn in my notebooks or on whiteboards and convert them into Excalidraw diagrams. This allows me to easily edit them and share them with others digitally.

**Process Overview**
1. Convert image to vectors
2. Detect shapes with OpenCV
3. Use OCR to Extract text and bounding boxes
4. Convert to Excalidraw JSON format
5. Save as Obsidian Excalidraw Markdown file

# Install the following packages:
- pip install opencv-python
- pip install Pillow
- pip install easyocr

## Run the script
1. activate your virtual environment
```bash
.venv\Scripts\activate.bat
```
2. run the script with the image file as an argument
```bash
python image-to-excalidraw.py
``` 

# Notes
https://youtu.be/oyqNdcbKhew
- keras_ocr seems like the best option for this as it supports using a GPU and has autoannotations 
- easyocr is also a good option

for keras_ocr there is an error in one of the dependencies: 
- https://github.com/aleju/imgaug/issues/859
- AttributeError: np.sctypes was removed in NumPy 2.0 release
- Fix: with the following code in the __init__.py file of the imgaug library:

```python
NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}
NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}
```


## keras-ocr and notes for other things
```
pip install keras-ocr
pip install tensorflow
tensorflow is needed for keras-ocr

not working:
pip install pypotrace
```
