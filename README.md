
# 1. Convert image to vectors (simplified)
- 

# 2. Detect shapes
# 3. OCR Extract text

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

![alt text](image.png)

# Install 

- pip install opencv-python
- pip install Pillow
- pip install easyocr


## keras-ocr and notes for other things
```
pip install keras-ocr
pip install tensorflow
tensorflow is needed for keras-ocr

not working:
pip install pypotrace
```

