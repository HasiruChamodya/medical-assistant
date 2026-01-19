# Medical-Assistant Brain Tumor Detection: Data Preparation (Stage 1)

This repo helps build an ML-based brain MRI classifier.  
**Data:** Multi-class (glioma, meningioma, pituitary, no_tumor)  
See `src/preview.py` for data inspection and `src/preprocess.py` for pipeline setup.

## Folder Structure
```
brain_tumor_dataset/
    Training/
        glioma_tumor/
        meningioma_tumor/
        pituitary_tumor/
        no_tumor/
    Testing/
        glioma_tumor/
        meningioma_tumor/
        pituitary_tumor/
        no_tumor/
src/
    preview.py
    preprocess.py
```
- **Data is loaded using Keras ImageDataGenerator (multi-class ready).**

## How to Run
1. Inspect and preview dataset:  
   `python src/preview.py`
2. Prepare and visualize data pipeline:  
   `python src/preprocess.py`