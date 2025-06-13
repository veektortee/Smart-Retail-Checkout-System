# 🛒 Smart Retail Object Detection System

A full end-to-end system that uses YOLOv8 to detect and classify retail products from images. It simulates a smart checkout experience by identifying products and calculating total cost based on class-level pricing.

---

## 🔍 Project Motivation

Traditional checkout systems rely heavily on barcode scanners, which can be slow and require manual effort. This project demonstrates how computer vision can replace barcode scanners in a retail setting, leading to faster, automated, and more flexible checkout systems.

---

## 🧠 Key Features

* Fine-tuned YOLOv8 model on a retail-specific dataset
* Object detection and classification of 22 retail product types
* Streamlit web app interface for visualizing predictions and calculating total estimated price
* Automatically detects and labels products in uploaded images

---

* 🔗 [Demo Video](https://youtu.be/VVNP3Kgs00I)


## 📃 Class Labels

The model recognizes multiple retail product classes. For the full list of class labels, please refer to the class definitions in `utils.py`.

---

## ⚠️ Important Disclaimer

> **Note**: This system assigns prices per class label and does not differentiate between different instances within a class. For example, all items labeled as `box` are treated the same regardless of their actual contents. This approach is not suitable for fine-grained product differentiation, which in real-world retail systems is typically handled via barcode scanning or product embedding matching. This project is intended for demonstration and portfolio purposes.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/veektortee/Smart-Retail-Checkout-System.git
cd Smart-Retail-Checkout-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

```bash
streamlit run smart_checkout_app.py
```

---

## 📊 Model Training and Inference

To recreate or fine-tune the model, refer to the training notebook included in the repository. It demonstrates how the model was trained using YOLOv8 with early stopping, model checkpointing, and validation.

Example inference usage:

```python
results = model.predict("path/to/image.jpg", conf=0.5)
results[0].show()
```

---

## 🛋️ Price Assignment

You can set a dictionary of class prices in `utils.py`, for example:

```python
CLASS_PRICES = {
    'coca cola bottle': 25,
    'box': 100,
    'fanta bottle': 20,
    # ...
}
```

The Streamlit app sums detected items based on this mapping.

---

## 📉 Performance

| Metric    | Value   |
| --------- | ------- |
| mAP50     | \~0.99  |
| mAP50-95  | \~0.85  |
| Precision | \~0.986 |
| Recall    | \~0.981 |
| Classes   | 22      |
| Samples   | 20,112  |

---

## 🚤 Future Work

* Add barcode reader fallback for ambiguous products
* Improve handling of overlapping detections
* Add real-time camera-based detection
* Build deployment-ready REST API

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Roboflow](https://roboflow.com/) for dataset generation and formatting

---

## 🪪 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

---


*This project was built for portfolio demonstration purposes and showcases TensorFlow/YOLO-based computer vision capabilities in retail environments.*
