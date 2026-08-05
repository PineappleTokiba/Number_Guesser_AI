# Digit Lab

Draw a digit on a 28×28 grid and classify it using either trained ONNX model.

- `app/static/` contains the browser app, ONNX Runtime Web, and deployable model files.
- `models/` contains the original trained checkpoints, ONNX models, and training scripts.

```bash
npm install
npm start
```

Then open <http://localhost:3001>. Inference runs entirely in the browser using ONNX Runtime Web; the Node server only serves static files. You can also deploy `app/static/` directly to any static host.

`export_models.py` is only needed if you retrain or replace a `.pth` checkpoint. Run it to regenerate the `.onnx` files.
