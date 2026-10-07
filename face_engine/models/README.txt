Put the real MiniFASNet ONNX weight files here to switch on real
anti-spoofing (instead of the default offline sharpness heuristic):

    2.7_80x80_MiniFASNetV2.onnx
    4_0_0_80x80_MiniFASNetV1SE.onnx

Where to get them:
  1. Clone the official repo: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  2. The original weights are .pth (PyTorch) files under
     resources/anti_spoof_models/. Export/convert them to ONNX using the
     model definitions in src/model_lib in that repo (torch.onnx.export),
     input size 80x80, 3-channel.
  3. Drop the two resulting .onnx files in this folder with the exact
     filenames above.

The app checks for these two files automatically on every liveness

check (face_engine/liveness.py -> _minifasnet_available()) and switches
from the heuristic to real MiniFASNet inference the moment both files
exist here -- no code changes or restart-time configuration needed.
