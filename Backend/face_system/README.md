# RetinaFace + Qualcomm MobileFaceNet prototype

This isolated prototype runs RetinaFace MobileNetV2 and Qualcomm MobileFaceNet
with the ONNX Runtime CPU provider. It detects and aligns faces and computes
128D embeddings. Local student embedding enrollment and open-set matching are
available as a prototype. It does not connect to attendance.

## Environment

From this directory, install into the module's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The selected ONNX graph is from the upstream
[yakhyo/retinaface-pytorch release v0.0.1](https://github.com/yakhyo/retinaface-pytorch/releases/tag/v0.0.1).
Its model input is dynamic `[1, 3, H, W]` float32. Inference follows the
upstream implementation: BGR input, subtract `[104, 117, 123]`, then HWC to
NCHW. Anchors use MobileNetV2 RetinaFace settings (strides 8/16/32, sizes
16/32, 64/128, 256/512; variances 0.1/0.2). The outputs are location offsets,
two-class scores, and five landmark offsets. Boxes and landmarks are decoded
against those anchors and scaled back to frame pixels, then filtered with
confidence and NMS.

## Run

```powershell
.\.venv\Scripts\python.exe .\webcam_demo.py
```

Select another camera with `--camera 1`, or adjust the confidence floor with
`--confidence 0.6`. Press `Q` or `Esc` to quit. The demo does not save frames.

The upstream inference and configuration references are
[detect.py](https://github.com/yakhyo/retinaface-pytorch/blob/main/detect.py),
[config.py](https://github.com/yakhyo/retinaface-pytorch/blob/main/config.py),
[prior box generation](https://github.com/yakhyo/retinaface-pytorch/blob/main/layers/functions/prior_box.py),
and [box decoding](https://github.com/yakhyo/retinaface-pytorch/blob/main/utils/box_utils.py).

## MobileFaceNet

The universal float model comes from the official
[Qualcomm MobileFaceNet model card](https://huggingface.co/qualcomm/MobileFaceNet)
and the [release v0.66.0 ONNX float archive](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mobile_facenet/releases/v0.66.0/mobile_facenet-onnx-float.zip). The extracted ONNX graph is
`models/mobilefacenet.onnx`; keep `models/mobile_facenet.data` beside it because
the graph stores its weights in ONNX external tensor data.

The graph inputs are `img1` and `img2`, each float32 `[1,3,112,112]` RGB in
`[0,1]`; the output is `embeddings`, float32 `[2,128]`. Qualcomm's graph path
normalizes with ImageNet mean `(0.485,0.456,0.406)` and standard deviation
`(0.229,0.224,0.225)`, and applies original plus horizontal-flip inference.
The demo passes the same aligned face to both inputs and normalizes the
resulting 128D vector. Face alignment follows Qualcomm's documented 112x112
five-landmark reference geometry and padding settings.

Run the model smoke check (synthetic crop only; not an identity-recognition
test):

```powershell
.\.venv\Scripts\python.exe .\verify_models.py
```

Run the combined webcam embedding demo:

```powershell
.\.venv\Scripts\python.exe .\recognition_demo.py
```

Press `Q` or `Esc` to exit. It does not save frames or embeddings. Qualcomm's
release and model wrapper references are
[MobileFaceNet model card](https://huggingface.co/qualcomm/MobileFaceNet),
[qualcomm/ai-hub-models v0.66.0 model definition](https://github.com/qualcomm/ai-hub-models/blob/v0.66.0/src/qai_hub_models/models/mobile_facenet/model.py),
and [official app preprocessing and alignment](https://github.com/qualcomm/ai-hub-models/blob/v0.66.0/src/qai_hub_models/models/mobile_facenet/app.py).

## Local enrollment and recognition prototype

Enroll 15 valid samples (press Space for each one):

```powershell
.\.venv\Scripts\python.exe .\enrollment_demo.py --roll YOUR_ROLL_NUMBER
```

The accepted range is 15-20 samples. Each capture requires one detected face,
minimum face size, adequate Laplacian sharpness, and acceptable mean brightness.
Press `Q` or `Esc` to cancel; cancellation does not save partial samples. Use
`--replace` only to replace an existing roll number. Only normalized embeddings
are written to `data/student_embeddings.json`; webcam frames are not saved.

Run local recognition with:

```powershell
.\.venv\Scripts\python.exe .\live_recognition_demo.py
```

The initial cosine threshold (`0.65`) is **provisional**, not validated for
attendance decisions. Calibrate it using separate genuine same-person and
impostor pairs captured under expected lighting, angles, and camera conditions;
select a threshold from the resulting score distributions and measure false
accept and false reject rates. If the same roll number is the top accepted
candidate for multiple faces in one frame, all those faces are shown as Unknown
for that frame rather than guessing.

Enable per-face candidate diagnostics (best and second-best score, margin,
template counts, threshold result, and Unknown status) with:

```powershell
.\.venv\Scripts\python.exe .\live_recognition_demo.py --diagnostics
```

An optional best-versus-runner-up margin can be supplied with
`--minimum-margin VALUE`. It is disabled by default because the current store
contains only one roll number, so a runner-up margin cannot help reject an
unregistered person in the single-roll case. Exact score ties among multiple
rolls are rejected as ambiguous.

Per-roll similarity currently uses the mean of the three highest scores among
that roll's templates. With 15 templates, the other 12 scores do not affect the
decision; this can emphasize a high-score outlier. The evaluator deliberately
uses the same rule so threshold results reflect current behavior. Whether this
aggregation contributed to a particular false acceptance cannot be determined
without that impostor's score or embedding (neither is stored by the demo).

### Accuracy evaluation protocol

`capture_protocol.py` captures consented evaluation stills from the webcam and
writes only per-candidate similarity scores and quality metadata; it never saves
frames or embeddings. Use stable anonymous participant labels beginning with
`person_`, and freeze the enrolled gallery throughout the evaluation. Capture
enrollment samples first, then use new capture sessions for calibration and
untouched test data. Do not reuse a session ID across partitions. Each camera
capture is stored as one capture group, not treated as a separate person.

Example calibration captures (repeat for each consenting person and condition):

```powershell
.\.venv\Scripts\python.exe .\capture_protocol.py `
  --participant person_01 --session session_day1 --partition calibration `
  --condition baseline --expected-roll YOUR_ENROLLED_ROLL --consent-confirmed
```

For a consenting unregistered person, replace `--expected-roll ...` with
`--open-set`. To gather held-out test samples, use `--partition test` and new
session IDs. Conditions include `baseline`, `left_angle`, `right_angle`,
`lighting_change`, and `different_distance`. Capture session labels should
reflect genuinely separate sessions (for example different days), not merely
separate commands from one continuous webcam session.

Analyze scores without reading enrollment records or embeddings:

```powershell
.\.venv\Scripts\python.exe .\analyze_protocol.py `
  --scores .\data\evaluation_protocol.csv --frozen-threshold 0.65
```

The report compares production top-three mean, full mean, and maximum
aggregation; includes pairwise score distributions, ROC/EER, threshold FAR/FRR,
open-set capture FAR/FRR and candidate margins; and evaluates test sessions at
the frozen threshold/margin. It also enforces calibration/test session
separation. The quality-weighted calculation is exposed only when per-template
quality metadata is supplied. Existing enrollment JSON stores embeddings but
not per-sample quality, so it is currently not measurable and is not used to
change production matching. Results mark calibration incomplete until there
are at least 20 distinct genuine people and 20 distinct impostor people in each
partition across at least two sessions per role; this is a minimum heuristic,
not proof of a low FAR. No automatic threshold or strategy recommendation is
made.

The existing 15–20 sample enrollment count has no empirical sufficiency result
for this project. The current workflow only debounces captures by 0.35 seconds;
it does not reject near-duplicate embeddings or ensure pose/lighting coverage.
Use a deliberate set across straight, slight left/right, natural expression,
normal lighting variation, and distance. Collect only sharp, adequately lit,
well-framed faces, spread across time; review sample diversity before replacing
an enrollment. Current quality gates are face box >=80 px per side, Laplacian
variance >=45, and crop mean gray brightness 30–225. These are heuristic gates,
not calibrated acceptance criteria.

To run a labeled evaluation, confirm all participants consented and launch:

```powershell
.\.venv\Scripts\python.exe .\evaluate_recognition.py --roll TEST001 --consent-confirmed --output data/evaluation_TEST001.json
```

Use one anonymous participant label per distinct person, mark each as
`genuine` or `impostor` in the preview window. Type the label, press Enter,
then press `G` or `I`. Press Space to capture a valid sample, `N` to finish a
condition and enter the next label, `C` to cancel the current condition, or
`Q`/`Esc` to cancel the active condition and exit. Reuse the same label for the
same person across conditions. The camera stays open between participants.
Evaluation keeps only scores in memory and optional score-only results on disk,
never face images or embeddings. It reports capture rates and person-grouped
rates; person metrics group repeated frames so they do not count as independent
people. Camera reads have a bounded wait and repeated failures stop with a clear
error. Thresholds are candidates, not an automatic recommendation.

Run the storage and synthetic pipeline checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_phase3.py" -v
```

These unit tests use synthetic vectors/images; they do not demonstrate real
student recognition. Enrollment and evaluation JSON files are ignored by this
module's `.gitignore`.

### Two-enrolled-student identification experiment

Make a byte-verified backup of `data/student_embeddings.json` before adding an
enrollment. Enroll a consenting second person without `--replace`:

```powershell
.\.venv\Scripts\python.exe .\enrollment_demo.py --roll TEST002 --samples 15
```

The enrollment workflow refuses an existing roll by default and stores only
embeddings. Run the two-identity experiment after both enrollments exist:

```powershell
.\.venv\Scripts\python.exe .\evaluate_two_students.py --rolls TEST001,TEST002 --consent-confirmed --output data/two_student_results.json
```

In the preview, press `1` or `2` for the person currently in frame, then press
Space for valid captures. `B`, `P`, and `L` tag baseline, modest head rotation,
and lighting-change conditions. Press `N` to select a person again and `F` to
finish. Use `--include-unregistered` to enable a separate open-set capture via
key `3`. Results include best and second candidate, score margin, thresholded
prediction, confusion matrix, and per-person counts. The current threshold is
reported as provisional; no margin gate is enabled. Captures are scores only;
raw frames and generated embeddings are not written to the report.

### Authenticated student face check

The FastAPI backend exposes `POST /api/student/verify-face` with one multipart
field named `image`. It derives the student's roll number from the verified
bearer-token subject, requires exactly one quality-approved face, and compares
only with that student's local enrollment using `score_roll()`. The upload is
processed in memory and is not retained. The response contains a similarity
score and threshold status, never embeddings or template data. A successful
response does not submit attendance.

The backend reads `FACE_MATCH_THRESHOLD` (default `0.65`) and
`FACE_MATCH_THRESHOLD_CALIBRATED` (default `false`). Until a threshold is
calibrated using separate genuine and impostor evaluation captures, a passing
result is explicitly provisional and is not approved for attendance decisions.
