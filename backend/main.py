"""
Illegal Parking Detection System
=================================

Pipeline:
    1. YOLOv8 -> vehicle detection
    2. Custom YOLO -> parking sign detection
    3. plate.pt -> license plate localization
    4. EasyOCR -> license plate text recognition
    5. Parking sign -> LEGAL / ILLEGAL / UNKNOWN
    6. Email alert for illegal parking

Models required in backend folder:
    yolov8n.pt
    best.pt
    plate.pt

Run:
    python main.py
"""

from __future__ import annotations

import base64
import os
import re
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

import cv2
import easyocr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Illegal Parking Detection System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL PATHS
# ============================================================

VEHICLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

SIGN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.pt"
)

PLATE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "plate.pt"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.isfile(
    VEHICLE_MODEL_PATH
):
    raise RuntimeError(
        "Vehicle model not found:\n"
        + VEHICLE_MODEL_PATH
    )

if not os.path.isfile(
    SIGN_MODEL_PATH
):
    raise RuntimeError(
        "Parking sign model not found:\n"
        + SIGN_MODEL_PATH
    )

if not os.path.isfile(
    PLATE_MODEL_PATH
):
    raise RuntimeError(
        "License plate model not found:\n"
        + PLATE_MODEL_PATH
    )


# ============================================================
# SETTINGS
# ============================================================

VEHICLE_CONF = float(
    os.getenv(
        "VEHICLE_CONF",
        "0.30"
    )
)

SIGN_CONF = float(
    os.getenv(
        "SIGN_CONF",
        "0.20"
    )
)

PLATE_CONF = float(
    os.getenv(
        "PLATE_CONF",
        "0.15"
    )
)

OCR_ALLOWLIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


SIGN_CLASSES = {
    0: "Fake sign",
    1: "No Parking",
    2: "No Parking Obstructed",
    3: "Parking",
    4: "Parking Obstructed",
}


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

print()
print("Loading vehicle model...")

vehicle_model: Any = YOLO(
    VEHICLE_MODEL_PATH
)

print("Vehicle model loaded.")

print()
print("Loading parking sign model...")

sign_model: Any = YOLO(
    SIGN_MODEL_PATH
)

print("Parking sign model loaded.")

print(
    "Parking sign classes:",
    sign_model.names
)

print()
print("Loading license plate model...")

plate_model: Any = YOLO(
    PLATE_MODEL_PATH
)

print(
    "License plate model loaded."
)

print(
    "License plate classes:",
    plate_model.names
)

print()
print("Loading EasyOCR...")

ocr_reader: Any = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=True
)

print("EasyOCR loaded.")
print("OCR mode: CPU")


# ============================================================
# EMAIL CONFIGURATION
# ============================================================

EMAIL_ENABLED = (
    os.getenv(
        "EMAIL_ENABLED",
        "false"
    ).lower()
    == "true"
)

SMTP_SERVER = os.getenv(
    "SMTP_SERVER",
    "smtp.gmail.com"
)

SMTP_PORT = int(
    os.getenv(
        "SMTP_PORT",
        "587"
    )
)

SENDER_EMAIL = os.getenv(
    "SENDER_EMAIL",
    ""
).strip()

SENDER_PASSWORD = os.getenv(
    "SENDER_PASSWORD",
    ""
).strip()

RECIPIENT_EMAIL = os.getenv(
    "RECIPIENT_EMAIL",
    ""
).strip()


# ============================================================
# SIGN CLASSIFICATION
# ============================================================

def classify_sign_class(
    class_id: int
) -> str:

    if class_id in (1, 2):
        return "NO_PARKING"

    if class_id in (3, 4):
        return "PARKING"

    if class_id == 0:
        return "FAKE_SIGN"

    return "UNKNOWN"


# ============================================================
# VEHICLE DETECTION
# ============================================================

def detect_vehicles(
    image: np.ndarray
) -> list[dict[str, Any]]:

    vehicles: list[
        dict[str, Any]
    ] = []

    try:

        results: Any = (
            vehicle_model.predict(
                source=image,
                conf=VEHICLE_CONF,
                iou=0.50,
                classes=list(
                    VEHICLE_CLASSES.keys()
                ),
                device="cpu",
                verbose=False,
            )
        )

    except Exception as error:

        print(
            "[VEHICLE MODEL ERROR]",
            error
        )

        return vehicles

    for result in results:

        boxes: Any = getattr(
            result,
            "boxes",
            None
        )

        if boxes is None:
            continue

        for box in boxes:

            try:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                x1 = int(
                    coordinates[0]
                )

                y1 = int(
                    coordinates[1]
                )

                x2 = int(
                    coordinates[2]
                )

                y2 = int(
                    coordinates[3]
                )

            except Exception as error:

                print(
                    "[VEHICLE BOX ERROR]",
                    error
                )

                continue

            if class_id not in VEHICLE_CLASSES:
                continue

            vehicles.append(
                {
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "vehicle_type":
                        VEHICLE_CLASSES[
                            class_id
                        ],
                    "confidence":
                        round(
                            confidence,
                            3
                        ),
                    "license_plate": None,
                    "plate_confidence": 0.0,
                    "plate_verified": False,
                    "parking_status":
                        "UNKNOWN",
                    "parking_reason":
                        "Waiting for sign detection",
                    "associated_sign": None,
                }
            )

    vehicles.sort(
        key=lambda item:
            item["confidence"],
        reverse=True
    )

    return vehicles


# ============================================================
# PARKING SIGN DETECTION
# ============================================================

def detect_parking_signs(
    image: np.ndarray
) -> list[dict[str, Any]]:

    signs: list[
        dict[str, Any]
    ] = []

    try:

        results: Any = (
            sign_model.predict(
                source=image,
                conf=SIGN_CONF,
                iou=0.50,
                device="cpu",
                verbose=False,
            )
        )

    except Exception as error:

        print(
            "[SIGN MODEL ERROR]",
            error
        )

        return signs

    image_height = int(
        image.shape[0]
    )

    image_width = int(
        image.shape[1]
    )

    for result in results:

        boxes: Any = getattr(
            result,
            "boxes",
            None
        )

        if boxes is None:
            continue

        for box in boxes:

            try:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                x1 = max(
                    0,
                    int(coordinates[0])
                )

                y1 = max(
                    0,
                    int(coordinates[1])
                )

                x2 = min(
                    image_width,
                    int(coordinates[2])
                )

                y2 = min(
                    image_height,
                    int(coordinates[3])
                )

            except Exception as error:

                print(
                    "[SIGN BOX ERROR]",
                    error
                )

                continue

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = SIGN_CLASSES.get(
                class_id,
                "UNKNOWN"
            )

            sign_type = (
                classify_sign_class(
                    class_id
                )
            )

            print()
            print(
                "=" * 30
            )
            print(
                "PARKING SIGN DETECTED"
            )
            print(
                "Class ID:",
                class_id
            )
            print(
                "Class Name:",
                class_name
            )
            print(
                "Confidence:",
                round(
                    confidence,
                    3
                )
            )
            print(
                "Decision:",
                sign_type
            )
            print(
                "=" * 30
            )

            signs.append(
                {
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "confidence":
                        round(
                            confidence,
                            3
                        ),
                    "class_id":
                        class_id,
                    "class_name":
                        class_name,
                    "sign_type":
                        sign_type,
                }
            )

    return signs


# ============================================================
# OCR TEXT CLEANING
# ============================================================

def clean_ocr_text(
    text: str
) -> str:

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


# ============================================================
# OCR VALIDITY CHECK
# ============================================================

def looks_like_plate(
    text: str
) -> bool:

    cleaned = clean_ocr_text(
        text
    )

    if len(cleaned) < 5:
        return False

    if len(cleaned) > 12:
        return False

    has_letter = bool(
        re.search(
            r"[A-Z]",
            cleaned
        )
    )

    has_digit = bool(
        re.search(
            r"[0-9]",
            cleaned
        )
    )

    if not has_letter:
        return False

    if not has_digit:
        return False

    blocked_words = {
        "PARKING",
        "PARK",
        "STOP",
        "ENTRY",
        "EXIT",
        "SCHOOL",
        "TRUCK",
    }

    if cleaned in blocked_words:
        return False

    return True


# ============================================================
# PLATE IMAGE PREPROCESSING
# ============================================================

def create_plate_variants(
    plate_crop: np.ndarray
) -> list[
    tuple[str, np.ndarray]
]:

    if (
        plate_crop is None
        or plate_crop.size == 0
    ):
        return []

    height = int(
        plate_crop.shape[0]
    )

    width = int(
        plate_crop.shape[1]
    )

    if height < 5 or width < 15:
        return []

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    target_height = 120

    scale = (
        target_height
        / float(height)
    )

    scale = max(
        2.0,
        min(
            scale,
            6.0
        )
    )

    enlarged = cv2.resize(
        plate_crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Mild sharpening
    # --------------------------------------------------------

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return [
        (
            "original",
            enlarged
        ),
        (
            "clahe",
            enhanced
        ),
        (
            "sharp",
            sharpened
        ),
        (
            "otsu",
            otsu
        ),
        (
            "adaptive",
            adaptive
        ),
    ]


# ============================================================
# EASY OCR
# ============================================================

def run_easyocr(
    image: np.ndarray
) -> list[dict[str, Any]]:

    results: list[
        dict[str, Any]
    ] = []

    try:

        output: Any = (
            ocr_reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=OCR_ALLOWLIST,
                decoder="beamsearch",
                beamWidth=5,
                text_threshold=0.25,
                low_text=0.15,
                link_threshold=0.20,
                width_ths=0.70,
                height_ths=0.70,
                mag_ratio=1.0,
            )
        )

    except Exception as error:

        print(
            "[OCR ERROR]",
            error
        )

        return results

    if not isinstance(
        output,
        list
    ):
        return results

    for item in output:

        if not isinstance(
            item,
            (list, tuple)
        ):
            continue

        if len(item) < 3:
            continue

        try:

            raw_text = str(
                item[1]
            )

            confidence = float(
                item[2]
            )

        except (
            ValueError,
            TypeError,
        ):

            continue

        cleaned = clean_ocr_text(
            raw_text
        )

        if not looks_like_plate(
            cleaned
        ):
            continue

        results.append(
            {
                "text": cleaned,
                "confidence":
                    confidence,
            }
        )

    return results


# ============================================================
# PLATE OCR CONSENSUS
# ============================================================

def choose_plate_text(
    readings: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:

    if not readings:
        return None

    groups: dict[
        str,
        list[float]
    ] = {}

    for reading in readings:

        text = str(
            reading["text"]
        )

        confidence = float(
            reading["confidence"]
        )

        groups.setdefault(
            text,
            []
        ).append(
            confidence
        )

    candidates: list[
        dict[str, Any]
    ] = []

    for text, confidences in groups.items():

        best_confidence = max(
            confidences
        )

        average_confidence = (
            sum(confidences)
            / len(confidences)
        )

        repetitions = len(
            confidences
        )

        # Confidence is primary.
        # Repetition is secondary.
        score = (
            best_confidence * 0.60
            + average_confidence * 0.25
            + min(
                repetitions,
                4
            ) * 0.0375
        )

        candidates.append(
            {
                "text": text,
                "score": score,
                "best_confidence":
                    best_confidence,
                "average_confidence":
                    average_confidence,
                "repetitions":
                    repetitions,
            }
        )

    candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )

    print()
    print(
        "[PLATE OCR RANKING]"
    )

    for candidate in candidates[:8]:

        print(
            f"    "
            f"{candidate['text']:<15}"
            f"score="
            f"{candidate['score']:.3f} "
            f"best="
            f"{candidate['best_confidence']:.3f} "
            f"reads="
            f"{candidate['repetitions']}"
        )

    best = candidates[0]

    if (
        best["best_confidence"]
        < 0.20
    ):
        return None

    return {
        "plate":
            best["text"],
        "confidence":
            round(
                best[
                    "best_confidence"
                ],
                3
            ),
        "verified":
            best[
                "repetitions"
            ] >= 2,
    }


# ============================================================
# LICENSE PLATE DETECTION
# ============================================================

def detect_license_plate(
    vehicle_crop: np.ndarray,
) -> Optional[
    dict[str, Any]
]:

    if (
        vehicle_crop is None
        or vehicle_crop.size == 0
    ):
        return None

    vehicle_height = int(
        vehicle_crop.shape[0]
    )

    vehicle_width = int(
        vehicle_crop.shape[1]
    )

    if (
        vehicle_width < 50
        or vehicle_height < 30
    ):
        print(
            "[PLATE] vehicle too small"
        )
        return None

    # ========================================================
    # 1. PLATE MODEL
    # ========================================================

    try:

        results: Any = (
            plate_model.predict(
                source=vehicle_crop,
                conf=PLATE_CONF,
                iou=0.50,
                imgsz=640,
                device="cpu",
                verbose=False,
            )
        )

    except Exception as error:

        print(
            "[PLATE MODEL ERROR]",
            error
        )

        return None

    if not results:
        return None

    result: Any = results[0]

    boxes: Any = getattr(
        result,
        "boxes",
        None
    )

    if boxes is None:
        print(
            "[PLATE] No boxes returned"
        )
        return None

    if len(boxes) == 0:

        print(
            "[PLATE] No license plate detected"
        )

        return None

    # ========================================================
    # 2. SELECT HIGHEST CONFIDENCE PLATE
    # ========================================================

    best_box: Optional[
        list[int]
    ] = None

    best_model_confidence = 0.0

    for box in boxes:

        try:

            confidence = float(
                box.conf[0].item()
            )

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            x1 = int(
                coordinates[0]
            )

            y1 = int(
                coordinates[1]
            )

            x2 = int(
                coordinates[2]
            )

            y2 = int(
                coordinates[3]
            )

        except Exception as error:

            print(
                "[PLATE BOX ERROR]",
                error
            )

            continue

        if confidence > best_model_confidence:

            best_model_confidence = (
                confidence
            )

            best_box = [
                x1,
                y1,
                x2,
                y2,
            ]

    if best_box is None:
        return None

    x1, y1, x2, y2 = (
        best_box
    )

    # ========================================================
    # 3. CLAMP BOX
    # ========================================================

    x1 = max(
        0,
        min(
            x1,
            vehicle_width - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            vehicle_height - 1
        )
    )

    x2 = max(
        1,
        min(
            x2,
            vehicle_width
        )
    )

    y2 = max(
        1,
        min(
            y2,
            vehicle_height
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    # ========================================================
    # 4. VERY SMALL PADDING
    # ========================================================
    #
    # We deliberately keep this small.
    # Large padding makes EasyOCR read the car body.
    # ========================================================

    plate_width = x2 - x1
    plate_height = y2 - y1

    pad_x = max(
        2,
        int(
            plate_width * 0.04
        )
    )

    pad_y = max(
        2,
        int(
            plate_height * 0.08
        )
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        vehicle_width,
        x2 + pad_x
    )

    y2 = min(
        vehicle_height,
        y2 + pad_y
    )

    plate_crop = vehicle_crop[
        y1:y2,
        x1:x2
    ]

    if plate_crop.size == 0:
        return None

    print()
    print(
        "[PLATE MODEL]"
    )

    print(
        "Confidence:",
        round(
            best_model_confidence,
            3
        )
    )

    print(
        "Box:",
        x1,
        y1,
        x2,
        y2
    )

    # ========================================================
    # 5. SAVE DEBUG PLATE CROP
    # ========================================================

    debug_folder = os.path.join(
        BASE_DIR,
        "processed",
        "plate_debug"
    )

    os.makedirs(
        debug_folder,
        exist_ok=True
    )

    debug_path = os.path.join(
        debug_folder,
        "latest_plate.jpg"
    )

    cv2.imwrite(
        debug_path,
        plate_crop
    )

    print(
        "Plate crop:",
        debug_path
    )

    # ========================================================
    # 6. CREATE OCR VARIANTS
    # ========================================================

    variants = create_plate_variants(
        plate_crop
    )

    if not variants:
        return None

    # ========================================================
    # 7. OCR
    # ========================================================

    all_readings: list[
        dict[str, Any]
    ] = []

    for variant_name, variant_image in variants:

        readings = run_easyocr(
            variant_image
        )

        for reading in readings:

            print(
                "    OCR:",
                reading["text"],
                "confidence=",
                round(
                    reading[
                        "confidence"
                    ],
                    3
                ),
                "variant=",
                variant_name,
            )

            all_readings.append(
                {
                    "text":
                        reading["text"],
                    "confidence":
                        reading[
                            "confidence"
                        ],
                    "variant":
                        variant_name,
                }
            )

    if not all_readings:

        print(
            "[PLATE] OCR found no valid text"
        )

        return None

    # ========================================================
    # 8. SELECT FINAL PLATE
    # ========================================================

    selected = choose_plate_text(
        all_readings
    )

    if selected is None:

        print(
            "[PLATE] No reliable OCR result"
        )

        return None

    print()
    print(
        "[PLATE FINAL]",
        selected["plate"]
    )

    print(
        "OCR confidence:",
        selected["confidence"]
    )

    print(
        "Repeated:",
        selected["verified"]
    )

    return selected


# ============================================================
# FIND NEAREST PARKING SIGN
# ============================================================

def find_sign_for_vehicle(
    vehicle_bbox: list[int],
    signs: list[dict[str, Any]],
    image_width: int,
) -> Optional[
    dict[str, Any]
]:

    if not signs:
        return None

    vx1, vy1, vx2, vy2 = [
        float(value)
        for value in vehicle_bbox
    ]

    vehicle_center_x = (
        vx1 + vx2
    ) / 2.0

    vehicle_bottom = vy2

    max_horizontal = max(
        250.0,
        image_width * 0.35
    )

    candidates: list[
        tuple[
            float,
            dict[str, Any]
        ]
    ] = []

    for sign in signs:

        sx1, sy1, sx2, sy2 = [
            float(value)
            for value in sign["bbox"]
        ]

        sign_center_x = (
            sx1 + sx2
        ) / 2.0

        sign_center_y = (
            sy1 + sy2
        ) / 2.0

        horizontal_distance = abs(
            vehicle_center_x
            - sign_center_x
        )

        vertical_distance = abs(
            vehicle_bottom
            - sign_center_y
        )

        if (
            horizontal_distance
            > max_horizontal
        ):
            continue

        distance = (
            horizontal_distance
            * 0.65
            +
            vertical_distance
            * 0.35
        )

        candidates.append(
            (
                distance,
                sign
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item[0]
    )

    return candidates[0][1]


# ============================================================
# PARKING DECISION
# ============================================================

def decide_vehicle_status(
    vehicle: dict[str, Any],
    signs: list[dict[str, Any]],
    image_width: int,
) -> tuple[
    str,
    str,
    Optional[
        dict[str, Any]
    ],
]:

    sign = find_sign_for_vehicle(
        vehicle["bbox"],
        signs,
        image_width
    )

    if sign is None:

        return (
            "UNKNOWN",
            "No nearby parking sign detected",
            None,
        )

    sign_type = str(
        sign["sign_type"]
    )

    sign_class = str(
        sign["class_name"]
    )

    if sign_type == "NO_PARKING":

        return (
            "ILLEGAL",
            f"{sign_class} detected near vehicle",
            sign,
        )

    if sign_type == "PARKING":

        return (
            "LEGAL",
            f"{sign_class} detected near vehicle",
            sign,
        )

    if sign_type == "FAKE_SIGN":

        return (
            "UNKNOWN",
            "Fake parking sign detected",
            sign,
        )

    return (
        "UNKNOWN",
        "Parking sign class could not be determined",
        sign,
    )


# ============================================================
# DRAW PARKING SIGNS
# ============================================================

WHITE = (
    255,
    255,
    255
)


def draw_signs(
    image: np.ndarray,
    signs: list[dict[str, Any]],
) -> None:

    for sign in signs:

        x1, y1, x2, y2 = [
            int(value)
            for value in sign["bbox"]
        ]

        label = (
            f"{sign['class_name']} "
            f"{float(sign['confidence']):.2f}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            WHITE,
            3,
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    30,
                    y1 - 10
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            WHITE,
            2,
        )


# ============================================================
# DRAW VEHICLES
# ============================================================

def draw_vehicles(
    image: np.ndarray,
    vehicles: list[dict[str, Any]],
) -> None:

    for vehicle in vehicles:

        x1, y1, x2, y2 = [
            int(value)
            for value in vehicle["bbox"]
        ]

        label = (
            f"{vehicle['vehicle_type']} - "
            f"{vehicle['parking_status']}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            WHITE,
            3,
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    30,
                    y1 - 10
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            WHITE,
            2,
        )

        plate = vehicle.get(
            "license_plate"
        )

        if plate:

            cv2.putText(
                image,
                "Plate: "
                + str(plate),
                (
                    x1,
                    min(
                        image.shape[0] - 10,
                        y2 + 25
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                WHITE,
                2,
            )


# ============================================================
# EMAIL
# ============================================================

def send_email(
    image: np.ndarray,
    illegal_vehicles: list[
        dict[str, Any]
    ],
) -> bool:

    if not EMAIL_ENABLED:

        print(
            "[EMAIL] Disabled"
        )

        return False

    if not illegal_vehicles:

        print(
            "[EMAIL] No illegal vehicles"
        )

        return False

    if (
        not SENDER_EMAIL
        or not SENDER_PASSWORD
        or not RECIPIENT_EMAIL
    ):

        print(
            "[EMAIL] Configuration missing"
        )

        return False

    try:

        message = MIMEMultipart()

        message["From"] = (
            SENDER_EMAIL
        )

        message["To"] = (
            RECIPIENT_EMAIL
        )

        message["Subject"] = (
            "Illegal Parking Detected"
        )

        body = [
            "Illegal parking detected.",
            "",
            "Time: "
            + datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "",
        ]

        for index, vehicle in enumerate(
            illegal_vehicles,
            1
        ):

            plate = (
                vehicle.get(
                    "license_plate"
                )
                or "Not detected"
            )

            body.append(
                f"{index}. "
                f"{vehicle['vehicle_type']} "
                f"- Plate: {plate} "
                f"- OCR confidence: "
                f"{vehicle['plate_confidence']}"
            )

            body.append(
                "Reason: "
                + vehicle[
                    "parking_reason"
                ]
            )

        message.attach(
            MIMEText(
                "\n".join(body),
                "plain"
            )
        )

        success, buffer = (
            cv2.imencode(
                ".jpg",
                image
            )
        )

        if success:

            attachment = MIMEImage(
                buffer.tobytes(),
                _subtype="jpeg"
            )

            attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=(
                    "illegal_parking.jpg"
                ),
            )

            message.attach(
                attachment
            )

        server = smtplib.SMTP(
            SMTP_SERVER,
            SMTP_PORT
        )

        server.starttls()

        server.login(
            SENDER_EMAIL,
            SENDER_PASSWORD
        )

        server.send_message(
            message
        )

        server.quit()

        print(
            "[EMAIL] Sent successfully"
        )

        return True

    except Exception as error:

        print(
            "[EMAIL ERROR]",
            error
        )

        return False


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(
    image: np.ndarray
) -> str:

    success, buffer = (
        cv2.imencode(
            ".jpg",
            image
        )
    )

    if not success:
        return ""

    return base64.b64encode(
        buffer.tobytes()
    ).decode(
        "utf-8"
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "system":
            "Illegal Parking Detection System",
        "vehicle_model":
            "yolov8n.pt",
        "sign_model":
            "best.pt",
        "plate_model":
            "plate.pt",
        "ocr":
            "EasyOCR CPU",
        "plate_pipeline":
            "plate.pt -> EasyOCR",
        "indian_plate_validation":
            False,
        "whole_vehicle_ocr":
            False,
    }


# ============================================================
# DETECT API
# ============================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Image is empty"
            )

        image = cv2.imdecode(
            np.frombuffer(
                contents,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise HTTPException(
                status_code=400,
                detail="Could not read image"
            )

        image_height = int(
            image.shape[0]
        )

        image_width = int(
            image.shape[1]
        )

        print()
        print(
            "=" * 70
        )
        print(
            "PROCESSING IMAGE"
        )
        print(
            "Image size:",
            image_width,
            "x",
            image_height
        )
        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # VEHICLES
        # ----------------------------------------------------

        vehicles = detect_vehicles(
            image
        )

        print(
            "Vehicles detected:",
            len(vehicles)
        )

        # ----------------------------------------------------
        # PARKING SIGNS
        # ----------------------------------------------------

        signs = detect_parking_signs(
            image
        )

        print(
            "Parking signs detected:",
            len(signs)
        )

        # ----------------------------------------------------
        # LICENSE PLATES
        # ----------------------------------------------------

        for index, vehicle in enumerate(
            vehicles,
            1
        ):

            print()
            print(
                "=" * 50
            )

            print(
                f"Vehicle "
                f"{index}/"
                f"{len(vehicles)}"
            )

            print(
                "Type:",
                vehicle[
                    "vehicle_type"
                ]
            )

            print(
                "Vehicle confidence:",
                vehicle[
                    "confidence"
                ]
            )

            print(
                "=" * 50
            )

            bx1, by1, bx2, by2 = [
                int(value)
                for value in vehicle[
                    "bbox"
                ]
            ]

            x1 = max(
                0,
                bx1
            )

            y1 = max(
                0,
                by1
            )

            x2 = min(
                image_width,
                bx2
            )

            y2 = min(
                image_height,
                by2
            )

            if x2 <= x1 or y2 <= y1:

                print(
                    "[PLATE] Invalid vehicle box"
                )

                continue

            vehicle_crop = image[
                y1:y2,
                x1:x2
            ]

            plate = detect_license_plate(
                vehicle_crop
            )

            if plate:

                vehicle[
                    "license_plate"
                ] = plate[
                    "plate"
                ]

                vehicle[
                    "plate_confidence"
                ] = plate[
                    "confidence"
                ]

                vehicle[
                    "plate_verified"
                ] = plate[
                    "verified"
                ]

                print(
                    "Plate:",
                    plate[
                        "plate"
                    ]
                )

                print(
                    "Verified:",
                    plate[
                        "verified"
                    ]
                )

            else:

                vehicle[
                    "license_plate"
                ] = None

                vehicle[
                    "plate_confidence"
                ] = 0.0

                vehicle[
                    "plate_verified"
                ] = False

                print(
                    "Plate: NOT DETECTED"
                )

        # ----------------------------------------------------
        # PARKING DECISION
        # ----------------------------------------------------

        illegal_vehicles: list[
            dict[str, Any]
        ] = []

        for vehicle in vehicles:

            status, reason, sign = (
                decide_vehicle_status(
                    vehicle,
                    signs,
                    image_width
                )
            )

            vehicle[
                "parking_status"
            ] = status

            vehicle[
                "parking_reason"
            ] = reason

            vehicle[
                "associated_sign"
            ] = sign

            print()
            print(
                "VEHICLE DECISION"
            )

            print(
                "Vehicle:",
                vehicle[
                    "vehicle_type"
                ]
            )

            print(
                "Plate:",
                vehicle[
                    "license_plate"
                ]
            )

            print(
                "Parking status:",
                status
            )

            print(
                "Reason:",
                reason
            )

            if status == "ILLEGAL":

                illegal_vehicles.append(
                    vehicle
                )

        # ----------------------------------------------------
        # DRAW RESULT
        # ----------------------------------------------------

        processed = image.copy()

        draw_signs(
            processed,
            signs
        )

        draw_vehicles(
            processed,
            vehicles
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        output_folder = os.path.join(
            BASE_DIR,
            "processed"
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        output_path = os.path.join(
            output_folder,
            "result.jpg"
        )

        saved = cv2.imwrite(
            output_path,
            processed
        )

        if saved:

            print(
                "Processed image saved:",
                output_path
            )

        else:

            print(
                "WARNING: Could not save result image"
            )

        # ----------------------------------------------------
        # EMAIL
        # ----------------------------------------------------

        email_sent = send_email(
            processed,
            illegal_vehicles
        )

        # ----------------------------------------------------
        # BASE64 IMAGE
        # ----------------------------------------------------

        encoded_image = (
            image_to_base64(
                processed
            )
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            "PROCESSING COMPLETE"
        )

        print(
            "Total vehicles:",
            len(vehicles)
        )

        print(
            "Illegal vehicles:",
            len(
                illegal_vehicles
            )
        )

        print(
            "Email sent:",
            email_sent
        )

        print(
            "=" * 70
        )

        return JSONResponse(
            content={
                "success": True,

                "vehicles":
                    vehicles,

                "parking_signs":
                    signs,

                "total_vehicles":
                    len(vehicles),

                "illegal_vehicles":
                    len(
                        illegal_vehicles
                    ),

                "email_sent":
                    email_sent,

                "image":
                    encoded_image,

                "result_image":
                    encoded_image,
            }
        )

    except HTTPException:
        raise

    except Exception as error:

        print()
        print(
            "=" * 70
        )

        print(
            "PROCESSING ERROR"
        )

        print(
            error
        )

        print(
            "=" * 70
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# EVALUATION
# ============================================================

def metric_from_env(
    name: str
) -> str:

    value = os.getenv(
        name,
        "XX.XX"
    ).strip()

    return (
        value
        if value
        else "XX.XX"
    )


MODEL_PERFORMANCE = {

    "Precision":
        metric_from_env(
            "PRECISION"
        ),

    "Recall":
        metric_from_env(
            "RECALL"
        ),

    "mAP@0.50":
        metric_from_env(
            "MAP50"
        ),

    "mAP@0.50:0.95":
        metric_from_env(
            "MAP50_95"
        ),

    "F1-score":
        metric_from_env(
            "F1_SCORE"
        ),

    "Accuracy":
        metric_from_env(
            "ACCURACY"
        ),

    "Average Processing Time":
        metric_from_env(
            "AVERAGE_PROCESSING_TIME_MS"
        ) + " ms/image",

    "FPS":
        metric_from_env(
            "FPS"
        ),

    "OCR Character Accuracy":
        metric_from_env(
            "OCR_CHARACTER_ACCURACY"
        ),

    "Plate Exact-Match Accuracy":
        metric_from_env(
            "PLATE_EXACT_MATCH_ACCURACY"
        ),
}


def print_model_performance():

    print()
    print(
        "=" * 70
    )

    print(
        "MODEL PERFORMANCE EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "Precision                  :",
        MODEL_PERFORMANCE[
            "Precision"
        ],
        "%"
    )

    print(
        "Recall                     :",
        MODEL_PERFORMANCE[
            "Recall"
        ],
        "%"
    )

    print(
        "mAP@0.50                   :",
        MODEL_PERFORMANCE[
            "mAP@0.50"
        ],
        "%"
    )

    print(
        "mAP@0.50:0.95              :",
        MODEL_PERFORMANCE[
            "mAP@0.50:0.95"
        ],
        "%"
    )

    print(
        "F1-score                   :",
        MODEL_PERFORMANCE[
            "F1-score"
        ],
        "%"
    )

    print(
        "Accuracy                   :",
        MODEL_PERFORMANCE[
            "Accuracy"
        ],
        "%"
    )

    print(
        "Average Processing Time    :",
        MODEL_PERFORMANCE[
            "Average Processing Time"
        ]
    )

    print(
        "FPS                        :",
        MODEL_PERFORMANCE[
            "FPS"
        ]
    )

    print(
        "OCR Character Accuracy     :",
        MODEL_PERFORMANCE[
            "OCR Character Accuracy"
        ],
        "%"
    )

    print(
        "Plate Exact-Match Accuracy :",
        MODEL_PERFORMANCE[
            "Plate Exact-Match Accuracy"
        ],
        "%"
    )

    print(
        "=" * 70
    )


@app.get("/evaluation")
def evaluation():

    return {
        "success": True,
        "model_performance":
            MODEL_PERFORMANCE,
        "note":
            "Metrics must be measured on a labelled validation/test dataset.",
    }


# ============================================================
# START SERVER
# ============================================================

print_model_performance()


if __name__ == "__main__":

    import uvicorn

    print()
    print(
        "=" * 70
    )

    print(
        "Illegal Parking Detection System"
    )

    print(
        "URL: http://localhost:8000"
    )

    print(
        "Vehicle model:",
        VEHICLE_MODEL_PATH
    )

    print(
        "Parking sign model:",
        SIGN_MODEL_PATH
    )

    print(
        "License plate model:",
        PLATE_MODEL_PATH
    )

    print(
        "OCR: EasyOCR CPU"
    )

    print(
        "Email enabled:",
        EMAIL_ENABLED
    )

    print(
        "=" * 70
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )