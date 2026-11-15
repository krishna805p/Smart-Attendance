"""FastAPI routes for the Smart Attendance System."""

import hashlib
import hmac
import base64
import json
import logging
import math
import os
import struct
import threading
import time
from datetime import date as Date, datetime, time as Time, timedelta
from typing import Literal

import cv2
import mysql.connector
import numpy as np
from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, VerifyMismatchError
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from database import TimedConnection, check_connection, get_connection, get_db

app = FastAPI(title="Smart Attendance System API")

logger = logging.getLogger(__name__)
_password_hasher = PasswordHasher()
MAX_FACE_IMAGE_BYTES = 2 * 1024 * 1024
MAX_FACE_REQUEST_BYTES = MAX_FACE_IMAGE_BYTES + 64 * 1024
MIN_FACE_ENROLLMENT_SAMPLES = 15
MAX_FACE_ENROLLMENT_SAMPLES = 20
MAX_ENROLLMENT_IMAGE_BYTES = 1024 * 1024
MAX_ENROLLMENT_REQUEST_BYTES = MAX_FACE_ENROLLMENT_SAMPLES * MAX_ENROLLMENT_IMAGE_BYTES + 256 * 1024
MAX_FACE_IMAGE_PIXELS = 12_000_000
FACE_MATCH_THRESHOLD_DEFAULT = 0.65
_face_service = None
_face_service_lock = threading.Lock()


class FaceUploadSizeLimitMiddleware:
    """Reject oversized face-upload bodies before multipart parsing buffers them."""
    def __init__(self, application, max_bytes: int):
        self.application = application
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        path = scope.get("path")
        request_limits = {
            "/api/student/verify-face": self.max_bytes,
            "/api/student/enroll-face": MAX_ENROLLMENT_REQUEST_BYTES,
        }
        request_limit = request_limits.get(path)
        if scope.get("type") != "http" or scope.get("method") != "POST" or request_limit is None:
            await self.application(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > request_limit:
                    await self._respond(send, 413, b'{"detail":"Face upload exceeds the request size limit."}')
                    return
            except ValueError:
                await self._respond(send, 400, b'{"detail":"Invalid Content-Length header."}')
                return

        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > request_limit:
                await self._respond(send, 413, b'{"detail":"Face upload exceeds the request size limit."}')
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.application(scope, replay_receive, send)

    @staticmethod
    async def _respond(send, status: int, body: bytes):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


# Add before CORS so the CORS middleware remains outermost for error responses.
app.add_middleware(FaceUploadSizeLimitMiddleware, max_bytes=MAX_FACE_REQUEST_BYTES)


def cors_origins(environment: str | None = None) -> list[str]:
    """Build explicit browser origins; never allow a wildcard."""
    environment = (environment or os.getenv("APP_ENV", "development")).strip().lower()
    origins = [origin.strip().rstrip("/") for origin in
               os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
    previews = [origin.strip().rstrip("/") for origin in
                os.getenv("CORS_PREVIEW_ORIGINS", "").split(",") if origin.strip()]
    origins.extend(previews)
    if environment != "production" and os.getenv("CORS_ALLOW_LOCAL_ORIGINS", "true").lower() in {"1", "true", "yes"}:
        origins.extend(("http://localhost:5500", "http://127.0.0.1:5500",
                        "http://localhost:5501", "http://127.0.0.1:5501"))
    return list(dict.fromkeys(origin for origin in origins
                              if origin != "*" and "YOUR_" not in origin))


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class LoginRequest(BaseModel):
    roll_no: str | None = None
    username: str | None = None
    password: str = Field(min_length=1)


class AttendanceCreateRequest(BaseModel):
    roll_no: str
    date: Date | None = None
    time: Time | None = None
    latitude: float = 0.0
    longitude: float = 0.0
    status: Literal["Present", "Late"] = "Present"


class LocationVerifyRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


TOKEN_LIFETIME_SECONDS = 8 * 60 * 60


def _token_secret() -> bytes:
    secret = os.getenv("AUTH_TOKEN_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="Authentication is not configured.")
    return secret.encode("utf-8")


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def create_access_token(role: Literal["student", "teacher"], subject: str) -> str:
    payload = _b64url(json.dumps({
        "role": role,
        "sub": subject,
        "exp": int(time.time()) + TOKEN_LIFETIME_SECONDS,
    }, separators=(",", ":")).encode("utf-8"))
    signature = _b64url(hmac.new(_token_secret(), payload.encode("ascii"), hashlib.sha256).digest())
    return f"{payload}.{signature}"


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        payload_part, signature_part = authorization[7:].split(".", 1)
        expected = _b64url(hmac.new(
            _token_secret(), payload_part.encode("ascii"), hashlib.sha256
        ).digest())
        if not hmac.compare_digest(signature_part, expected):
            raise ValueError("invalid signature")
        padded = payload_part + "=" * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if (not isinstance(payload, dict) or payload.get("role") not in ("student", "teacher")
                or not isinstance(payload.get("sub"), str)
                or int(payload.get("exp", 0)) <= int(time.time())):
            raise ValueError("invalid or expired token")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token.")


def require_role(user: dict, role: str, subject: str | None = None) -> dict:
    if user.get("role") != role:
        raise HTTPException(status_code=403, detail="This account cannot access that resource.")
    if subject is not None and user.get("sub") != subject:
        raise HTTPException(status_code=403, detail="This account cannot access that resource.")
    return user


def hash_password(password: str) -> str:
    """Hash newly set passwords with Argon2id."""
    return _password_hasher.hash(password)


def verify_password(password: str, stored: str) -> bool:
    """Verify Argon2id and legacy PBKDF2/plaintext values during migration."""
    if stored.startswith("$argon2"):
        try:
            return _password_hasher.verify(stored, password)
        except (VerifyMismatchError, Argon2Error, TypeError, ValueError):
            return False
    if not stored.startswith("pbkdf2_sha256$"):
        return hmac.compare_digest(stored, password)
    try:
        _, iterations, salt_hex, digest_hex = stored.split("$", 3)
        expected = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        ).hex()
        return hmac.compare_digest(expected, digest_hex)
    except (TypeError, ValueError):
        return False


def password_needs_rehash(stored: str) -> bool:
    if not stored.startswith("$argon2id$"):
        return True
    try:
        return _password_hasher.check_needs_rehash(stored)
    except (Argon2Error, TypeError, ValueError):
        return True


def upgrade_password_hash(db, table: Literal["students", "teachers"], row_id: int,
                          password: str, old_hash: str) -> None:
    """Upgrade a verified legacy password without changing the account or blocking login."""
    if not password_needs_rehash(old_hash):
        return
    cursor = None
    try:
        cursor = db.cursor()
        cursor.execute(f"UPDATE {table} SET password = %s WHERE id = %s AND password = %s",
                       (hash_password(password), row_id, old_hash))
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("Password hash upgrade failed; login remains available for this existing account.")
    finally:
        if cursor is not None:
            cursor.close()


class LoginRateLimiter:
    """Small process-local sliding-window limiter; use an edge limiter at scale."""
    def __init__(self, limit: int = 5, window_seconds: int = 60, clock=time.monotonic):
        self.limit = limit
        self.window_seconds = window_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._events: dict[str, list[float]] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        now = self.clock()
        with self._lock:
            for old_key in [item for item, values in self._events.items()
                            if not values or now - values[-1] >= self.window_seconds]:
                self._events.pop(old_key, None)
            events = [stamp for stamp in self._events.get(key, [])
                      if now - stamp < self.window_seconds]
            if len(events) >= self.limit:
                retry = max(1, int(self.window_seconds - (now - events[0])))
                self._events[key] = events
                return False, retry
            if len(self._events) >= 10000 and key not in self._events:
                self._events.pop(next(iter(self._events)))
            events.append(now)
            self._events[key] = events
            return True, 0


login_rate_limiter = LoginRateLimiter()
login_ip_rate_limiter = LoginRateLimiter(limit=30)


def enforce_login_limit(request: Request, identity: str) -> None:
    address = request.client.host if request.client else "unknown"
    identity_allowed, identity_retry = login_rate_limiter.allow(
        f"{address}:{identity.strip().casefold()}"
    )
    ip_allowed, ip_retry = login_ip_rate_limiter.allow(address)
    if not identity_allowed or not ip_allowed:
        retry_after = max(identity_retry, ip_retry)
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.",
                            headers={"Retry-After": str(retry_after)})


def format_time(value: Time | timedelta | None) -> str:
    if not value:
        return "--"
    if isinstance(value, timedelta):
        seconds = int(value.total_seconds())
        value = Time(seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return value.strftime("%I:%M %p")


def format_date(value: Date | str) -> str:
    if isinstance(value, str):
        year, month, day = value.split("-")
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return value.isoformat()


def geofence_config() -> tuple[float, float, float]:
    """Read campus coordinates only from the backend environment."""
    try:
        latitude = float(os.environ["CAMPUS_LATITUDE"])
        longitude = float(os.environ["CAMPUS_LONGITUDE"])
        radius_meters = float(os.environ["GEOFENCE_RADIUS_METERS"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=503, detail="Campus geofence is not configured.")

    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180 or radius_meters <= 0:
        raise HTTPException(status_code=503, detail="Campus geofence is not configured.")
    return latitude, longitude, radius_meters


def haversine_distance_meters(
    latitude_one: float, longitude_one: float, latitude_two: float, longitude_two: float
) -> float:
    """Calculate the great-circle distance between two latitude/longitude pairs."""
    earth_radius_meters = 6_371_000
    latitude_delta = math.radians(latitude_two - latitude_one)
    longitude_delta = math.radians(longitude_two - longitude_one)
    a = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(math.radians(latitude_one))
        * math.cos(math.radians(latitude_two))
        * math.sin(longitude_delta / 2) ** 2
    )
    return earth_radius_meters * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def student_payload(student: dict, stats: dict | None = None) -> dict:
    stats = stats or {}
    total_classes = int(stats.get("total_classes") or 0)
    present = int(stats.get("present") or 0)
    late = int(stats.get("late") or 0)
    attended = present + late
    return {
        "id": student["id"],
        "name": student["name"],
        "rollNo": student["roll_number"],
        "course": "BBA - Semester 5",
        "attendancePercent": round((attended / total_classes) * 100) if total_classes else 0,
        "totalClasses": total_classes,
        "present": present,
        "late": late,
        "absent": 0,
    }


def attendance_payload(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "studentId": record["student_id"],
        "rollNo": record.get("roll_number"),
        "studentName": record.get("student_name"),
        "date": format_date(record["date"]),
        "time": format_time(record.get("time")),
        "status": record["status"],
        # GPS and face verification are not implemented yet.
        "method": "GPS + Face placeholder" if record.get("id") else "--",
    }


@app.get("/health")
def health_check():
    """Backward-compatible readiness endpoint; reports DB failure with HTTP 503."""
    db_connected = check_connection()
    if not db_connected:
        return JSONResponse(status_code=503, content={"status": "error", "database": "disconnected"})
    return {"status": "ok", "database": "connected"}


@app.get("/health/live")
def liveness_check():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness_check():
    return health_check()


@app.post("/api/verify-location")
def verify_location(location: LocationVerifyRequest, user: dict = Depends(get_current_user)):
    require_role(user, "student")
    campus_latitude, campus_longitude, radius_meters = geofence_config()
    distance_meters = haversine_distance_meters(
        location.latitude, location.longitude, campus_latitude, campus_longitude
    )
    return {
        "inside_geofence": distance_meters <= radius_meters,
        "distance_meters": round(distance_meters, 1),
        "radius_meters": radius_meters,
    }


def _image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read dimensions from PNG/JPEG headers before decoding pixel data."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 24 or data[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if not data.startswith(b"\xff\xd8"):
        return None

    offset = 2
    start_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in start_of_frame and segment_length >= 7:
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            return width, height
        offset += segment_length
    return None


def _decode_face_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_FACE_IMAGE_BYTES:
        raise ValueError("image size is outside the accepted range")
    dimensions = _image_dimensions(data)
    if not dimensions:
        raise ValueError("only valid JPEG or PNG images are accepted")
    width, height = dimensions
    if width <= 0 or height <= 0 or width * height > MAX_FACE_IMAGE_PIXELS:
        raise ValueError("image dimensions are outside the accepted range")
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("image could not be decoded") from exc
    if image is None or image.shape[:2] != (height, width):
        raise ValueError("image could not be decoded")
    return image


def _get_face_service():
    """Load the selected ONNX models once per process; templates always come from MySQL."""
    global _face_service
    if _face_service is None:
        with _face_service_lock:
            if _face_service is None:
                try:
                    from face_system.detector import RetinaFaceDetector
                    from face_system.recognizer import MobileFaceNetRecognizer

                    _face_service = (RetinaFaceDetector(), MobileFaceNetRecognizer())
                except Exception as exc:
                    logger.error("Face verification models unavailable error_type=%s",
                                 type(exc).__name__)
                    raise HTTPException(status_code=503,
                                        detail="Face verification is temporarily unavailable.") from exc
    return _face_service


def get_face_embedding_store():
    """Request-scoped MySQL store; deliberately has no JSON fallback."""
    connection = None
    try:
        from face_system.db_embedding_store import MySQLEmbeddingStore
        connection = TimedConnection(get_connection())
        store = MySQLEmbeddingStore(connection)
    except Exception as exc:
        if connection is not None:
            connection.close()
        logger.error("Face embedding database unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503,
                            detail="Face enrollment storage is temporarily unavailable.") from exc
    try:
        yield store
    finally:
        connection.close()


def _face_threshold_config() -> tuple[float, bool]:
    try:
        threshold = float(os.getenv("FACE_MATCH_THRESHOLD", str(FACE_MATCH_THRESHOLD_DEFAULT)))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Face verification is not configured.") from exc
    if not -1.0 <= threshold <= 1.0:
        raise HTTPException(status_code=503, detail="Face verification is not configured.")
    calibrated = os.getenv("FACE_MATCH_THRESHOLD_CALIBRATED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    return threshold, calibrated


def verify_face_image(image_bytes: bytes, student_roll: str, store) -> dict:
    """Verify one uploaded face against only the named authenticated student's templates."""
    try:
        image = _decode_face_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detector, recognizer = _get_face_service()
    try:
        from face_system.quality import assess_face_quality
        from face_system.recognition import RecognitionEngine
        from face_system.recognizer import align_face
        faces = detector.detect(image)
    except Exception as exc:
        logger.error("Face verification detector unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503,
                            detail="Face verification is temporarily unavailable.") from exc
    if not faces:
        return {"verified": False, "reason": "no_face",
                "message": "No face was detected. Center your face and try again."}
    if len(faces) != 1:
        return {"verified": False, "reason": "multiple_faces",
                "message": "Exactly one face must be visible."}
    try:
        quality = assess_face_quality(image, faces[0])
    except Exception as exc:
        logger.error("Face verification quality check failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503,
                            detail="Face verification is temporarily unavailable.") from exc
    if not quality.accepted:
        return {"verified": False, "reason": "poor_quality",
                "message": "The face image did not meet the clarity and lighting requirements."}

    threshold, calibrated = _face_threshold_config()
    try:
        aligned = align_face(image, faces[0].landmarks)
        embedding = recognizer.generate_embedding(aligned)
        student_id = store.get_student_id_by_roll(student_roll)
        matcher = RecognitionEngine(store=store, detector=detector, recognizer=recognizer,
                                    threshold=threshold)
        score, _sample_count = matcher.score_student_id(student_id, embedding)
    except KeyError as exc:
        raise HTTPException(status_code=404,
                            detail="No face enrollment is available for this student.") from exc
    except Exception as exc:
        from face_system.db_embedding_store import (
            EmbeddingStorageUnavailable, StudentRecordNotFound,
        )
        if isinstance(exc, StudentRecordNotFound):
            raise HTTPException(status_code=404, detail="Student record was not found.") from exc
        if isinstance(exc, EmbeddingStorageUnavailable):
            raise HTTPException(status_code=503,
                                detail="Face enrollment storage is temporarily unavailable.") from exc
        logger.error("Face verification inference failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503,
                            detail="Face verification is temporarily unavailable.") from exc
    finally:
        if "embedding" in locals():
            del embedding

    verified = score >= threshold
    return {
        "verified": verified,
        "reason": "match" if verified else "not_matched",
        "score": round(float(score), 4),
        "threshold": threshold,
        "threshold_calibrated": calibrated,
        "message": (
            "Face matched using a provisional threshold; this is not attendance approval."
            if verified and not calibrated else
            "Face matched." if verified else
            "Face did not match this student's enrollment."
        ),
    }


def verify_authenticated_face_image(image_bytes: bytes, user: dict, store) -> dict:
    """Bind verification to the subject in an already-validated bearer token."""
    require_role(user, "student")
    if len(image_bytes) > MAX_FACE_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Face image exceeds the 2 MiB limit.")
    return verify_face_image(image_bytes, user["sub"], store)


@app.post(
    "/api/student/verify-face",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["image"],
                        "properties": {"image": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    },
)
async def verify_student_face(request: Request, user: dict = Depends(get_current_user),
                              store=Depends(get_face_embedding_store)):
    require_role(user, "student")
    try:
        form = await request.form(max_files=1, max_fields=1)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail="Upload one webcam image using multipart field 'image'.") from exc
    try:
        files = form.getlist("image")
        if len(form) != 1 or len(files) != 1 or not hasattr(files[0], "read"):
            raise HTTPException(status_code=400,
                                detail="Upload one webcam image using multipart field 'image'.")
        image_bytes = await files[0].read(MAX_FACE_IMAGE_BYTES + 1)
    finally:
        await form.close()
    return verify_authenticated_face_image(image_bytes, user, store)


def enroll_authenticated_student_images(image_bytes_list: list[bytes], user: dict, store) -> dict:
    """Validate a complete batch, then atomically persist embeddings for token subject."""
    require_role(user, "student")
    if not MIN_FACE_ENROLLMENT_SAMPLES <= len(image_bytes_list) <= MAX_FACE_ENROLLMENT_SAMPLES:
        raise HTTPException(status_code=422,
                            detail="Submit between 15 and 20 enrollment images.")
    for data in image_bytes_list:
        if not data or len(data) > MAX_ENROLLMENT_IMAGE_BYTES:
            raise HTTPException(status_code=413,
                                detail="An enrollment image is empty or exceeds the 1 MiB limit.")

    try:
        student_id = store.get_student_id_by_roll(user["sub"])
        if store.student_has_enrollment(student_id):
            raise HTTPException(status_code=409,
                                detail="Face enrollment already exists for this student.")
    except HTTPException:
        raise
    except Exception as exc:
        from face_system.db_embedding_store import EmbeddingStorageUnavailable, StudentRecordNotFound
        if isinstance(exc, StudentRecordNotFound):
            raise HTTPException(status_code=404, detail="Student record was not found.") from exc
        if isinstance(exc, EmbeddingStorageUnavailable):
            raise HTTPException(status_code=503,
                                detail="Face enrollment storage is temporarily unavailable.") from exc
        logger.error("Face enrollment database unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503,
                            detail="Face enrollment storage is temporarily unavailable.") from exc

    detector, recognizer = _get_face_service()
    from face_system.quality import assess_face_quality
    from face_system.recognizer import align_face
    embeddings = []
    for index, data in enumerate(image_bytes_list, start=1):
        try:
            image = _decode_face_image(data)
            faces = detector.detect(image)
            if len(faces) != 1:
                raise HTTPException(status_code=422,
                                    detail=f"Enrollment image {index} must contain exactly one face.")
            quality = assess_face_quality(image, faces[0])
            if not quality.accepted:
                raise HTTPException(status_code=422,
                                    detail=f"Enrollment image {index} failed face quality checks.")
            aligned = align_face(image, faces[0].landmarks)
            embedding = recognizer.generate_embedding(aligned)
            from face_system.embedding_store import normalize_embedding
            embeddings.append(normalize_embedding(embedding))
            del embedding
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Face enrollment processing failed error_type=%s", type(exc).__name__)
            raise HTTPException(status_code=503,
                                detail="Face enrollment could not be processed.") from exc

    try:
        stored_count = store.save_student_embeddings(student_id, embeddings)
    except Exception as exc:
        from face_system.db_embedding_store import EnrollmentAlreadyExists, EmbeddingStorageUnavailable
        if isinstance(exc, EnrollmentAlreadyExists):
            raise HTTPException(status_code=409,
                                detail="Face enrollment already exists for this student.") from exc
        if isinstance(exc, EmbeddingStorageUnavailable):
            raise HTTPException(status_code=503,
                                detail="Face enrollment storage is temporarily unavailable.") from exc
        logger.error("Face enrollment storage failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503,
                            detail="Face enrollment storage is temporarily unavailable.") from exc
    return {"enrolled": True, "samples_saved": stored_count,
            "message": "Face enrollment saved successfully."}


@app.post("/api/student/enroll-face", status_code=201)
async def enroll_student_face(request: Request, user: dict = Depends(get_current_user),
                              store=Depends(get_face_embedding_store)):
    require_role(user, "student")
    try:
        form = await request.form(max_files=MAX_FACE_ENROLLMENT_SAMPLES, max_fields=1)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail="Upload 15 to 20 images using repeated multipart field 'images'.") from exc
    try:
        files = form.getlist("images")
        if len(form) != 1 or not MIN_FACE_ENROLLMENT_SAMPLES <= len(files) <= MAX_FACE_ENROLLMENT_SAMPLES \
                or not all(hasattr(item, "read") for item in files):
            raise HTTPException(status_code=400,
                                detail="Upload 15 to 20 images using repeated multipart field 'images'.")
        image_bytes_list = []
        total = 0
        for upload in files:
            content = await upload.read(MAX_ENROLLMENT_IMAGE_BYTES + 1)
            total += len(content)
            if len(content) > MAX_ENROLLMENT_IMAGE_BYTES or total > MAX_ENROLLMENT_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="Enrollment upload exceeds the request limit.")
            image_bytes_list.append(content)
    finally:
        await form.close()
    return enroll_authenticated_student_images(image_bytes_list, user, store)


@app.post("/api/student/login")
def student_login(credentials: LoginRequest, request: Request, db=Depends(get_db)):
    if not credentials.roll_no:
        raise HTTPException(status_code=422, detail="roll_no is required.")
    roll_no = credentials.roll_no.strip()
    enforce_login_limit(request, roll_no)
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, roll_number, name, password FROM students WHERE roll_number = %s",
            (roll_no,),
        )
        student = cursor.fetchone()
    finally:
        cursor.close()
    if not student or not verify_password(credentials.password, student["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    upgrade_password_hash(db, "students", student["id"], credentials.password, student["password"])
    result = student_payload(student)
    result["access_token"] = create_access_token("student", student["roll_number"])
    result["token_type"] = "bearer"
    return result


@app.get("/api/student/{roll_no}")
def get_student(roll_no: str, user: dict = Depends(get_current_user), db=Depends(get_db)):
    if user.get("role") == "student":
        require_role(user, "student", roll_no)
    else:
        require_role(user, "teacher")
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, roll_number, name FROM students WHERE roll_number = %s",
            (roll_no,),
        )
        student = cursor.fetchone()
        if not student:
            raise HTTPException(status_code=404, detail="Student not found.")
        cursor.execute(
            """
            SELECT COUNT(*) AS total_classes,
                   SUM(status = 'Present') AS present,
                   SUM(status = 'Late') AS late
            FROM attendance WHERE student_id = %s
            """,
            (student["id"],),
        )
        return student_payload(student, cursor.fetchone())
    finally:
        cursor.close()


@app.get("/api/student/{roll_no}/attendance")
def get_student_attendance(
    roll_no: str,
    limit: int | None = Query(None, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    if user.get("role") == "student":
        require_role(user, "student", roll_no)
    else:
        require_role(user, "teacher")
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM students WHERE roll_number = %s", (roll_no,))
        student = cursor.fetchone()
        if not student:
            raise HTTPException(status_code=404, detail="Student not found.")
        query = (
            "SELECT id, student_id, date, time, status FROM attendance "
            "WHERE student_id = %s ORDER BY date DESC, time DESC"
        )
        params: tuple = (student["id"],)
        # Opt-in pagination preserves the legacy unbounded array response by default.
        if limit is not None:
            query += " LIMIT %s OFFSET %s"
            params += (limit, offset)
        cursor.execute(query, params)
        return [attendance_payload(record) for record in cursor.fetchall()]
    finally:
        cursor.close()


@app.post("/api/teacher/login")
def teacher_login(credentials: LoginRequest, request: Request, db=Depends(get_db)):
    if not credentials.username:
        raise HTTPException(status_code=422, detail="username is required.")
    username = credentials.username.strip()
    enforce_login_limit(request, username)
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, username, name, role, password FROM teachers WHERE username = %s",
            (username,),
        )
        teacher = cursor.fetchone()
    finally:
        cursor.close()
    if not teacher or not verify_password(credentials.password, teacher["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    upgrade_password_hash(db, "teachers", teacher["id"], credentials.password, teacher["password"])
    result = {"id": teacher["id"], "name": teacher["name"], "role": teacher["role"], "username": teacher["username"]}
    result["access_token"] = create_access_token("teacher", teacher["username"])
    result["token_type"] = "bearer"
    return result


@app.get("/api/teacher/students")
def get_teacher_students(user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "teacher")
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, roll_number, name FROM students ORDER BY roll_number")
        return [{"id": row["id"], "rollNo": row["roll_number"], "name": row["name"], "course": "BBA - Sem 5"} for row in cursor.fetchall()]
    finally:
        cursor.close()


@app.get("/api/teacher/attendance")
def get_teacher_attendance(
    date_filter: Date | None = Query(None, alias="date"),
    status: Literal["Present", "Absent", "Late"] | None = None,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    require_role(user, "teacher")
    cursor = db.cursor(dictionary=True)
    try:
        if date_filter:
            query = """
                SELECT a.id, s.id AS student_id, s.roll_number, s.name AS student_name,
                       %s AS date, a.time, COALESCE(a.status, 'Absent') AS status
                FROM students s LEFT JOIN attendance a ON a.student_id = s.id AND a.date = %s
            """
            params: list = [date_filter, date_filter]
            if status:
                query += " WHERE COALESCE(a.status, 'Absent') = %s"
                params.append(status)
            query += " ORDER BY s.roll_number"
            cursor.execute(query, tuple(params))
        else:
            query = """
                SELECT a.id, a.student_id, s.roll_number, s.name AS student_name, a.date, a.time, a.status
                FROM attendance a JOIN students s ON s.id = a.student_id
            """
            params = []
            if status and status != "Absent":
                query += " WHERE a.status = %s"
                params.append(status)
            elif status == "Absent":
                return []
            query += " ORDER BY a.date DESC, a.time DESC"
            cursor.execute(query, tuple(params))
        return [attendance_payload(record) for record in cursor.fetchall()]
    finally:
        cursor.close()


@app.post("/api/attendance", status_code=201)
def create_attendance(attendance: AttendanceCreateRequest,
                      user: dict = Depends(get_current_user), db=Depends(get_db)):
    require_role(user, "student", attendance.roll_no.strip())
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM students WHERE roll_number = %s", (attendance.roll_no.strip(),))
        student = cursor.fetchone()
        if not student:
            raise HTTPException(status_code=404, detail="Student not found.")
        attendance_date = attendance.date or Date.today()
        attendance_time = attendance.time or datetime.now().time().replace(microsecond=0)
        try:
            cursor.execute(
                """
                INSERT INTO attendance (student_id, date, time, latitude, longitude, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (student["id"], attendance_date, attendance_time, attendance.latitude, attendance.longitude, attendance.status),
            )
            db.commit()
        except mysql.connector.Error as error:
            db.rollback()
            if error.errno == 1062:
                raise HTTPException(status_code=409, detail="Attendance is already recorded for this student and date.")
            raise HTTPException(status_code=500, detail="Unable to create attendance record.")
        return {
            "message": "Attendance marked successfully.",
            "attendance": {
                "id": cursor.lastrowid,
                "student_id": student["id"],
                "roll_no": attendance.roll_no.strip(),
                "date": attendance_date.isoformat(),
                "time": format_time(attendance_time),
                "status": attendance.status,
            },
        }
    finally:
        cursor.close()
