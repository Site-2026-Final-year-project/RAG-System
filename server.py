from __future__ import annotations

import base64
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Tuple

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, or_, select, text

from database import ChatMessageModel, ChatSessionModel, SessionLocal
from scripts.chat import RAGAssistant


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


JWT_SECRET = os.environ.get("JWT_SECRET", "").strip()
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "").strip() or None
JWT_ISSUER = os.environ.get("JWT_ISSUER", "").strip() or None

bearer_scheme = HTTPBearer(auto_error=False)


def _b64url_decode_padded(s: str) -> bytes:
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.urlsafe_b64decode(s.encode("ascii"))


def encode_before_cursor(created_at: datetime, message_id: str) -> str:
    payload = {"t": created_at.isoformat(), "id": message_id}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_before_cursor(cursor: str) -> Tuple[datetime, str]:
    try:
        raw = _b64url_decode_padded(cursor)
        data = json.loads(raw.decode("utf-8"))
        ts = datetime.fromisoformat(str(data["t"]))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        mid = str(data["id"])
        return ts, mid
    except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid before cursor: {e}") from e


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str:
    if JWT_SECRET:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid Authorization header (Bearer token required).",
            )
        try:
            decode_kwargs: Dict[str, Any] = {"algorithms": [JWT_ALGORITHM]}
            if JWT_AUDIENCE:
                decode_kwargs["audience"] = JWT_AUDIENCE
            if JWT_ISSUER:
                decode_kwargs["issuer"] = JWT_ISSUER
            payload = jwt.decode(credentials.credentials, JWT_SECRET, **decode_kwargs)
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e
        user_id = payload.get("sub") or payload.get("id")
        if not user_id or not str(user_id).strip():
            raise HTTPException(status_code=401, detail="Token missing 'sub' (or 'id') claim.")
        return str(user_id).strip()

    if x_user_id and x_user_id.strip():
        return x_user_id.strip()
    raise HTTPException(
        status_code=401,
        detail="Set JWT_SECRET and send Authorization: Bearer <jwt>, or for local dev send X-User-Id.",
    )


class VehicleHealthComponent(BaseModel):
    label: str
    percent: int = Field(ge=0, le=100)


class VehicleHealthPayload(BaseModel):
    overallPercent: int = Field(ge=0, le=100)
    summary: str | None = None
    components: List[VehicleHealthComponent] = Field(default_factory=list)


class VehiclePayload(BaseModel):
    id: str
    driverId: str | None = None
    plateNumber: str | None = None
    make: str | None = None
    model: str | None = None
    year: int | None = None
    displayName: str | None = None
    type: str | None = None
    color: str | None = None
    vin: str | None = None
    mileage: int | None = None
    fuelType: str | None = None
    status: str | None = None
    imageUrl: str | None = None
    insuranceDocumentUrl: str | None = None
    insuranceExpiresAt: datetime | None = None
    registrationDocumentUrl: str | None = None
    registrationExpiresAt: datetime | None = None
    createdAt: datetime | None = None
    updatedAt: datetime | None = None


class VehicleContextPayload(BaseModel):
    vehicle: VehiclePayload | None = None
    vehicle_health: VehicleHealthPayload | None = None
    snapshot_at: datetime | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default="New chat", max_length=255)
    vehicle_id: str | None = None
    car_context: str = Field(default="", max_length=255)
    vehicle_context: VehicleContextPayload | None = None
    vehicle: VehiclePayload | None = None
    vehicle_health: VehicleHealthPayload | None = None


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)
    car_context: str = Field(default="", max_length=255)
    vehicle_id: str | None = None
    title: str | None = None
    vehicle_context: VehicleContextPayload | None = None
    vehicle: VehiclePayload | None = None
    vehicle_health: VehicleHealthPayload | None = None
    use_user_manual: bool = True


class ChatSessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    car_context: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime


class MessagesPageResponse(BaseModel):
    items: List[ChatMessageResponse]
    next_before: str | None = None
    has_more: bool = False


_assistant_cache: Dict[str, RAGAssistant] = {}


def build_assistant_key(user_id: str, car_context: str, use_user_manual: bool) -> str:
    return f"{user_id}::{car_context.strip()}::{use_user_manual}"


def get_assistant(user_id: str, car_context: str, use_user_manual: bool) -> RAGAssistant:
    key = build_assistant_key(user_id, car_context, use_user_manual)
    cached = _assistant_cache.get(key)
    if cached is not None:
        return cached

    assistant = RAGAssistant(
        user_id=user_id,
        car_context=car_context,
        use_user_manual=use_user_manual,
    )
    _assistant_cache[key] = assistant
    return assistant


def to_session_response(session: ChatSessionModel) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        car_context=session.car_context,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def to_message_response(message: ChatMessageModel) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def _shorten_context(text: str, max_len: int = 255) -> str:
    text = " ".join(text.split()).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def normalize_vehicle_context(payload: Any) -> tuple[str, str]:
    """Return (short_context_for_session, detailed_priority_context_for_retrieval)."""
    vehicle_context = getattr(payload, "vehicle_context", None)
    vehicle = (vehicle_context.vehicle if vehicle_context else None) or getattr(payload, "vehicle", None)
    health = (vehicle_context.vehicle_health if vehicle_context else None) or getattr(
        payload, "vehicle_health", None
    )
    manual_context = (getattr(payload, "car_context", "") or "").strip()

    parts: List[str] = []
    if vehicle is not None:
        label = (
            vehicle.displayName
            or " ".join(
                p for p in [str(vehicle.year) if vehicle.year else "", vehicle.make or "", vehicle.model or ""]
                if p
            ).strip()
        )
        if label:
            parts.append(f"Vehicle: {label}")
        if vehicle.plateNumber:
            parts.append(f"Plate: {vehicle.plateNumber}")
        if vehicle.fuelType:
            parts.append(f"Fuel: {vehicle.fuelType}")
        if vehicle.mileage is not None:
            parts.append(f"Mileage: {vehicle.mileage} km")

    priority_lines: List[str] = []
    if health is not None:
        priority_lines.append(f"Vehicle health overall: {health.overallPercent}%")
        if health.summary:
            priority_lines.append(f"Health summary: {health.summary.strip()}")
        concerning = [c for c in health.components if c.percent <= 60]
        if concerning:
            issues = ", ".join(f"{c.label} {c.percent}%" for c in concerning)
            priority_lines.append(f"Priority issues: {issues}")

    if manual_context:
        parts.append(f"Driver context: {manual_context}")
        priority_lines.append(f"Driver context: {manual_context}")

    short_context = _shorten_context(" | ".join(parts) if parts else manual_context)
    detailed_context = "\n".join(priority_lines + parts).strip()
    return short_context, detailed_context


def _extract_vehicle_id(payload: Any) -> str:
    if getattr(payload, "vehicle_id", None):
        return str(payload.vehicle_id).strip()
    vehicle_context = getattr(payload, "vehicle_context", None)
    if vehicle_context and vehicle_context.vehicle and vehicle_context.vehicle.id:
        return str(vehicle_context.vehicle.id).strip()
    if getattr(payload, "vehicle", None) and payload.vehicle.id:
        return str(payload.vehicle.id).strip()
    return ""


def _health_from_db(raw_health: Any) -> VehicleHealthPayload | None:
    if not isinstance(raw_health, dict):
        return None

    numeric_components: List[VehicleHealthComponent] = []
    for key, value in raw_health.items():
        if key == "custom":
            continue
        if isinstance(value, (int, float)):
            pct = max(0, min(100, int(value)))
            numeric_components.append(
                VehicleHealthComponent(label=str(key).replace("_", " ").title(), percent=pct)
            )

    overall = int(round(sum(c.percent for c in numeric_components) / len(numeric_components))) if numeric_components else 0
    summary = f"{len([c for c in numeric_components if c.percent <= 60])} component(s) need attention."
    return VehicleHealthPayload(
        overallPercent=overall,
        summary=summary,
        components=numeric_components,
    )


def fetch_vehicle_context_from_db(vehicle_id: str, user_id: str) -> tuple[VehiclePayload | None, VehicleHealthPayload | None]:
    if not vehicle_id:
        return None, None

    vehicle_sql_owner = """
    SELECT
      id, "driverId", "plateNumber", make, model, year, type, color, vin, mileage,
      fuel_type, image_url, insurance_document_url, insurance_expires_at,
      registration_document_url, registration_expires_at, "createdAt", "updatedAt"
    FROM "Vehicle"
    WHERE id = :vehicle_id
      AND "driverId" = :user_id
    LIMIT 1
    """
    vehicle_sql_by_id = """
    SELECT
      id, "driverId", "plateNumber", make, model, year, type, color, vin, mileage,
      fuel_type, image_url, insurance_document_url, insurance_expires_at,
      registration_document_url, registration_expires_at, "createdAt", "updatedAt"
    FROM "Vehicle"
    WHERE id = :vehicle_id
    LIMIT 1
    """
    health_sql = """
    SELECT health
    FROM "VehicleMaintenanceHealth"
    WHERE "vehicleId" = :vehicle_id
    ORDER BY "updatedAt" DESC
    LIMIT 1
    """

    with SessionLocal() as db:
        row = db.execute(text(vehicle_sql_owner), {"vehicle_id": vehicle_id, "user_id": user_id}).mappings().first()
        if row is None:
            # Dev fallback: allow vehicle lookup by id when X-User-Id does not match Driver.id yet.
            row = db.execute(text(vehicle_sql_by_id), {"vehicle_id": vehicle_id}).mappings().first()
        if row is None:
            return None, None

        vehicle = VehiclePayload(
            id=str(row["id"]),
            driverId=str(row["driverId"]) if row.get("driverId") is not None else None,
            plateNumber=row.get("plateNumber"),
            make=row.get("make"),
            model=row.get("model"),
            year=row.get("year"),
            displayName=" ".join(
                p
                for p in [
                    str(row.get("year")) if row.get("year") else "",
                    row.get("make") or "",
                    row.get("model") or "",
                ]
                if p
            ).strip()
            or None,
            type=row.get("type"),
            color=row.get("color"),
            vin=row.get("vin"),
            mileage=row.get("mileage"),
            fuelType=row.get("fuel_type"),
            imageUrl=row.get("image_url"),
            insuranceDocumentUrl=row.get("insurance_document_url"),
            insuranceExpiresAt=row.get("insurance_expires_at"),
            registrationDocumentUrl=row.get("registration_document_url"),
            registrationExpiresAt=row.get("registration_expires_at"),
            createdAt=row.get("createdAt"),
            updatedAt=row.get("updatedAt"),
        )

        health_row = db.execute(text(health_sql), {"vehicle_id": vehicle_id}).mappings().first()
        health = _health_from_db(health_row["health"]) if health_row and "health" in health_row else None
        return vehicle, health


def resolve_context_hybrid(payload: Any, user_id: str) -> tuple[str, str]:
    vehicle_id = _extract_vehicle_id(payload)
    vehicle_db, health_db = fetch_vehicle_context_from_db(vehicle_id, user_id)

    vehicle_context = getattr(payload, "vehicle_context", None)
    vehicle_flutter = (vehicle_context.vehicle if vehicle_context else None) or getattr(payload, "vehicle", None)
    health_flutter = (vehicle_context.vehicle_health if vehicle_context else None) or getattr(
        payload, "vehicle_health", None
    )

    # Hybrid: DB is canonical when available; otherwise use Flutter payload.
    vehicle = vehicle_db or vehicle_flutter
    health = health_db or health_flutter
    manual_context = (getattr(payload, "car_context", "") or "").strip()

    shadow = type("ContextShadow", (), {})()
    shadow.vehicle_context = type("VC", (), {"vehicle": vehicle, "vehicle_health": health})()
    shadow.vehicle = vehicle
    shadow.vehicle_health = health
    shadow.car_context = manual_context
    return normalize_vehicle_context(shadow)


def get_owned_session_or_404(session_id: str, user_id: str) -> ChatSessionModel:
    with SessionLocal() as db:
        session = db.get(ChatSessionModel, session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        return session


app = FastAPI(title="RAG Chat History API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "RAG Chat History API", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=ChatSessionResponse)
def create_session(
    payload: CreateSessionRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> ChatSessionResponse:
    short_ctx, _ = resolve_context_hybrid(payload, user_id)
    now = utcnow()
    session = ChatSessionModel(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=((payload.title or "").strip() or "New chat"),
        car_context=short_ctx,
        created_at=now,
        updated_at=now,
    )
    with SessionLocal() as db:
        db.add(session)
        db.commit()
        db.refresh(session)
    return to_session_response(session)


@app.get("/sessions", response_model=List[ChatSessionResponse])
def list_sessions(user_id: Annotated[str, Depends(get_current_user_id)]) -> List[ChatSessionResponse]:
    with SessionLocal() as db:
        rows = db.scalars(
            select(ChatSessionModel)
            .where(ChatSessionModel.user_id == user_id)
            .order_by(desc(ChatSessionModel.updated_at))
        ).all()
    return [to_session_response(s) for s in rows]


@app.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> ChatSessionResponse:
    session = get_owned_session_or_404(session_id, user_id)
    return to_session_response(session)


@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> Response:
    with SessionLocal() as db:
        session = db.get(ChatSessionModel, session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        db.delete(session)
        db.commit()
    return Response(status_code=204)


@app.get("/sessions/{session_id}/messages", response_model=MessagesPageResponse)
def list_messages(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    limit: int = Query(default=50, ge=1, le=200),
    before: str | None = Query(
        default=None,
        description="Opaque cursor from a previous response's next_before; loads older messages.",
    ),
) -> MessagesPageResponse:
    get_owned_session_or_404(session_id, user_id)

    anchor_ts: datetime | None = None
    anchor_id: str | None = None
    if before:
        anchor_ts, anchor_id = decode_before_cursor(before)

    with SessionLocal() as db:
        q = select(ChatMessageModel).where(ChatMessageModel.session_id == session_id)
        if anchor_ts is not None and anchor_id is not None:
            q = q.where(
                or_(
                    ChatMessageModel.created_at < anchor_ts,
                    and_(ChatMessageModel.created_at == anchor_ts, ChatMessageModel.id < anchor_id),
                )
            )
        q = q.order_by(desc(ChatMessageModel.created_at), desc(ChatMessageModel.id)).limit(limit + 1)
        rows = list(db.scalars(q).all())

    has_more = len(rows) > limit
    page_rows = rows[:limit]

    items_chrono = list(reversed(page_rows))
    items = [to_message_response(m) for m in items_chrono]

    next_before: str | None = None
    if has_more and items_chrono:
        oldest = items_chrono[0]
        next_before = encode_before_cursor(oldest.created_at, oldest.id)

    return MessagesPageResponse(items=items, next_before=next_before, has_more=has_more)


@app.post("/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    payload: ChatMessageRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> Dict[str, Any]:
    text = payload.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    with SessionLocal() as db:
        session = db.get(ChatSessionModel, session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")

        user_message = ChatMessageModel(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=text,
            created_at=utcnow(),
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)

    short_ctx, detailed_ctx = resolve_context_hybrid(payload, user_id)
    active_context = short_ctx or session.car_context
    if payload.title and payload.title.strip():
        session_title = payload.title.strip()
    else:
        session_title = None

    assistant = get_assistant(
        user_id=session.user_id,
        car_context=active_context,
        use_user_manual=payload.use_user_manual,
    )
    answer = assistant.generate_answer(
        text,
        car_context=active_context,
        priority_context=detailed_ctx,
    )

    with SessionLocal() as db:
        assistant_message = ChatMessageModel(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=answer,
            created_at=utcnow(),
        )
        db.add(assistant_message)

        session_to_update = db.get(ChatSessionModel, session_id)
        if session_to_update is not None:
            session_to_update.updated_at = utcnow()
            if active_context:
                session_to_update.car_context = _shorten_context(active_context)
            if session_title:
                session_to_update.title = session_title

        db.commit()
        db.refresh(assistant_message)
        refreshed_session = db.get(ChatSessionModel, session_id)

    return {
        "session": to_session_response(refreshed_session if refreshed_session is not None else session),
        "user_message": to_message_response(user_message),
        "assistant_message": to_message_response(assistant_message),
    }
