from __future__ import annotations

import csv
import os
from datetime import date as _date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

# ---------------------------------------------------------------------------
# Compatibility patch
# ---------------------------------------------------------------------------
try:
    import gradio_client.utils as _gc_utils

    _orig_json = getattr(_gc_utils, "_json_schema_to_python_type", None)

    if callable(_orig_json):
        def _json_schema_to_python_type_safe(schema, defs=None):  # type: ignore
            if isinstance(schema, bool) or schema is None:
                return "Any"
            if not isinstance(schema, dict):
                return "Any"
            return _orig_json(schema, defs)

        _gc_utils._json_schema_to_python_type = _json_schema_to_python_type_safe  # type: ignore
except Exception:
    pass

DEFAULT_API_BASE = (os.getenv("API_BASE_URL") or os.getenv("BACKEND_URL") or "http://localhost:8000").rstrip("/")
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "180"))  # read timeout seconds for API calls

BREED_INFO_DIR = Path(__file__).resolve().parents[1] / "knowledge_base" / "raw" / "breed_info"


def _load_breeds(filename: str, fallback: List[str]) -> List[str]:
    path = BREED_INFO_DIR / filename
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = csv.DictReader(fh)
            breeds = [row.get("breed", "").strip() for row in rows if row.get("breed", "").strip()]
        seen = set()
        ordered = []
        for breed in breeds:
            if breed not in seen:
                ordered.append(breed)
                seen.add(breed)
        return ordered or fallback
    except Exception:
        return fallback


DOG_BREEDS = _load_breeds(
    "dog_breeds.csv",
    [
        "Mixed Breed", "Akita", "Beagle", "Boxer", "Bulldog", "Dachshund",
        "French Bulldog", "German Shepherd", "Golden Retriever", "Labrador Retriever",
        "Poodle", "Pug", "Rottweiler", "Shiba Inu", "Siberian Husky", "Yorkshire Terrier",
    ],
)
CAT_BREEDS = _load_breeds(
    "cat_breeds.csv",
    [
        "Domestic Shorthair", "Domestic Longhair", "Bengal", "British Shorthair",
        "Maine Coon", "Persian", "Ragdoll", "Scottish Fold", "Siamese", "Sphynx",
    ],
)


def _breed_choices_for_species(species: str) -> List[str]:
    return CAT_BREEDS if species == "cat" else DOG_BREEDS


def _default_breed_for_species(species: str) -> str:
    breeds = _breed_choices_for_species(species)
    preferred = "Domestic Shorthair" if species == "cat" else "Mixed Breed"
    if preferred in breeds:
        return preferred
    return breeds[0] if breeds else preferred


def _update_breed_dropdown(species: str, current_breed: str = "", preserve_custom: bool = False):
    breeds = _breed_choices_for_species(species)
    if preserve_custom and current_breed:
        breed_value = current_breed
    else:
        breed_value = current_breed if current_breed in breeds else _default_breed_for_species(species)
    return gr.update(choices=breeds, value=breed_value)

# -----------------------------
# HTTP helpers
# -----------------------------
def _url(base: str, path: str) -> str:
    base = (base or "").rstrip("/")
    path = "/" + path.lstrip("/")
    return f"{base}{path}"

def _req(
    method: str,
    api_base: str,
    path: str,
    token: str = "",
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Session-Token"] = token
    try:
        r = requests.request(
            method=method,
            url=_url(api_base, path),
            json=json_body,
            params=params,
            headers=headers,
            timeout=(5, HTTP_TIMEOUT_S),
        )
        if r.status_code >= 400:
            try:
                detail = r.json()
            except Exception:
                detail = {"detail": r.text}
            return {"ok": False, "status": r.status_code, "error": detail}
        return {"ok": True, "data": r.json()}
    except Exception as e:
        return {"ok": False, "status": 0, "error": {"detail": str(e)}}

def _pretty_err(resp: Dict[str, Any]) -> str:
    if resp.get("status") == 0:
        return f"Network error: {resp.get('error', {}).get('detail', 'unknown')}"
    err = resp.get("error", {})
    if isinstance(err, dict) and "detail" in err:
        return str(err["detail"])
    return str(err)

# -----------------------------
# Auth actions
# -----------------------------
def signup(api_base: str, email: str, password: str) -> Tuple[str, str, str]:
    resp = _req("POST", api_base, "/api/auth/signup", json_body={"email": email, "password": password})
    if not resp["ok"]:
        return "", "", f"❌ Signup failed: {_pretty_err(resp)}"
    token = resp["data"]["token"]
    em = resp["data"]["email"]
    return token, em, "✅ Signed up & logged in successfully."

def login(api_base: str, email: str, password: str) -> Tuple[str, str, str]:
    resp = _req("POST", api_base, "/api/auth/login", json_body={"email": email, "password": password})
    if not resp["ok"]:
        return "", "", f"❌ Login failed: {_pretty_err(resp)}"
    token = resp["data"]["token"]
    em = resp["data"]["email"]
    return token, em, "✅ Logged in successfully."

def logout(api_base: str, token: str) -> Tuple[str, str, str]:
    if token:
        _req("POST", api_base, "/api/auth/logout", token=token)
    return "", "", "✅ Logged out successfully."

# -----------------------------
# Pets
# -----------------------------
def fetch_pets(api_base: str, token: str) -> Tuple[List[dict], List[Tuple[str, int]], str]:
    resp = _req("GET", api_base, "/api/pets", token=token)
    if not resp["ok"]:
        return [], [], f"❌ Failed to load pets: {_pretty_err(resp)}"
    pets = resp["data"]
    choices = [(f"{p['name']} ({p['species']} - {p['breed']})", p["id"]) for p in pets]
    return pets, choices, f"✅ Loaded {len(pets)} pet(s)."

def create_pet(
    api_base: str, token: str, name: str, species: str, breed: str,
    age_years: float, weight_kg: float, is_neutered: bool,
    activity_level: str, health_conditions: str, allergies: str,
) -> str:
    if not name.strip(): return "❌ Pet name is required."
    payload = {
        "name": name.strip(), "species": species, "breed": breed.strip(),
        "age_years": float(age_years or 0), "weight_kg": float(weight_kg or 0),
        "is_neutered": bool(is_neutered), "activity_level": activity_level,
        "health_conditions": health_conditions.strip() or None, "allergies": allergies.strip() or None,
    }
    resp = _req("POST", api_base, "/api/pets", token=token, json_body=payload)
    if not resp["ok"]: return f"❌ Create failed: {_pretty_err(resp)}"
    return f"✅ '{name}' added successfully!"

def load_pet_into_form(pets: List[dict], pet_id: int):
    p = next((x for x in pets if x["id"] == pet_id), None)
    if not p:
        return (
            gr.update(value=""),
            gr.update(value="dog"),
            _update_breed_dropdown("dog"),
            gr.update(value=0.0),
            gr.update(value=0.0),
            gr.update(value=False),
            gr.update(value="moderate"),
            gr.update(value=""),
            gr.update(value=""),
            "",
        )

    species = p.get("species", "dog")
    breed = p.get("breed", "")
    return (
        gr.update(value=p.get("name", "")),
        gr.update(value=species),
        _update_breed_dropdown(species, breed, preserve_custom=True),
        gr.update(value=float(p.get("age_years", 0.0))),
        gr.update(value=float(p.get("weight_kg", 0.0))),
        gr.update(value=bool(p.get("is_neutered", False))),
        gr.update(value=p.get("activity_level", "moderate")),
        gr.update(value=p.get("health_conditions") or ""),
        gr.update(value=p.get("allergies") or ""),
        f"✅ Loaded details for {p.get('name')}.",
    )

def clear_pet_form():
    return (
        gr.update(value=""),
        gr.update(value="dog"),
        _update_breed_dropdown("dog"),
        gr.update(value=1.0),
        gr.update(value=5.0),
        gr.update(value=False),
        gr.update(value="moderate"),
        gr.update(value=""),
        gr.update(value=""),
        "✨ Form cleared. Enter details to add a new pet."
    )

def update_pet(
    api_base: str, token: str, pet_id: int, name: str, species: str, breed: str,
    age_years: float, weight_kg: float, is_neutered: bool,
    activity_level: str, health_conditions: str, allergies: str,
) -> str:
    if not pet_id: return "❌ Please select a pet to update."
    payload = {
        "name": name.strip(), "species": species, "breed": breed.strip(),
        "age_years": float(age_years or 0), "weight_kg": float(weight_kg or 0),
        "is_neutered": bool(is_neutered), "activity_level": activity_level,
        "health_conditions": health_conditions.strip() or None, "allergies": allergies.strip() or None,
    }
    resp = _req("PUT", api_base, f"/api/pets/{pet_id}", token=token, json_body=payload)
    if not resp["ok"]: return f"❌ Update failed: {_pretty_err(resp)}"
    return f"✅ '{name}' updated successfully."

def delete_pet(api_base: str, token: str, pet_id: int) -> str:
    if not pet_id: return "❌ Please select a pet to delete."
    resp = _req("DELETE", api_base, f"/api/pets/{pet_id}", token=token)
    if not resp["ok"]: return f"❌ Delete failed: {_pretty_err(resp)}"
    return "✅ Pet profile deleted."

# -----------------------------
# Activity
# -----------------------------
def add_activity(
    api_base: str,
    token: str,
    pet_id: int,
    activity_date: str,
    activity_type: str,
    duration_minutes: int,
    intensity: int,
) -> str:
    if not pet_id:
        return "❌ Please select a pet."

    # Convert human-friendly activity -> (steps, active_minutes)
    at = (activity_type or "").strip().lower()
    dur = max(0, int(duration_minutes or 0))
    inten = max(1, min(5, int(intensity or 3)))

    steps_per_min = {
        "none": 0,
        "walk_slow": 80,
        "walk_normal": 100,
        "walk_fast": 120,
        "play_fetch": 110,
        "training": 60,
        "run": 140,
    }.get(at, 100)

    intensity_mult = {1: 0.7, 2: 0.85, 3: 1.0, 4: 1.15, 5: 1.3}[inten]
    steps = int(round(steps_per_min * dur * intensity_mult))
    active_minutes = 0 if at == "none" else dur

    payload = {
        "pet_id": int(pet_id),
        "date": activity_date.strip(),
        "steps": int(steps),
        "active_minutes": int(active_minutes),
    }
    resp = _req("POST", api_base, "/api/activity/logs", token=token, json_body=payload)
    if not resp["ok"]:
        return f"❌ Save failed: {_pretty_err(resp)}"
    return f"✅ Activity saved (estimated {steps} steps, {active_minutes} active min)."

def list_activity(api_base: str, token: str, pet_id: int) -> Tuple[List[List[Any]], str]:
    if not pet_id: return [], ""
    resp = _req("GET", api_base, "/api/activity/logs", token=token, params={"pet_id": pet_id, "limit": 14})
    if not resp["ok"]: return [], f"❌ Failed to load history: {_pretty_err(resp)}"
    rows = resp["data"]
    table = [[r["activity_date"], r["steps"], r["active_minutes"], r["calories_burned"]] for r in rows]
    return table, "✅ Activity history refreshed."

def adjust_today(api_base: str, token: str, pet_id: int, activity_date: str) -> str:
    if not pet_id: return "❌ Please select a pet first."
    resp = _req("GET", api_base, f"/api/activity/adjust/{pet_id}", token=token, params={"activity_date": activity_date})
    if not resp["ok"]: return f"❌ Plan generation failed: {_pretty_err(resp)}"
    d = resp["data"]
    mp = d.get("meal_plan", {}) or {}
    return (
        f"### 🎯 Daily Nutrition Targets\n\n"
        f"- **Base Calories:** {d.get('base_calories'):.0f} kcal\n"
        f"- **Adjusted Calories:** {d.get('adjusted_calories'):.0f} kcal *(Adjustment: {d.get('adjustment_percent')}% based on activity)*\n"
        f"- **Daily Food Portion:** {mp.get('total_grams_per_day')} g\n"
        f"- **Meal Breakdown:** {mp.get('meal_frequency')} meals × {mp.get('grams_per_meal')} g per meal\n\n"
        f"---\n"
        f"**💡 Advisor Recommendation:**\n{d.get('recommendation')}"
    )

# -----------------------------
# Chat
# -----------------------------
def chat_send(api_base: str, token: str, pet_id: int, message: str, history: List[Dict[str, Any]]):
    """Send a chat message and keep the latest Q/A plus chat_log_id for thumbs feedback."""
    message = (message or "").strip()
    if not message:
        return history, "", "", "", pet_id, None

    if not pet_id:
        history = list(history or [])
        history.append({"role": "assistant", "content": "❌ Please select a pet profile first."})
        return history, "", message, "", pet_id, None

    history = list(history or [])
    history.append({"role": "user", "content": message})

    payload = {"pet_id": int(pet_id), "question": message}
    resp = _req("POST", api_base, "/api/nutrition/chat", token=token, json_body=payload)
    if not resp["ok"]:
        err = f"❌ {_pretty_err(resp)}"
        history.append({"role": "assistant", "content": err})
        return history, "", message, err, pet_id, None

    answer = resp["data"].get("answer", "")
    chat_log_id = resp["data"].get("chat_log_id")
    history.append({"role": "assistant", "content": answer})
    return history, "", message, answer, pet_id, chat_log_id


# -----------------------------
# Chat thumbs (👍/👎) for the latest answer
# -----------------------------
def send_chat_vote(
    api_base: str,
    token: str,
    chat_log_id: Optional[int],
    rating: int,
    reason: str = "",
    comment: str = "",
) -> str:
    if not chat_log_id:
        return "❌ No recent answer to rate yet."

    payload = {
        "chat_log_id": int(chat_log_id),
        "rating": int(rating),
        "reason": (reason or "").strip() or None,
        "comment": (comment or "").strip() or None,
    }
    resp = _req("POST", api_base, "/api/chat/feedback", token=token, json_body=payload)
    if not resp["ok"]:
        return f"❌ Vote failed: {_pretty_err(resp)}"
    return "✅ Saved. Thank you!"


# -----------------------------
# Feedback
# -----------------------------
def submit_feedback(
    api_base: str,
    token: str,
    pet_id: Optional[int],
    page: str,
    category: str,
    rating: Optional[int],
    message: str,
) -> Tuple[str, str]:
    message = (message or "").strip()
    if not message:
        return "❌ Please enter feedback before submitting.", ""

    payload = {
        "pet_id": int(pet_id) if pet_id else None,
        "page": (page or "general").strip().lower(),
        "category": (category or "other").strip().lower(),
        "rating": int(rating) if rating else None,
        "message": message,
    }
    resp = _req("POST", api_base, "/api/feedback", token=token, json_body=payload)
    if not resp["ok"]:
        return f"❌ Feedback submit failed: {_pretty_err(resp)}", message

    return "✅ Thanks. Your feedback has been saved.", ""


def list_feedback(api_base: str, token: str) -> Tuple[List[List[Any]], str]:
    resp = _req("GET", api_base, "/api/feedback", token=token, params={"limit": 20})
    if not resp["ok"]:
        return [], f"<div class='status-bar'>❌ Failed to load feedback history: {_pretty_err(resp)}</div>"

    rows = resp["data"] or []
    table = [
        [
            (r.get("created_at") or "")[:19].replace("T", " "),
            r.get("page") or "general",
            r.get("category") or "other",
            r.get("rating"),
            r.get("pet_id") or "-",
            r.get("message") or "",
        ]
        for r in rows
    ]
    return table, "<div class='status-bar'>✅ Feedback history refreshed.</div>"


def get_chat_summary(api_base: str, token: str) -> str:
    resp = _req("GET", api_base, "/api/chat/summary", token=token)
    if not resp["ok"]:
        return f"""### Chat monitoring

❌ Failed to load summary: {_pretty_err(resp)}
"""

    data = resp["data"] or {}
    total = int(data.get("total_chats", 0) or 0)
    avg_latency = float(data.get("avg_latency_ms", 0.0) or 0.0)
    negative_rate = float(data.get("negative_feedback_rate", 0.0) or 0.0) * 100
    error_rate = float(data.get("error_rate", 0.0) or 0.0) * 100

    return f"""### Chat monitoring

- **Total chats:** {total}
- **Avg latency:** {avg_latency:.2f} ms
- **Negative feedback rate:** {negative_rate:.1f}%
- **Error rate:** {error_rate:.1f}%
"""


def list_chat_logs(api_base: str, token: str) -> Tuple[List[List[Any]], str]:
    resp = _req("GET", api_base, "/api/chat/logs", token=token, params={"limit": 20})
    if not resp["ok"]:
        return [], f"<div class='status-bar'>❌ Failed to load chat logs: {_pretty_err(resp)}</div>"

    rows = resp["data"] or []
    table = [
        [
            (r.get("created_at") or "")[:19].replace("T", " "),
            r.get("pet_id") or "-",
            r.get("route_type") or "-",
            r.get("status") or "-",
            float(r.get("latency_ms") or 0),
            int(r.get("retrieved_docs_count") or 0),
            r.get("question") or "",
        ]
        for r in rows
    ]
    return table, "<div class='status-bar'>✅ Chat monitoring refreshed.</div>"


# -----------------------------
# UI Layout & Wiring
# -----------------------------

# SENIOR NOTE: Switched to a Teal/Emerald theme to promote a "health & wellness" feel.
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg, # Makes everything slightly more rounded and friendly
)

CSS = """
/* ── Base ── */
body {
    background-color: #f1f5f9;
}

/* ── Auth card ── */
.auth-box {
    max-width: 480px;
    margin: 8vh auto !important;
    padding: 40px;
    border-radius: 24px;
    background: white;
    box-shadow: 0 20px 40px -15px rgba(15, 118, 110, 0.15), 0 4px 10px rgba(0, 0, 0, 0.04);
}
.auth-header { text-align: center; margin-bottom: 24px; color: #0f766e; }

/* ── App container ── */
.app-container {
    max-width: 1100px !important;
    width: 95vw;
    margin: 4vh auto !important;
    background: transparent;
}

/* ── Cards / Grouping ── */
.modern-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

.highlight-card {
    background: #f0fdfa; /* Light teal background */
    border: 1px solid #ccfbf1;
    border-radius: 16px;
    padding: 24px;
}

/* ── Chat pet context selector ── */
.pet-context-card {
    background: #fff7ed; /* warm highlight */
    border: 1px solid #fed7aa;
    border-radius: 16px;
    padding: 16px;
    margin: 12px 0 16px 0;
}
.pet-context-note {
    color: #9a3412;
    font-size: 0.9rem;
    margin-top: 8px;
}

/* ── Header ── */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    padding: 16px 24px;
    border-radius: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    margin-bottom: 24px;
    border: 1px solid #e2e8f0;
}
.app-title { margin: 0 !important; color: #0f766e !important; }

/* ── Delete confirm row ── */
.delete-confirm {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 16px;
    margin-top: 16px;
}

/* ── Status bar ── */
.status-bar {
    text-align: right;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 8px;
}

/* ── Footer admin monitor ── */
#footer-monitor-shell {
    display: inline-flex;
    align-items: center;
    margin-left: 0;
}
#footer-monitor-shell details {
    position: relative;
    border: 0;
    background: transparent;
    box-shadow: none;
}
#footer-monitor-shell summary {
    list-style: none;
    cursor: pointer;
    font-size: 0.9rem;
    color: #64748b;
    letter-spacing: 0;
    text-transform: none;
    padding: 0;
}
#footer-monitor-shell summary::-webkit-details-marker {
    display: none;
}
#footer-monitor-shell details:not([open]) > div {
    display: none;
}
#footer-monitor-shell details[open] > div {
    display: block;
    position: absolute;
    right: 0;
    bottom: calc(100% + 10px);
    width: min(560px, calc(100vw - 28px));
    max-height: min(70vh, 520px);
    overflow: auto;
    border: 1px solid #dbe4ee;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.98);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
    padding: 10px 12px;
    z-index: 40;
}
#footer-monitor-shell .hide-label label,
#footer-monitor-shell .hide-label > .label-wrap {
    display: none !important;
}
#footer-monitor-shell .gr-dataframe {
    max-height: 220px;
}
.footer-admin-sep {
    color: #94a3b8;
    margin: 0 8px;
}
"""

# NOTE:
# Put theme/css on Blocks (stable across Gradio versions).
# Passing theme/css/footer args to `launch()` can cause blank UI in some versions.
with gr.Blocks(title="Pet Nutrition Planner", theme=theme, css=CSS) as demo:
    api_base = gr.State(DEFAULT_API_BASE)
    token_state = gr.State("")
    email_state = gr.State("")
    pets_state = gr.State([])
    last_q_state = gr.State("")
    last_a_state = gr.State("")
    last_pet_state = gr.State(None)
    last_chat_log_id_state = gr.State(None)

    # -----------------------------
    # 1. AUTHENTICATION VIEW
    # -----------------------------
    with gr.Column(elem_classes=["auth-box"], visible=True) as auth_view:
        gr.Markdown("## 🐾 Pet Nutrition Planner", elem_classes=["auth-header"])
        gr.Markdown("<p style='text-align: center; color: #64748b; margin-bottom: 20px;'>Welcome! Please log in to manage your pet's health and activity.</p>")
        
        with gr.Tabs():
            with gr.Tab("Login"):
                li_email = gr.Textbox(label="Email Address", placeholder="you@example.com")
                li_pass = gr.Textbox(label="Password", type="password")
                li_btn = gr.Button("Login", variant="primary", size="lg")
                li_msg = gr.Markdown()
            
            with gr.Tab("Sign Up"):
                su_email = gr.Textbox(label="Email Address", placeholder="you@example.com")
                su_pass = gr.Textbox(label="Password", type="password", info="Minimum 6 characters")
                su_btn = gr.Button("Create Account", variant="primary", size="lg")
                su_msg = gr.Markdown()
        
        with gr.Accordion("⚙️ Advanced Setup (API Base URL)", open=False):
            api_in = gr.Textbox(label="Backend API URL", value=DEFAULT_API_BASE, show_label=False)

    # -----------------------------
    # 2. MAIN DASHBOARD VIEW
    # -----------------------------
    # SENIOR NOTE: Removed the fake sidebar logic and instead wrapped everything in standard gr.Tabs.
    with gr.Column(visible=False, elem_classes=["app-container"]) as app_view:
        
        # Sleek App Header
        with gr.Row(elem_classes=["app-header"]):
            gr.Markdown("## 🐾 Pet Health Dashboard", elem_classes=["app-title"])
            with gr.Row():
                user_badge = gr.Markdown("👤 Not logged in")
                btn_logout = gr.Button("🚪 Logout", size="sm", variant="secondary")

        with gr.Tabs():
            
            # --- Tab 1: AI Advisor (Chat) ---
            with gr.Tab("💬 AI Advisor"):
                with gr.Column(elem_classes=["modern-card"]):
                    gr.Markdown("### Ask personalized questions about pet food, portion sizes, and general wellness.")

                    # Make the pet context selector visually distinct and place the warning inside it.
                    with gr.Group(elem_classes=["pet-context-card"]):
                        gr.Markdown("#### 🐾 Select a pet profile (required)")
                        with gr.Row():
                            chat_pet = gr.Dropdown(
                                label="Pet",
                                choices=[],
                                value=None,
                                interactive=True,
                                scale=4,
                            )
                            btn_clear_chat = gr.Button("🗑️ Clear Chat History", scale=1)
                        gr.Markdown(
                            "⚠️ Select a pet from the dropdown first so the advisor can use the correct profile. "
                            "If you don’t have one yet, create a pet profile in the ‘🐶 Pet Profiles’ tab.",
                            elem_classes=["pet-context-note"],
                        )
                        
                    chatbot = gr.Chatbot(
                        height=450,
                        show_label=False,
                        placeholder=(
                            "👋 **Welcome to your AI Pet Advisor!**\n\n"
                            "Ask me anything about your pet's nutrition, portion sizes, food safety, or general wellness.\n\n"
                            "*Examples:*\n"
                            "- 'Can my dog eat strawberries?'\n"
                            "- 'What portion size is right for a 5kg cat?'\n"
                            "- 'My pet has diabetes — what foods should I avoid?'"
                        ),
                    )
                    with gr.Row():
                        msg = gr.Textbox(show_label=False, placeholder="Type your question here...", scale=5)
                        send = gr.Button("Send", variant="primary", scale=1)

                    # 👍/👎 for the latest answer
                    with gr.Row():
                        btn_up = gr.Button("👍 Helpful", size="sm")
                        btn_down = gr.Button("👎 Not helpful", size="sm")
                        vote_status = gr.Markdown()

                    corrected_box = gr.Textbox(
                        label="If not helpful, what would be a better answer? (optional)",
                        placeholder="Type a corrected answer or extra context...",
                        visible=False,
                        lines=3,
                    )
                    btn_submit_correction = gr.Button("Submit correction", visible=False)

            # --- Tab 2: Pet Profiles ---
            with gr.Tab("🐶 Pet Profiles"):
                with gr.Row():
                    # Left side: Selection & Creation
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["modern-card"]):
                            gr.Markdown("### 🐾 Your Pets")
                            pet_select = gr.Dropdown(label="Select a pet to edit...", choices=[])
                            gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 8px;'>— OR —</div>")
                            btn_clear = gr.Button("➕ Create New Profile", variant="secondary")
                            
                    # Right side: The Form
                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["modern-card"]):
                            gr.Markdown("### 📋 Pet Details")
                            with gr.Row():
                                p_name = gr.Textbox(label="Name", placeholder="e.g. Bella")
                                p_species = gr.Dropdown(label="Species", choices=[("Dog", "dog"), ("Cat", "cat")], value="dog")
                            
                            p_breed = gr.Dropdown(label="Breed", choices=DOG_BREEDS, value=_default_breed_for_species("dog"), allow_custom_value=True)
                            
                            with gr.Row():
                                p_age = gr.Number(label="Age (years)", value=1.0, precision=1)
                                p_weight = gr.Number(label="Weight (kg)", value=5.0, precision=1)
                            
                            with gr.Row():
                                p_activity = gr.Dropdown(label="Activity Level", choices=[("Sedentary", "sedentary"), ("Moderate", "moderate"), ("Active", "active"), ("Very Active", "very_active")], value="moderate")
                                p_neutered = gr.Checkbox(label="Is Neutered/Spayed?", value=False)
                            
                            with gr.Accordion("🩺 Health Conditions & Allergies (Optional)", open=False):
                                p_health = gr.Textbox(label="Health Conditions", placeholder="e.g. Diabetes, Arthritis", lines=2)
                                p_allergy = gr.Textbox(label="Allergies", placeholder="e.g. Chicken, Grain", lines=2)

                            pet_msg = gr.Markdown()
                            
                            with gr.Row():
                                btn_create = gr.Button("➕ Save New", variant="primary")
                                btn_update = gr.Button("💾 Update", variant="secondary")
                                btn_delete = gr.Button("🗑️ Delete", variant="stop")
                            
                            # Delete confirmation
                            with gr.Row(visible=False, elem_classes=["delete-confirm"]) as delete_confirm_row:
                                gr.Markdown("⚠️ **Are you sure?** This will permanently remove the pet profile.")
                                btn_confirm_delete = gr.Button("Yes, Delete", variant="stop")
                                btn_cancel_delete = gr.Button("Cancel")

            # --- Tab 3: Daily Activity & Nutrition ---
            with gr.Tab("🏃 Daily Activity & Plan"):
                with gr.Row():
                    # Left Column: Log Activity
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["modern-card"]):
                            gr.Markdown("### 📝 Log Activity")
                            act_pet = gr.Dropdown(label="Select Pet", choices=[])
                            act_date = gr.Textbox(label="Date (YYYY-MM-DD)", value=_date.today().isoformat())

                            act_type = gr.Dropdown(
                                label="Activity Type",
                                choices=[
                                    ("Walk (slow)", "walk_slow"),
                                    ("Walk (normal)", "walk_normal"),
                                    ("Walk (fast)", "walk_fast"),
                                    ("Play fetch", "play_fetch"),
                                    ("Training", "training"),
                                    ("Run", "run"),
                                    ("No activity", "none"),
                                ],
                                value="walk_normal",
                            )
                            act_duration = gr.Number(label="Duration (minutes)", value=30, precision=0)
                            act_intensity = gr.Slider(label="Intensity (1–5)", minimum=1, maximum=5, step=1, value=3)

                            btn_save_act = gr.Button("Save Activity", variant="secondary")
                            act_msg = gr.Markdown()
                            
                        # Removed the plan generation to the right column to make it a primary action there

                    # Right Column: The Plan & History
                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["highlight-card"]):
                            gr.Markdown("### 🎯 Today's Nutrition Plan")
                            gr.Markdown("Calculate portions based on the latest logged activity.")
                            btn_adjust = gr.Button("✨ Generate Nutrition Plan", variant="primary", size="lg")
                            plan_md = gr.Markdown("*(Select a pet and click generate to view recommendations)*")
                        
                        with gr.Group(elem_classes=["modern-card"]):
                            gr.Markdown("### 🗓️ Recent History")
                            table = gr.Dataframe(
                                headers=["Date", "Steps", "Active Min", "Calories Burned"],
                                datatype=["str", "number", "number", "number"],
                                interactive=False,
                            )

            # --- Tab 4: Feedback ---
            with gr.Tab("📝 Feedback"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["modern-card"]):
                            gr.Markdown("### Share feedback")
                            fb_pet = gr.Dropdown(label="Pet (optional)", choices=[], value=None)
                            fb_page = gr.Dropdown(
                                label="Related page",
                                choices=[
                                    ("General", "general"),
                                    ("AI Advisor", "advisor"),
                                    ("Pet Profiles", "pets"),
                                    ("Daily Activity", "activity"),
                                    ("Feedback Page", "feedback"),
                                ],
                                value="general",
                            )
                            fb_category = gr.Dropdown(
                                label="Type",
                                choices=[
                                    ("Bug", "bug"),
                                    ("Idea", "idea"),
                                    ("UI / UX", "ui"),
                                    ("Answer Accuracy", "accuracy"),
                                    ("Performance", "performance"),
                                    ("Other", "other"),
                                ],
                                value="other",
                            )
                            fb_rating = gr.Slider(label="Overall rating (1-5)", minimum=1, maximum=5, step=1, value=5)
                            fb_message = gr.Textbox(
                                label="Message",
                                lines=8,
                                placeholder="Tell me what is working, what is confusing, or what answer/page should be improved.",
                            )
                            with gr.Row():
                                btn_submit_feedback = gr.Button("Submit Feedback", variant="primary")
                                btn_refresh_feedback = gr.Button("Refresh History", variant="secondary")
                            fb_msg = gr.Markdown()

                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["modern-card"]):
                            gr.Markdown("### Recent feedback")
                            fb_table = gr.Dataframe(
                                headers=["Created At", "Page", "Category", "Rating", "Pet ID", "Message"],
                                datatype=["str", "str", "str", "number", "str", "str"],
                                interactive=False,
                            )

        # Bottom System Status
        with gr.Row():
            btn_refresh = gr.Button("🔄 Sync Data", size="sm", scale=1)
            status_bar = gr.HTML("<div class='status-bar'>System ready.</div>")

        with gr.Accordion("Admin", open=False, elem_id="footer-monitor-shell"):
            with gr.Column():
                chat_summary_md = gr.Markdown("No chat data yet.", elem_classes=["hide-label"])
                chat_table = gr.Dataframe(
                    headers=["Created At", "Pet ID", "Route", "Status", "Latency (ms)", "Docs", "Question"],
                    datatype=["str", "str", "str", "str", "number", "number", "str"],
                    interactive=False,
                    elem_classes=["hide-label"],
                )

        gr.HTML("""
<script>
(function () {
  function moveAdminToFooter() {
    const shell = document.getElementById('footer-monitor-shell');
    const footer = document.querySelector('footer');
    if (!shell || !footer) return;
    if (shell.dataset.footerMounted === '1') return;

    const target = footer.querySelector('.built-with') || footer.firstElementChild || footer;
    const sep = document.createElement('span');
    sep.className = 'footer-admin-sep';
    sep.textContent = '·';

    shell.dataset.footerMounted = '1';
    target.appendChild(sep);
    target.appendChild(shell);
  }

  window.addEventListener('load', moveAdminToFooter);
  document.addEventListener('DOMContentLoaded', moveAdminToFooter);
  const observer = new MutationObserver(moveAdminToFooter);
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
""")

    # -----------------------------
    # UI EVENT WIRING
    # -----------------------------
    
    # Utilities
    def _set_api_base(new_base: str):
        return (new_base or DEFAULT_API_BASE).rstrip("/")

    def _after_auth(token: str, email: str):
        logged_in = bool(token)
        badge = f"👤 **`{email}`**" if logged_in else "👤 Not logged in"
        return (
            gr.update(visible=not logged_in),  # auth_view
            gr.update(visible=logged_in),      # app_view
            badge,                             # user_badge
        )

    def _refresh(api_base_val: str, token: str):
        pets, choices, msg = fetch_pets(api_base_val, token)
        return (
            pets,
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            f"<div class='status-bar'>{msg}</div>",
        )

    def _refresh_monitoring(api_base_val: str, token: str):
        summary = get_chat_summary(api_base_val, token)
        logs, status = list_chat_logs(api_base_val, token)
        return summary, logs, status

    # Authentication wiring
    api_in.change(_set_api_base, inputs=api_in, outputs=api_base)

    su_btn.click(signup, inputs=[api_base, su_email, su_pass], outputs=[token_state, email_state, su_msg])\
          .then(_after_auth, inputs=[token_state, email_state], outputs=[auth_view, app_view, user_badge])\
          .then(_refresh, inputs=[api_base, token_state], outputs=[pets_state, chat_pet, pet_select, act_pet, fb_pet, status_bar])\
          .then(list_feedback, inputs=[api_base, token_state], outputs=[fb_table, status_bar])\
          .then(_refresh_monitoring, inputs=[api_base, token_state], outputs=[chat_summary_md, chat_table, status_bar])
          
    li_btn.click(login, inputs=[api_base, li_email, li_pass], outputs=[token_state, email_state, li_msg])\
          .then(_after_auth, inputs=[token_state, email_state], outputs=[auth_view, app_view, user_badge])\
          .then(_refresh, inputs=[api_base, token_state], outputs=[pets_state, chat_pet, pet_select, act_pet, fb_pet, status_bar])\
          .then(list_feedback, inputs=[api_base, token_state], outputs=[fb_table, status_bar])\
          .then(_refresh_monitoring, inputs=[api_base, token_state], outputs=[chat_summary_md, chat_table, status_bar])

    btn_logout.click(logout, inputs=[api_base, token_state], outputs=[token_state, email_state, status_bar])\
              .then(_after_auth, inputs=[token_state, email_state], outputs=[auth_view, app_view, user_badge])\
              .then(lambda: ([], gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), "<div class='status-bar'>Logged out</div>"),
                    outputs=[pets_state, chat_pet, pet_select, act_pet, fb_pet, status_bar])\
              .then(lambda: [], outputs=[fb_table])\
              .then(lambda: ("No chat data yet.", []), outputs=[chat_summary_md, chat_table])

    # Navbar/Refresh wiring (No longer need manual nav functions since we use gr.Tabs!)
    btn_refresh.click(_refresh, inputs=[api_base, token_state], outputs=[pets_state, chat_pet, pet_select, act_pet, fb_pet, status_bar])\
               .then(list_feedback, inputs=[api_base, token_state], outputs=[fb_table, status_bar])\
               .then(_refresh_monitoring, inputs=[api_base, token_state], outputs=[chat_summary_md, chat_table, status_bar])

    # Pet CRUD Wiring
    pet_select.change(
        load_pet_into_form,
        inputs=[pets_state, pet_select],
        outputs=[p_name, p_species, p_breed, p_age, p_weight, p_neutered, p_activity, p_health, p_allergy, pet_msg],
    )
    p_species.change(_update_breed_dropdown, inputs=[p_species, p_breed], outputs=[p_breed])
    btn_clear.click(
        clear_pet_form,
        outputs=[p_name, p_species, p_breed, p_age, p_weight, p_neutered, p_activity, p_health, p_allergy, pet_msg]
    ).then(
        lambda: gr.update(value=None), outputs=[pet_select] # Clear the dropdown selection visually
    )
    btn_create.click(
        create_pet, inputs=[api_base, token_state, p_name, p_species, p_breed, p_age, p_weight, p_neutered, p_activity, p_health, p_allergy], outputs=[pet_msg]
    ).then(_refresh, inputs=[api_base, token_state], outputs=[pets_state, chat_pet, pet_select, act_pet, fb_pet, status_bar])
    
    btn_update.click(
        update_pet, inputs=[api_base, token_state, pet_select, p_name, p_species, p_breed, p_age, p_weight, p_neutered, p_activity, p_health, p_allergy], outputs=[pet_msg]
    ).then(_refresh, inputs=[api_base, token_state], outputs=[pets_state, chat_pet, pet_select, act_pet, fb_pet, status_bar])
    
    btn_delete.click(lambda: gr.update(visible=True), outputs=[delete_confirm_row])
    btn_cancel_delete.click(lambda: gr.update(visible=False), outputs=[delete_confirm_row])
    
    btn_confirm_delete.click(
        delete_pet, inputs=[api_base, token_state, pet_select], outputs=[pet_msg]
    ).then(
        lambda: gr.update(visible=False), outputs=[delete_confirm_row]
    ).then(_refresh, inputs=[api_base, token_state], outputs=[pets_state, chat_pet, pet_select, act_pet, fb_pet, status_bar])

    # Activity Wiring
    btn_save_act.click(
        add_activity, inputs=[api_base, token_state, act_pet, act_date, act_type, act_duration, act_intensity], outputs=[act_msg]
    ).then(list_activity, inputs=[api_base, token_state, act_pet], outputs=[table, status_bar])
    
    btn_adjust.click(adjust_today, inputs=[api_base, token_state, act_pet, act_date], outputs=[plan_md])
    act_pet.change(list_activity, inputs=[api_base, token_state, act_pet], outputs=[table, status_bar])

    # Chat Wiring & History Management
    send.click(chat_send, inputs=[api_base, token_state, chat_pet, msg, chatbot], outputs=[chatbot, msg, last_q_state, last_a_state, last_pet_state, last_chat_log_id_state])
    msg.submit(chat_send, inputs=[api_base, token_state, chat_pet, msg, chatbot], outputs=[chatbot, msg, last_q_state, last_a_state, last_pet_state, last_chat_log_id_state])
    
    btn_up.click(
        lambda api_base, token, chat_log_id: send_chat_vote(api_base, token, chat_log_id, 1, "", ""),
        inputs=[api_base, token_state, last_chat_log_id_state],
        outputs=[vote_status],
    ).then(_refresh_monitoring, inputs=[api_base, token_state], outputs=[chat_summary_md, chat_table, status_bar])
    btn_down.click(
        lambda: (gr.update(visible=True), gr.update(visible=True), "👎 Please add a correction (optional) then submit."),
        outputs=[corrected_box, btn_submit_correction, vote_status],
    )
    btn_submit_correction.click(
        lambda api_base, token, chat_log_id, corr: (send_chat_vote(api_base, token, chat_log_id, -1, "incorrect", corr), gr.update(value="", visible=False), gr.update(visible=False)),
        inputs=[api_base, token_state, last_chat_log_id_state, corrected_box],
        outputs=[vote_status, corrected_box, btn_submit_correction],
    ).then(_refresh_monitoring, inputs=[api_base, token_state], outputs=[chat_summary_md, chat_table, status_bar])

    chat_pet.change(lambda: [], outputs=[chatbot])
    btn_clear_chat.click(lambda: [], outputs=[chatbot])

    # Feedback Wiring
    btn_submit_feedback.click(
        submit_feedback,
        inputs=[api_base, token_state, fb_pet, fb_page, fb_category, fb_rating, fb_message],
        outputs=[fb_msg, fb_message],
    ).then(
        list_feedback,
        inputs=[api_base, token_state],
        outputs=[fb_table, status_bar],
    )

    btn_refresh_feedback.click(
        list_feedback,
        inputs=[api_base, token_state],
        outputs=[fb_table, status_bar],
    ).then(
        _refresh_monitoring,
        inputs=[api_base, token_state],
        outputs=[chat_summary_md, chat_table, status_bar],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    share = os.getenv("GRADIO_SHARE", "0").strip() in ("1", "true", "True", "yes", "YES")
    demo.launch(server_name=server_name, server_port=port, share=share)
