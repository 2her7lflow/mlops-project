from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SEC", "20"))


def _headers(token: str) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if token:
        h["X-Session-Token"] = token
    return h


def _req(method: str, path: str, token: str = "", **kwargs):
    url = f"{BACKEND_URL}{path}"
    try:
        r = requests.request(method, url, headers=_headers(token), timeout=TIMEOUT, **kwargs)
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json()
        else:
            data = {"raw": r.text}
        if r.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            raise RuntimeError(f"{r.status_code} {r.reason}: {detail or data}")
        return data
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")


# -------------------------
# Auth
# -------------------------
def signup(email: str, password: str) -> Tuple[str, str]:
    data = _req("POST", "/api/auth/signup", json={"email": email, "password": password})
    token = data.get("token", "")
    return token, f"✅ Signed up as {data.get('email', email)}"


def login(email: str, password: str) -> Tuple[str, str]:
    data = _req("POST", "/api/auth/login", json={"email": email, "password": password})
    token = data.get("token", "")
    return token, f"✅ Logged in as {data.get('email', email)}"


def logout(token: str) -> Tuple[str, str]:
    if not token:
        return "", "ℹ️ Already logged out."
    _req("POST", "/api/auth/logout", token=token)
    return "", "✅ Logged out."


def whoami(token: str) -> str:
    if not token:
        return "Not logged in."
    data = _req("GET", "/api/auth/me", token=token)
    return f"Logged in as: {data.get('email', '-')}"


# -------------------------
# Pets
# -------------------------
def list_pets(token: str) -> Tuple[List[Tuple[str, int]], str]:
    if not token:
        return [], "⚠️ Please log in first."
    pets = _req("GET", "/api/pets", token=token)
    choices: List[Tuple[str, int]] = []
    for p in pets:
        label = f"#{p['id']} • {p.get('name','(no name)')} ({p.get('species','?')})"
        choices.append((label, int(p["id"])))
    msg = f"✅ Loaded {len(choices)} pet(s)."
    return choices, msg


def create_pet(
    token: str,
    name: str,
    species: str,
    breed: str,
    age_years: float,
    weight_kg: float,
    is_neutered: bool,
    activity_level: str,
    health_conditions: str,
    allergies: str,
) -> str:
    if not token:
        return "⚠️ Please log in first."
    payload = {
        "name": name.strip(),
        "species": species,
        "breed": breed.strip(),
        "age_years": float(age_years),
        "weight_kg": float(weight_kg),
        "is_neutered": bool(is_neutered),
        "activity_level": activity_level,
        "health_conditions": health_conditions.strip() or None,
        "allergies": allergies.strip() or None,
    }
    pet = _req("POST", "/api/pets", token=token, json=payload)
    return f"✅ Created pet #{pet.get('id')} ({pet.get('name')})"


# -------------------------
# Activity
# -------------------------
def add_activity(token: str, pet_id: int, day: str, steps: int, active_minutes: int) -> str:
    if not token:
        return "⚠️ Please log in first."
    if not pet_id:
        return "⚠️ Select a pet."
    payload = {
        "pet_id": int(pet_id),
        "date": day,
        "steps": int(steps),
        "active_minutes": int(active_minutes),
    }
    log = _req("POST", "/api/activity/logs", token=token, json=payload)
    return f"✅ Saved activity: {log.get('activity_date')} • steps={log.get('steps')} • active={log.get('active_minutes')} min"


def list_activity(token: str, pet_id: int, limit: int):
    if not token:
        return gr.update(value=[]), "⚠️ Please log in first."
    if not pet_id:
        return gr.update(value=[]), "⚠️ Select a pet."
    logs = _req("GET", f"/api/activity/logs?pet_id={int(pet_id)}&limit={int(limit)}", token=token)
    rows = [
        [x.get("activity_date"), x.get("steps"), x.get("active_minutes"), x.get("calories_burned")]
        for x in logs
    ]
    return gr.update(value=rows), f"✅ Loaded {len(rows)} log(s)."


# -------------------------
# Nutrition chat (RAG)
# -------------------------
def chat_submit(token: str, pet_id: Optional[int], question: str, history: List[List[str]]):
    if not token:
        history = history + [["", "⚠️ Please log in first."]]
        return history, history
    if not question.strip():
        return history, history

    payload: Dict[str, Any] = {"question": question.strip()}
    if pet_id:
        payload["pet_id"] = int(pet_id)

    try:
        data = _req("POST", "/api/nutrition/chat", token=token, json=payload)
        answer = data.get("answer", "")
        sources = data.get("sources") or []
        if sources:
            src_lines = []
            for s in sources[:6]:
                src = s.get("source", "source")
                page = s.get("page")
                snippet = (s.get("snippet") or "").strip()
                if page is not None:
                    src = f"{src} (p.{page})"
                if snippet:
                    src_lines.append(f"- {src}: {snippet[:220]}")
                else:
                    src_lines.append(f"- {src}")
            answer = answer.rstrip() + "\n\nSources:\n" + "\n".join(src_lines)
        history = history + [[question, answer]]
        return history, history
    except Exception as e:
        history = history + [[question, f"❌ {e}"]]
        return history, history


def health() -> str:
    try:
        data = _req("GET", "/health")
        return f"Backend OK • {data.get('timestamp','')}"
    except Exception as e:
        return f"Backend not reachable: {e}"


def build_ui():
    with gr.Blocks(title="Pet Nutrition AI (Gradio)") as demo:
        token_state = gr.State("")
        pets_state = gr.State([])  # list of (label, id)

        gr.Markdown("# Pet Nutrition AI\nGradio frontend (talks to FastAPI backend).")

        with gr.Row():
            backend_info = gr.Markdown(f"**BACKEND_URL:** `{BACKEND_URL}`")
            health_btn = gr.Button("Check backend health", scale=0)
        health_out = gr.Textbox(label="Health", interactive=False)
        health_btn.click(fn=health, outputs=health_out)

        with gr.Tabs():
            with gr.Tab("Auth"):
                gr.Markdown("### Login / Signup")
                email = gr.Textbox(label="Email", placeholder="you@example.com")
                password = gr.Textbox(label="Password", type="password")

                with gr.Row():
                    signup_btn = gr.Button("Sign up")
                    login_btn = gr.Button("Log in")
                    logout_btn = gr.Button("Log out")

                auth_status = gr.Markdown("Not logged in.")
                me_btn = gr.Button("Who am I?")
                me_out = gr.Textbox(label="Current user", interactive=False)

                signup_btn.click(fn=signup, inputs=[email, password], outputs=[token_state, auth_status])
                login_btn.click(fn=login, inputs=[email, password], outputs=[token_state, auth_status])
                logout_btn.click(fn=logout, inputs=[token_state], outputs=[token_state, auth_status])
                me_btn.click(fn=whoami, inputs=[token_state], outputs=[me_out])

            with gr.Tab("Pets"):
                gr.Markdown("### Manage pets (requires login)")
                refresh_btn = gr.Button("Refresh pet list")
                pet_status = gr.Markdown("")
                pet_dropdown = gr.Dropdown(label="Select pet", choices=[], value=None)

                def _refresh(token: str):
                    choices, msg = list_pets(token)
                    # Gradio wants choices as list[str] or list[tuple[str, value]]
                    return gr.update(choices=choices, value=None), choices, msg

                refresh_btn.click(fn=_refresh, inputs=[token_state], outputs=[pet_dropdown, pets_state, pet_status])

                gr.Markdown("### Create pet")
                with gr.Row():
                    name = gr.Textbox(label="Name")
                    species = gr.Dropdown(label="Species", choices=["dog", "cat"], value="dog")
                with gr.Row():
                    breed = gr.Textbox(label="Breed", value="mixed")
                    activity_level = gr.Dropdown(
                        label="Activity level",
                        choices=["sedentary", "moderate", "active", "very_active"],
                        value="moderate",
                    )
                with gr.Row():
                    age_years = gr.Number(label="Age (years)", value=2)
                    weight_kg = gr.Number(label="Weight (kg)", value=10)
                    is_neutered = gr.Checkbox(label="Neutered", value=False)

                health_conditions = gr.Textbox(label="Health conditions (optional)")
                allergies = gr.Textbox(label="Allergies (optional)")

                create_btn = gr.Button("Create pet")
                create_out = gr.Markdown("")

                create_btn.click(
                    fn=create_pet,
                    inputs=[
                        token_state,
                        name,
                        species,
                        breed,
                        age_years,
                        weight_kg,
                        is_neutered,
                        activity_level,
                        health_conditions,
                        allergies,
                    ],
                    outputs=[create_out],
                ).then(
                    fn=_refresh,
                    inputs=[token_state],
                    outputs=[pet_dropdown, pets_state, pet_status],
                )

            with gr.Tab("Activity"):
                gr.Markdown("### Activity logs (requires login)")
                with gr.Row():
                    pet_id = gr.Dropdown(label="Pet", choices=[], value=None)
                    limit = gr.Slider(label="Limit", minimum=1, maximum=365, value=30, step=1)
                # keep this dropdown synced with Pets tab
                def _sync_pets(choices):
                    return gr.update(choices=choices, value=None)
                pet_dropdown.change(fn=_sync_pets, inputs=[pets_state], outputs=[pet_id])

                with gr.Row():
                    day = gr.Textbox(label="Date (YYYY-MM-DD)", value=str(date.today()))
                    steps = gr.Number(label="Steps", value=0)
                    active_minutes = gr.Number(label="Active minutes", value=0)

                save_btn = gr.Button("Save / Update log")
                save_out = gr.Markdown("")

                save_btn.click(fn=add_activity, inputs=[token_state, pet_id, day, steps, active_minutes], outputs=[save_out])

                list_btn = gr.Button("Load logs")
                logs_df = gr.Dataframe(headers=["date", "steps", "active_minutes", "calories_burned"], value=[], interactive=False)
                logs_status = gr.Markdown("")
                list_btn.click(fn=list_activity, inputs=[token_state, pet_id, limit], outputs=[logs_df, logs_status])

            with gr.Tab("Nutrition Chat"):
                gr.Markdown("### Chat with the AI (RAG) (requires login)")
                with gr.Row():
                    chat_pet = gr.Dropdown(label="Pet (optional)", choices=[], value=None)
                    sync_btn = gr.Button("Sync pets", scale=0)

                def _sync_choices(choices):
                    return gr.update(choices=choices, value=None)

                sync_btn.click(fn=_refresh, inputs=[token_state], outputs=[pet_dropdown, pets_state, pet_status]).then(
                    fn=_sync_choices, inputs=[pets_state], outputs=[chat_pet]
                )
                pet_dropdown.change(fn=_sync_choices, inputs=[pets_state], outputs=[chat_pet])

                chatbot = gr.Chatbot(label="Conversation", height=420)
                chat_state = gr.State([])

                question = gr.Textbox(label="Your question", placeholder="Ask about nutrition, feeding, allergies, etc.")
                send_btn = gr.Button("Send")

                send_btn.click(
                    fn=chat_submit,
                    inputs=[token_state, chat_pet, question, chat_state],
                    outputs=[chatbot, chat_state],
                )

        gr.Markdown(
            "Tip: If running via Docker Compose, open the Gradio UI on the mapped port (default 7860)."
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port, show_error=True)
