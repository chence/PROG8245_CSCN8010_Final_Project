from __future__ import annotations

import argparse
from typing import Any

import gradio as gr

from src.config import get_config
from src.database import ChatDatabase
from src.predict import MediChatEngine
from src.speech_to_text import transcribe_audio
from src.translation import detect_language, translate_text
from src.utils import compact_text

engine: MediChatEngine | None = None
database = ChatDatabase()
APP_CSS = """
html, body, .gradio-container {
    height: 100%;
    margin: 0;
    overflow: hidden;
}
.gradio-container {
    background:
        radial-gradient(circle at top left, #f4f8ff 0%, #eef3f8 38%, #e8edf4 100%);
}
#page-shell {
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
#app-header {
    flex: 0 0 auto;
    padding: 5px 8px 5px;
    border-bottom: 1px solid #d8e1ee;
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(10px);
}
#app-header h1 {
    margin: 0;
    font-size: 1.15rem;
    display: inline;
}
#app-header p {
    margin: 0;
    color: #4f6278;
    display: inline;
    margin-left: 10px;
    font-size: 0.95rem;
}
#content-shell {
    flex: 1 1 auto;
    min-height: 0;
    padding: 1px 1px;
    gap: 6px;
    align-items: stretch;
    flex-wrap: nowrap !important;
}
#history-sidebar,
#details-sidebar,
#center-panel {
    min-height: 0;
    height: 100%;
    box-sizing: border-box;
}
#history-sidebar,
#details-sidebar {
    flex: 0 0 280px !important;
    width: 280px;
    min-width: 280px;
    max-width: 280px;
    display: flex;
    flex-direction: column;
    padding: 14px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d9e2ef;
    border-radius: 18px;
    box-shadow: 0 14px 40px rgba(104, 128, 156, 0.12);
    overflow: hidden;
    transition: width 0.2s ease, min-width 0.2s ease, max-width 0.2s ease, padding 0.2s ease;
}
#history-sidebar.sidebar-collapsed {
    flex: 0 0 112px !important;
    width: 112px;
    min-width: 112px;
    max-width: 112px;
    padding-left: 8px;
    padding-right: 8px;
}
#details-sidebar.sidebar-collapsed {
    flex: 0 0 156px !important;
    width: 156px;
    min-width: 156px;
    max-width: 156px;
    padding-left: 8px;
    padding-right: 8px;
}
.sidebar-header {
    align-items: center;
    justify-content: stretch;
    gap: 10px;
    margin-bottom: 12px;
}
.sidebar-body {
    flex: 1 1 auto;
    min-height: 0;
    gap: 12px;
    display: flex;
    flex-direction: column;
}
.sidebar-toggle {
    width: 100% !important;
    min-width: 0 !important;
    justify-content: center !important;
    border-radius: 14px !important;
    border: 1px solid #c8d8ef !important;
    background: linear-gradient(180deg, #f8fbff 0%, #eaf2ff 100%) !important;
    color: #29456f !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75), 0 4px 14px rgba(88, 115, 166, 0.08);
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.sidebar-toggle:hover {
    border-color: #9db9e8 !important;
    background: linear-gradient(180deg, #ffffff 0%, #edf4ff 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 8px 18px rgba(88, 115, 166, 0.12);
    transform: translateY(-1px);
}
.sidebar-toggle:active {
    transform: translateY(0);
    box-shadow: inset 0 2px 4px rgba(88, 115, 166, 0.12);
}
#history-sidebar .sidebar-toggle {
    color: #204768 !important;
}
#details-sidebar .sidebar-toggle {
    color: #5a3f75 !important;
    background: linear-gradient(180deg, #fbf9ff 0%, #f1ebff 100%) !important;
    border-color: #d5c8ef !important;
}
#details-sidebar .sidebar-toggle:hover {
    background: linear-gradient(180deg, #ffffff 0%, #f4efff 100%) !important;
    border-color: #bda7ea !important;
}
#history-sidebar.sidebar-collapsed .sidebar-toggle,
#details-sidebar.sidebar-collapsed .sidebar-toggle {
    font-size: 0.9rem !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
}
#center-panel {
    flex: 1 1 auto !important;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow: hidden;
}
#session-list {
    flex: 1 1 auto;
    min-height: 0;
    max-height: none;
    overflow-y: auto;
    padding-right: 4px;
}
#session-list label {
    border: 1px solid #d7dde8;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    background: #ffffff;
    cursor: pointer;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}
#session-list label:hover {
    background: #f6f9fc;
    border-color: #b8c7dc;
}
#session-list input[type="radio"] {
    display: none;
}
#session-list label:has(input[type="radio"]:checked) {
    background: #e8f1ff;
    border-color: #6b9cff;
    box-shadow: inset 0 0 0 1px #6b9cff;
}
#conversation-card {
    flex: 1 1 auto !important;
    min-height: 0;
    display: flex;
    flex-direction: column;
    padding: 10px;
    background: rgba(255, 255, 255, 0.97);
    border: 1px solid #d9e2ef;
    border-radius: 18px;
    box-shadow: 0 14px 40px rgba(104, 128, 156, 0.12);
    overflow: hidden;
}
#chat-panel {
    flex: 1 1 0;
    min-height: 0;
    height: 100% !important;
}
#composer-shell {
    flex: 0 0 auto !important;
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.97);
    border: 1px solid #d9e2ef;
    border-radius: 18px;
    box-shadow: 0 14px 40px rgba(104, 128, 156, 0.12);
}
#message-row {
    align-items: stretch;
    gap: 12px;
}
#action-column {
    display: flex;
    flex-direction: column;
    justify-content: stretch;
    gap: 8px;
}
#message-box {
    margin-bottom: 0 !important;
}
#message-box textarea {
    min-height: 64px !important;
    max-height: 88px !important;
}
#send-button {
    flex: 1 1 0;
    min-height: 34px;
    width: 100%;
}
#new-session-button {
    min-height: 34px;
    width: 100%;
    background: #2f6fed !important;
    border-color: #2f6fed !important;
    color: #ffffff !important;
}
#audio-capture-panel {
    margin: 0;
    min-height: 0 !important;
    flex: 0 0 auto !important;
}
#audio-capture-panel .gradio-audio,
#audio-capture-panel .gradio-audio > .wrap,
#audio-capture-panel .gradio-audio > .wrap > .block,
#audio-capture-panel .gradio-audio .block {
    width: 100%;
    min-height: 0 !important;
}
#audio-capture-panel .gradio-audio,
#audio-capture-panel .gradio-audio .block,
#audio-capture-panel .gradio-audio .wrap {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
#audio-capture-panel .record-container,
#audio-capture-panel .controls,
#audio-capture-panel .control-wrap,
#audio-capture-panel button,
#audio-capture-panel audio {
    min-height: 32px !important;
}
#audio-capture-panel .record-container,
#audio-capture-panel .controls {
    padding: 0 !important;
    margin: 0 !important;
}
#audio-capture-panel .record-container {
    gap: 8px !important;
}
#audio-capture-panel audio {
    max-height: 36px !important;
}
#chat-panel .icon-button-wrapper.top-panel,
#chat-panel .top-panel,
#chat-panel button[aria-label="Copy"],
#chat-panel button[aria-label="Share"],
#chat-panel button[aria-label="Download"] {
    display: none !important;
}
#details-box {
    flex: 1 1 auto;
    min-height: 0;
}
#details-box textarea,
#details-box .scroll-hide {
    height: 100% !important;
    min-height: 0 !important;
}
#app-footer {
    flex: 0 0 auto;
    padding: 8px 20px 20px;
    border-top: 1px solid #d8e1ee;
    color: #5d6e82;
    background: rgba(255, 255, 255, 0.88);
}
#app-footer p {
    margin: 0;
    text-align: center;
}
@media (max-width: 960px) {
    #page-shell {
        height: auto;
        min-height: 100vh;
    }
    #content-shell {
        flex-direction: column;
        overflow: auto;
    }
    #history-sidebar,
    #details-sidebar,
    #center-panel {
        width: 100%;
        min-width: 0;
        max-width: none;
        height: auto;
    }
    #history-sidebar.sidebar-collapsed,
    #details-sidebar.sidebar-collapsed {
        flex: 0 0 100% !important;
        width: 100%;
        min-width: 0;
        max-width: none;
    }
    #conversation-card {
        min-height: 420px;
    }
    #chat-panel {
        min-height: 320px;
    }
    #message-row {
        flex-wrap: wrap;
    }
    #action-column {
        width: 100%;
    }
    #app-header p {
        display: block;
        margin-left: 0;
        margin-top: 4px;
    }
}
"""


def get_engine() -> MediChatEngine:
    global engine
    if engine is None:
        engine = MediChatEngine(model_name=get_config().default_model_name)
    return engine


def _format_response(result: dict[str, Any], transcribed_text: str = "") -> tuple[str, str]:
    main_text = result["response"]
    if result["language"] != "en":
        main_text = (
            f"{result['response']}\n\n"
            f"English translation:\n{result['english_response']}"
        )

    details = [
        f"Agent route: {result.get('route', 'standard_qa')}",
        f"Route reason: {result.get('route_reason', 'n/a')}",
        f"Intent: {result['intent']}",
        f"Detected language: {result['language']}",
        f"Classifier confidence: {result['confidence']:.3f}",
        f"Retrieval score: {result['retrieval_score']:.3f}",
        f"Supported answer: {'yes' if result['supported'] else 'no'}",
        f"Support mode: {result.get('support_mode', 'unknown')}",
    ]
    if result.get("clarification_reason"):
        details.append(f"Clarification trigger: {result['clarification_reason']}")
    if result.get("session_summary"):
        details.append(f"Session summary: {result['session_summary']}")
    if transcribed_text:
        details.append(f"Transcribed audio: {transcribed_text}")
    return main_text, "\n".join(details)


def _format_user_message(content: str, language: str, english_text: str) -> str:
    if language != "en" and english_text and english_text != content:
        return f"{content}\n\nEnglish translation:\n{english_text}"
    return content


def _messages_to_chatbot(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    chatbot_messages: list[dict[str, str]] = []
    for message in messages:
        content = message["original_text"]
        english_text = message.get("english_text") or ""
        language = message.get("language") or "en"
        if message["role"] in {"user", "assistant"}:
            content = _format_user_message(content, language, english_text)
        chatbot_messages.append({"role": message["role"], "content": content})
    return chatbot_messages


def load_session_from_url(session_id: str | None):
    normalized_session_id = (session_id or "").strip()
    if not normalized_session_id:
        return [], {}, "Start a new conversation or open a shared session link.", ""

    messages = database.get_messages(normalized_session_id)
    if not messages:
        return [], {}, f"Session `{normalized_session_id}` was not found.", ""

    state = {"session_id": normalized_session_id}
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    details = f"Loaded session {normalized_session_id} with {len(messages)} messages."
    session_summary = database.get_session_summary(normalized_session_id)
    if assistant_messages:
        last_assistant = assistant_messages[-1]
        details_lines = [details]
        if last_assistant.get("intent"):
            details_lines.append(f"Last intent: {last_assistant['intent']}")
        if last_assistant.get("confidence") is not None:
            details_lines.append(f"Last confidence: {last_assistant['confidence']:.3f}")
        route = (last_assistant.get("metadata") or {}).get("route")
        if route:
            details_lines.append(f"Last route: {route}")
        if session_summary:
            details_lines.append(f"Session summary: {session_summary}")
        details = "\n".join(details_lines)
    elif session_summary:
        details = f"{details}\nSession summary: {session_summary}"

    return _messages_to_chatbot(messages), state, details, normalized_session_id


def get_session_choices(selected_session_id: str | None = None):
    sessions = database.list_sessions(limit=50)
    choices = []
    selected_value = None
    for session in sessions:
        label = (
            f"{session['updated_at'][:16]} | "
            f"{compact_text(session['first_user_message'] or 'New conversation', 40)} "
            f"({session['message_count']})"
        )
        value = session["session_id"]
        choices.append((label, value))
        if selected_session_id and selected_session_id == value:
            selected_value = value
    return gr.update(choices=choices, value=selected_value)


def load_session_from_picker(session_id: str | None):
    return load_session_from_url(session_id)


def clear_session_picker():
    return gr.update(value=None)


def initialize_message_box_behavior():
    return None


def start_new_session():
    return [], {}, "Start a new conversation.", ""


def handle_text_message(
    pending_message: str,
    chat_history: list[dict[str, str]] | None,
    app_state: dict[str, Any] | None,
):
    message = (pending_message or "").strip()
    app_state = app_state or {}
    chat_history = chat_history or []
    if not message:
        return chat_history, app_state, "Enter a question before sending.", "", app_state.get("session_id", "")

    source_language = detect_language(message)
    translated_history = chat_history
    if source_language != "en":
        english_text = translate_text(message, "en", source_language=source_language).text
        translated_history = chat_history.copy()
        if translated_history and translated_history[-1].get("role") == "user":
            translated_history[-1] = {
                "role": "user",
                "content": f"{message}\n\nEnglish translation:\n{english_text}",
            }
        yield translated_history, app_state, "Generating response...", pending_message, app_state.get("session_id", "")

    result = get_engine().process_message(message, session_id=app_state.get("session_id"))
    bot_text, details = _format_response(result)

    final_history = translated_history + [{"role": "assistant", "content": bot_text}]
    app_state["session_id"] = result["session_id"]
    app_state["last_result"] = result
    yield final_history, app_state, details, "", result["session_id"]


def handle_audio_message(
    audio_file: str | None,
    chat_history: list[dict[str, str]] | None,
    app_state: dict[str, Any] | None,
):
    if not audio_file:
        return (
            chat_history or [],
            app_state or {},
            "Recording cancelled.",
            app_state.get("session_id", "") if app_state else "",
            None,
            "",
        )

    transcription = transcribe_audio(audio_file)
    if not transcription.success:
        return (
            chat_history or [],
            app_state or {},
            transcription.error or "Audio transcription failed.",
            app_state.get("session_id", "") if app_state else "",
            None,
            "",
        )

    app_state = app_state or {}
    chat_history = chat_history or []
    result = get_engine().process_message(transcription.text, session_id=app_state.get("session_id"))
    bot_text, details = _format_response(result, transcribed_text=transcription.text)
    user_text = _format_user_message(
        f"[Audio] {transcription.text}",
        result["language"],
        f"[Audio] {result['english_text']}" if result["language"] != "en" else result["english_text"],
    )

    if chat_history and chat_history[-1].get("content") == "[Voice message]":
        chat_history[-1] = {"role": "user", "content": user_text}
    else:
        chat_history = chat_history + [{"role": "user", "content": user_text}]
    chat_history = chat_history + [{"role": "assistant", "content": bot_text}]
    app_state["session_id"] = result["session_id"]
    app_state["last_result"] = result
    return chat_history, app_state, details, result["session_id"], None, ""


def reset_chat():
    return [], {}, "Session cleared.", ""


def queue_text_message(
    user_message: str,
    chat_history: list[dict[str, str]] | None,
):
    message = (user_message or "").strip()
    chat_history = chat_history or []
    if not message:
        return chat_history, "", "", "Enter a question before sending."
    updated_history = chat_history + [{"role": "user", "content": message}]
    return updated_history, "", message, "Generating response..."


def queue_audio_message(
    audio_file: str | None,
    chat_history: list[dict[str, str]] | None,
):
    chat_history = chat_history or []
    if not audio_file:
        return chat_history, None, "", "Recording cancelled."
    updated_history = chat_history + [{"role": "user", "content": "[Voice message]"}]
    return updated_history, None, audio_file, "Transcribing and generating response..."


def cancel_audio_message(
    chat_history: list[dict[str, str]] | None,
    app_state: dict[str, Any] | None,
):
    chat_history = chat_history or []
    if chat_history and chat_history[-1].get("content") == "[Voice message]":
        chat_history = chat_history[:-1]
    session_id = app_state.get("session_id", "") if app_state else ""
    return chat_history, app_state or {}, "Recording cancelled.", session_id, None, ""


def stream_audio_message(
    audio_file: str | None,
    chat_history: list[dict[str, str]] | None,
    app_state: dict[str, Any] | None,
):
    chat_history = chat_history or []
    app_state = app_state or {}
    session_id = app_state.get("session_id", "")

    if not audio_file:
        yield chat_history, app_state, "Recording cancelled.", session_id, None
        return

    queued_history = chat_history + [{"role": "user", "content": "[Voice message]"}]
    yield queued_history, app_state, "Transcribing and generating response...", session_id, None

    transcription = transcribe_audio(audio_file)
    if not transcription.success:
        yield (
            chat_history,
            app_state,
            transcription.error or "Audio transcription failed.",
            session_id,
            None,
        )
        return

    result = get_engine().process_message(transcription.text, session_id=session_id or None)
    bot_text, details = _format_response(result, transcribed_text=transcription.text)
    queued_history[-1] = {
        "role": "user",
        "content": _format_user_message(
            f"[Audio] {transcription.text}",
            result["language"],
            f"[Audio] {result['english_text']}" if result["language"] != "en" else result["english_text"],
        ),
    }
    final_history = queued_history + [{"role": "assistant", "content": bot_text}]
    app_state["session_id"] = result["session_id"]
    app_state["last_result"] = result
    yield final_history, app_state, details, result["session_id"], None


with gr.Blocks(title="MediChat") as demo:
    state = gr.State({})
    pending_text_state = gr.State("")
    pending_audio_state = gr.State("")
    session_id_box = gr.Textbox(label="Session ID", visible=False)

    with gr.Column(elem_id="page-shell"):
        with gr.Row(elem_id="app-header"):
            gr.Markdown("**MediChat**")

        with gr.Row(elem_id="content-shell"):
            with gr.Column(
                scale=3,
                min_width=60,
                visible=True,
                elem_id="history-sidebar",
                elem_classes=["sidebar-panel", "sidebar-collapsed"],
            ) as sidebar_column:
                with gr.Row(elem_classes="sidebar-header"):
                    sidebar_button = gr.Button("History", size="sm", variant="secondary", elem_classes="sidebar-toggle")
                with gr.Column(elem_id="history-body", elem_classes="sidebar-body"):
                    session_picker = gr.Radio(
                        label="Sessions",
                        choices=[],
                        value=None,
                        interactive=True,
                        elem_id="session-list",
                    )
                    refresh_sessions_button = gr.Button("Refresh Sessions", size="sm", variant="secondary")

            with gr.Column(scale=8, elem_id="center-panel"):
                with gr.Column(scale=1, elem_id="conversation-card"):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        height="100%",
                        elem_id="chat-panel",
                    )

                with gr.Column(scale=0, elem_id="composer-shell"):
                    with gr.Row(elem_id="audio-capture-panel"):
                        audio_input = gr.Audio(
                            show_label=False,
                            sources=["microphone"],
                            type="filepath",
                            elem_id="audio-capture",
                        )

                    with gr.Row(elem_id="message-row"):
                        with gr.Column(scale=9):
                            message_box = gr.Textbox(
                                show_label=False,
                                placeholder="Press Enter to send. Use Shift+Enter for a new line.",
                                lines=2,
                                elem_id="message-box",
                            )
                        with gr.Column(scale=3, min_width=120, elem_id="action-column"):
                            clear_button = gr.Button("New Session", size="sm", variant="primary", elem_id="new-session-button")
                            send_button = gr.Button("Send", variant="primary", elem_id="send-button")

            with gr.Column(
                scale=3,
                min_width=60,
                visible=True,
                elem_id="details-sidebar",
                elem_classes=["sidebar-panel"],
            ) as details_column:
                with gr.Row(elem_classes="sidebar-header"):
                    details_button = gr.Button("Turn Details", size="sm", variant="secondary", elem_classes="sidebar-toggle")
                with gr.Column(elem_id="details-body", elem_classes="sidebar-body"):
                    details_box = gr.Textbox(
                        show_label=False,
                        lines=18,
                        interactive=False,
                        elem_id="details-box",
                    )

        with gr.Row(elem_id="app-footer"):
            gr.Markdown(
                """
                PROG8245 & CSCN8010 | Final Project | Group 1 & 7/8 | Members: Ce Chen, Zhuoran Zhang, Haibo Yuan, Abdallah Mohamed
                """
            )

    send_button.click(
        queue_text_message,
        inputs=[message_box, chatbot],
        outputs=[chatbot, message_box, pending_text_state, details_box],
    ).then(
        handle_text_message,
        inputs=[pending_text_state, chatbot, state],
        outputs=[chatbot, state, details_box, pending_text_state, session_id_box],
    ).then(
        fn=None,
        inputs=[session_id_box],
        outputs=None,
        js="""
        (session_id) => {
            const url = new URL(window.location.href);
            if (session_id) {
                url.searchParams.set("session_id", session_id);
            } else {
                url.searchParams.delete("session_id");
            }
            window.history.replaceState({}, "", url);
        }
        """,
    ).then(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    )
    message_box.submit(
        queue_text_message,
        inputs=[message_box, chatbot],
        outputs=[chatbot, message_box, pending_text_state, details_box],
    ).then(
        handle_text_message,
        inputs=[pending_text_state, chatbot, state],
        outputs=[chatbot, state, details_box, pending_text_state, session_id_box],
    ).then(
        fn=None,
        inputs=[session_id_box],
        outputs=None,
        js="""
        (session_id) => {
            const url = new URL(window.location.href);
            if (session_id) {
                url.searchParams.set("session_id", session_id);
            } else {
                url.searchParams.delete("session_id");
            }
            window.history.replaceState({}, "", url);
        }
        """,
    ).then(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    )
    audio_input.stop_recording(
        stream_audio_message,
        inputs=[audio_input, chatbot, state],
        outputs=[chatbot, state, details_box, session_id_box, audio_input],
    ).then(
        fn=None,
        inputs=[session_id_box],
        outputs=None,
        js="""
        (session_id) => {
            const url = new URL(window.location.href);
            if (session_id) {
                url.searchParams.set("session_id", session_id);
            } else {
                url.searchParams.delete("session_id");
            }
            window.history.replaceState({}, "", url);
        }
        """,
    ).then(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    )
    audio_input.clear(
        cancel_audio_message,
        inputs=[chatbot, state],
        outputs=[chatbot, state, details_box, session_id_box, audio_input, pending_audio_state],
    )
    sidebar_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            const sidebar = document.getElementById("history-sidebar");
            const button = sidebar?.querySelector(".sidebar-toggle");
            const body = document.getElementById("history-body");
            if (sidebar) {
                sidebar.classList.toggle("sidebar-collapsed");
                const collapsed = sidebar.classList.contains("sidebar-collapsed");
                sidebar.style.width = collapsed ? "112px" : "280px";
                sidebar.style.minWidth = collapsed ? "112px" : "280px";
                sidebar.style.maxWidth = collapsed ? "112px" : "280px";
                sidebar.style.flexBasis = collapsed ? "112px" : "280px";
                if (body) {
                    body.style.display = collapsed ? "none" : "flex";
                }
                if (button) {
                    button.style.width = "100%";
                    button.style.minWidth = "0";
                }
            }
        }
        """,
    ).then(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    )
    details_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            const sidebar = document.getElementById("details-sidebar");
            const button = sidebar?.querySelector(".sidebar-toggle");
            const body = document.getElementById("details-body");
            if (sidebar) {
                sidebar.classList.toggle("sidebar-collapsed");
                const collapsed = sidebar.classList.contains("sidebar-collapsed");
                sidebar.style.width = collapsed ? "156px" : "280px";
                sidebar.style.minWidth = collapsed ? "156px" : "280px";
                sidebar.style.maxWidth = collapsed ? "156px" : "280px";
                sidebar.style.flexBasis = collapsed ? "156px" : "280px";
                if (body) {
                    body.style.display = collapsed ? "none" : "flex";
                }
                if (button) {
                    button.style.width = "100%";
                    button.style.minWidth = "0";
                }
            }
        }
        """,
    )
    refresh_sessions_button.click(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    )
    session_picker.change(
        load_session_from_picker,
        inputs=[session_picker],
        outputs=[chatbot, state, details_box, session_id_box],
    ).then(
        fn=None,
        inputs=[session_id_box],
        outputs=None,
        js="""
        (session_id) => {
            const url = new URL(window.location.href);
            if (session_id) {
                url.searchParams.set("session_id", session_id);
            } else {
                url.searchParams.delete("session_id");
            }
            window.history.replaceState({}, "", url);
        }
        """,
    ).then(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    )
    clear_button.click(start_new_session, outputs=[chatbot, state, details_box, session_id_box]).then(
        fn=None,
        inputs=[session_id_box],
        outputs=None,
        js="""
        (session_id) => {
            const url = new URL(window.location.href);
            if (session_id) {
                url.searchParams.set("session_id", session_id);
            } else {
                url.searchParams.delete("session_id");
            }
            window.history.replaceState({}, "", url);
        }
        """,
    ).then(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    )
    demo.load(
        load_session_from_url,
        inputs=[session_id_box],
        outputs=[chatbot, state, details_box, session_id_box],
        js="""
        () => {
            const url = new URL(window.location.href);
            return [url.searchParams.get("session_id") || ""];
        }
        """,
    ).then(
        get_session_choices,
        inputs=[session_id_box],
        outputs=[session_picker],
    ).then(
        fn=None,
        inputs=[],
        outputs=[],
        js="""
        () => {
            const restoreSidebar = (id, bodyId, collapsedWidth, expandedWidth, startCollapsed) => {
                const sidebar = document.getElementById(id);
                if (!sidebar) {
                    return;
                }
                sidebar.classList.toggle("sidebar-collapsed", startCollapsed);
                const width = startCollapsed ? collapsedWidth : expandedWidth;
                sidebar.style.width = width;
                sidebar.style.minWidth = width;
                sidebar.style.maxWidth = width;
                sidebar.style.flexBasis = width;
                const body = document.getElementById(bodyId);
                const button = sidebar.querySelector(".sidebar-toggle");
                if (body) {
                    body.style.display = startCollapsed ? "none" : "flex";
                }
                if (button) {
                    button.style.width = "100%";
                    button.style.minWidth = "0";
                }
            };
            restoreSidebar("history-sidebar", "history-body", "112px", "280px", true);
            restoreSidebar("details-sidebar", "details-body", "156px", "280px", false);
        }
        """,
    ).then(
        initialize_message_box_behavior,
        outputs=[],
        js="""
        () => {
            const textarea = document.querySelector('#message-box textarea');
            const sendButton = document.querySelector('#send-button button, #send-button');
            if (!textarea || !sendButton) {
                return;
            }
            if (textarea.dataset.medichatBound === 'true') {
                return;
            }
            textarea.dataset.medichatBound = 'true';
            textarea.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendButton.click();
                }
            });
        }
        """,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the MediChat Gradio app.")
    parser.add_argument(
        "--clear-sessions",
        action="store_true",
        help="Clear all stored chat sessions before launch.",
    )
    args = parser.parse_args()

    if args.clear_sessions:
        database.clear_all_sessions()
        print("Cleared all stored sessions before launch.")

    demo.launch(css=APP_CSS)
