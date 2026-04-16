from __future__ import annotations

import gradio as gr

from src.predict import predict_intent


def medichat_reply(user_text: str):
    if not user_text or not user_text.strip():
        return 'Please enter a medical question.', '', '', '', ''
    result = predict_intent(user_text)
    return (
        result['response'],           # translated response
        result['intent'],              # predicted intent
        result['language'],            # detected language
        result['english_text'],        # english question
        result['english_response'],    # english response
    )


with gr.Blocks(title='MediChat') as demo:
    gr.Markdown('# MediChat A multilingual educational medical chatbot for the PROG8245 final project.')
    with gr.Row():
        with gr.Column():
            user_text = gr.Textbox(label='Enter your question', lines=4, placeholder='Example: I have a sore throat and mild fever.')
            submit = gr.Button('Ask MediChat')
        with gr.Column():
            answer = gr.Textbox(label='Response', lines=6)
            intent = gr.Textbox(label='Predicted Intent')
            language = gr.Textbox(label='Detected Language')
    
    with gr.Row():
        with gr.Column():
            english_question = gr.Textbox(label='English Question', lines=3, interactive=False)
        with gr.Column():
            english_answer = gr.Textbox(label='English Response', lines=3, interactive=False)

    submit.click(medichat_reply, inputs=user_text, outputs=[answer, intent, language, english_question, english_answer])


if __name__ == '__main__':
    demo.launch()
