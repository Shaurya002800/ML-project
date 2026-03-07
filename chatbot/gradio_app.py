import gradio as gr
import sys
sys.path.append('..')
from agrigpt.rag_pipeline import agrigpt_answer

def chat(message, history):
    # history is list of [user_msg, bot_msg] pairs
    answer = agrigpt_answer(message)
    return answer

demo = gr.ChatInterface(
    fn=chat,
    title="🌿 AgriGPT — Your Agriculture Assistant",
    description="""Ask me anything about:
    • Crop diseases and pesticide treatment
    • Farming practices and soil health
    • Government schemes and how to apply
    • Best time to plant or harvest your crops""",
    examples=[
        "What pesticide should I use for tomato early blight?",
        "How do I apply for PM-KISAN scheme?",
        "When is the best time to plant onions in Tamil Nadu?",
        "How to improve soil fertility naturally?",
        "What are the symptoms of rice blast disease?"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives a public URL instantly