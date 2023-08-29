import gradio as gr
from transformers.pipelines import pipeline

gr.Interface.from_pipeline(
    pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")
).launch()
