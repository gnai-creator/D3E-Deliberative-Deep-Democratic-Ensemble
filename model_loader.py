# model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from runtime_utils import log

model_id = "Qwen/Qwen2.5-7B-Instruct"

log(f"[INFO] Carregando modelo '{model_id}' com suporte a chat_template...")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Verificação do chat_template
if not tokenizer.chat_template:
    raise ValueError(f"O tokenizer carregado de '{model_id}' não possui um chat_template configurado.")
log("[INFO] Modelo com chat_template carregado com sucesso!")
