
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="SciReason AI",
    page_icon="🔬",
    layout="centered"
)

MODEL_ID = "sansyed/SciReason-AI"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    model.eval()
    return tokenizer, model

st.title("🔬 SciReason AI")
st.caption("Scientific Reasoning and Explanation powered by a fine-tuned Qwen2.5-0.5B model.")

st.info(
    "Enter a scientific question to generate an explanation. "
    "Generated answers may contain errors and should be independently verified."
)

question = st.text_area(
    "Your scientific question",
    placeholder="Why does the Moon appear to change shape during the month?",
    height=120
)

if st.button("Generate Explanation", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Loading model and generating explanation..."):
            try:
                tokenizer, model = load_model()

                prompt = (
                    "Answer the following scientific question with "
                    "a clear explanation.\n\n"
                    f"Question: {question.strip()}\n"
                    "Explanation:"
                )

                inputs = tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )

                answer = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                st.subheader("Generated Explanation")
                st.write(answer if answer.strip() else "No explanation was generated.")

            except Exception as e:
                st.error(f"Model loading or generation failed: {e}")

st.divider()
st.caption("MAAI 902 | Natural Language Processing & LLMs | SciReason AI")
