import os
import logging
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


MODEL_NAME = "gpt-4o"
MAX_CONTEXT_LENGTH = 32000
client = OpenAI(api_key='sk-proj-')  

class ProcessingError(Exception):
    pass

def create_retry_decorator():
    def after_retry(retry_state):
        if retry_state.attempt_number >= 2:
            raise ProcessingError("Failed to process after maximum retries")
        print(f"Retry attempt {retry_state.attempt_number} after {retry_state.outcome.exception()}")
    return retry(
        retry=retry_if_exception_type((openai.APIError, openai.APITimeoutError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=after_retry
    )

@create_retry_decorator()
def make_api_call(client, messages, model):
    return client.chat.completions.create(
        messages=messages,
        model=model,
        stream=False,
    )

def summarize(source_text):
    chat_completion = client.chat.completions.create(model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": """
            You are an expert summarizer tasked with condensing transcripts from long videos.
            Your summary should:
            - Extract key discussions, arguments, and speaker perspectives.
            - Present the flow of conversation or narrative clearly.
            - Keep it precise yet detailed, with no unnecessary repetition.
            """,
        },
        {
            "role": "user",
            "content": source_text,
        }
    ])
    return chat_completion.choices[0].message.content

def chunk_text(text, max_chunk_size=MAX_CONTEXT_LENGTH):
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        current_len += len(word) + 1  
        current_chunk.append(word)
        if current_len >= max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def process_transcript_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    chunks = chunk_text(full_text)
    summaries = []

    for idx, chunk in enumerate(chunks):
        try:
            print(f"Processing chunk {idx + 1}/{len(chunks)}...")
            summary = summarize(chunk)
            summaries.append(f"--- Summary of Chunk {idx + 1} ---\n{summary}\n")
        except Exception as e:
            logging.error(f"Failed to summarize chunk {idx + 1}: {str(e)}")

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write('\n\n'.join(summaries))
    print(f"Summary written to {output_file}")


process_transcript_file(r"rosenbeef_nemo_1st.txt", "rosenbeef_nemo_1st_summary.txt")
