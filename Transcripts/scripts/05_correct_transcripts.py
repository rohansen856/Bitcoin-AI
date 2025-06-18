import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GRMR_MODEL_NAME = "qingy2024/GRMR-V3-Q1.7B" 
DEFAULT_TRANSCRIPT_SOURCE_DIR = "./transcripts_whisper"  # transcripts_nemo or transcripts_whispe
OUTPUT_DIR = "./transcripts_corrected_whisper"

correction_model = None
correction_tokenizer = None

def load_correction_model_and_tokenizer(model_name=GRMR_MODEL_NAME):
    global correction_model, correction_tokenizer
    if correction_model is not None and correction_tokenizer is not None:
        print("Correction model already loaded.")
        return correction_model, correction_tokenizer

    device_map = "auto" # Automatically uses GPU if available
    print(f"Loading GRMR correction model: {model_name} (this may take time and VRAM)...")
    try:
        correction_tokenizer = AutoTokenizer.from_pretrained(model_name)
        correction_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if supported, else float16 for GPU
            trust_remote_code=True # GRMR might require this
        )
        if correction_tokenizer.pad_token_id is None:
            correction_tokenizer.pad_token_id = correction_tokenizer.eos_token_id
        
        print("GRMR Correction model and tokenizer loaded.")
        return correction_model, correction_tokenizer
    except Exception as e:
        print(f"Error loading GRMR model: {e}")
        return None, None

def correct_text_with_grmr(text_to_correct, tokenizer, model):
    if not text_to_correct.strip():
        print("Input text is empty, skipping correction.")
        return ""


    instruction = "Correct the grammar and punctuation of the following text. Preserve the original meaning and speaker style as much as possible. Only output the corrected text."
    messages = [
        {"role": "user", "content": f"{instruction}\n\nText to correct:\n{text_to_correct}"}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"    Warning: apply_chat_template failed ({e}). Falling back to manual prompt string.")
        prompt = f"Human: {instruction}\n\nText to correct:\n{text_to_correct}\n\nAssistant:"


    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device) 
    
    estimated_output_length = len(tokenizer.encode(text_to_correct)) + 200 # A bit of buffer
    max_new_tokens = min(2048, estimated_output_length) # Cap at a reasonable max

    print(f"Correcting text (input tokens: {inputs['input_ids'].shape[1]}, max_new_tokens: {max_new_tokens})...")
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,      
            top_p=0.9,            
            do_sample=True, 
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        
        full_decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Assistant:" in full_decoded_text:
            corrected_text = full_decoded_text.split("Assistant:")[-1].strip()
        elif "ASSISTANT:" in full_decoded_text:
            corrected_text = full_decoded_text.split("ASSISTANT:")[-1].strip()
        else:
            if prompt.endswith("Assistant:"):
                 if full_decoded_text.startswith(prompt[:-len("Assistant:")]):
                    corrected_text = full_decoded_text[len(prompt[:-len("Assistant:")]):].strip()
                 else:
                    corrected_text = full_decoded_text
            else:
                corrected_text = full_decoded_text
        
        return corrected_text

    except Exception as e_gen:
        print(f" Error during GRMR model.generate(): {e_gen}")
        return f"Error during correction: {e_gen}"


def batch_correct_transcripts(input_dir_to_check, output_dir_corrected):
    os.makedirs(output_dir_corrected, exist_ok=True)
    
    model, tokenizer = load_correction_model_and_tokenizer()
    if model is None or tokenizer is None:
        print("Correction model not loaded. Aborting batch correction.")
        return

    print(f"Starting transcript correction from '{input_dir_to_check}' to '{output_dir_corrected}'...")
    
    processed_count = 0
    error_count = 0

    if not os.path.exists(input_dir_to_check) or not os.listdir(input_dir_to_check):
        print(f"Input directory '{input_dir_to_check}' is empty or does not exist. Skipping correction for this source.")
        return
        
    for filename in os.listdir(input_dir_to_check):
        if filename.lower().endswith(".txt"):
            input_path = os.path.join(input_dir_to_check, filename)
            
            print(f"\n Correcting: {filename} from {input_dir_to_check}")
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    original_text = f.read()

                if not original_text.strip():
                    print(f"    File {filename} is empty. Skipping.")
                    with open(os.path.join(output_dir_corrected, f"corrected_{filename}"), "w", encoding="utf-8") as f_out:
                        f_out.write("") # Create an empty corrected file
                    continue

                corrected_text = correct_text_with_grmr(original_text, tokenizer, model)
                
                base_name = os.path.splitext(filename)[0]
                source_suffix = "nemo" if "nemo" in input_dir_to_check.lower() else "whisper"
                output_filename = f"{base_name}_corrected_{source_suffix}.txt"
                output_path = os.path.join(output_dir_corrected, output_filename)
                
                with open(output_path, "w", encoding="utf-8") as f_out:
                    f_out.write(corrected_text.strip() + "\n")
                print(f"Corrected transcript saved to: {output_path}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                error_count +=1
        else:
            print(f"Skipping non-TXT file: {filename}")

    print(f"\nTranscript correction from '{input_dir_to_check}' finished. Processed: {processed_count}, Errors: {error_count}")


if __name__ == "__main__":
    if __name__ == "__main__":
        input_source_to_use = DEFAULT_TRANSCRIPT_SOURCE_DIR
        batch_correct_transcripts(input_dir_to_check=input_source_to_use, output_dir_corrected=OUTPUT_DIR)