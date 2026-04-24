import base64
import json
import os
from io import BytesIO

import datasets
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from tqdm import tqdm

# Configuration
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'

GUIDE = """
## Answer Instruction 
Please provide an answer to the question outlined above. Your response should adhere to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:
{{
    "solution": "[Detailed step-by-step explanation]",
    "short answer": "[Concise Answer]"
}}
"""


# --- Utility Functions ---

def preprocess(text_input):
    """Clean and extract JSON content from model response."""
    text_input = str(text_input)
    if 0 <= text_input.find("{") < text_input.rfind("}"):
        text_input = text_input[text_input.find("{"): text_input.rfind("}") + 1]
    
    clean_text = text_input.replace("\\", "").replace("\\n", "\n")
    return clean_text


def transfer(value_str):
    """Handle pi symbols and convert strings to floats."""
    if "\u03c0" in value_str:
        parts = value_str.split('\u03c0')
        multiplier = parts[0] if parts[0] else '1'
        return float(multiplier) * np.pi
    return float(value_str)


def parse_answer(answer, answer_type="multiple choice"):
    """Parse raw answer strings based on expected type."""
    if answer_type == "float":
        if str(answer).isdigit():
            return True, float(answer)
        
        try:
            parts = str(answer).split(' ')
            parsed_val = transfer(parts[0])
            return True, parsed_val
        except Exception:
            return False, None

    elif answer_type == "multiple choice":
        answer_str = str(answer).upper()
        if len(answer_str) == 1:
            return True, answer_str
        
        in_flag = [ch in answer_str for ch in 'ABCDE']
        if sum(in_flag) == 1:
            for ch in 'ABCDE':
                if ch in answer_str:
                    return True, ch
        return False, None

    return True, str(answer)


def evaluate_prediction(pred_text, ground_truth, answer_type):
    """Compare prediction against ground truth logic."""
    pred_clean = preprocess(pred_text)
    succeed, short_answer = False, None
    
    try:
        dj = json.loads(pred_clean, strict=False)
        short_answer = dj.get("short answer")
        if short_answer is not None:
            succeed, short_answer = parse_answer(short_answer, answer_type=answer_type)
    except Exception:
        pass

    if answer_type == 'float':
        if succeed and short_answer is not None:
            try:
                diff = float(short_answer) - float(ground_truth)
                return abs(diff) <= 0.001
            except Exception:
                return False
        return False
        
    elif answer_type == 'multiple choice':
        if succeed and short_answer is not None:
            return str(short_answer).strip().upper() == str(ground_truth).strip().upper()
        return str(ground_truth).strip().upper() in pred_text[:3].upper()
            
    else:
        if succeed and short_answer is not None:
            return str(short_answer).lower().strip() in str(ground_truth).lower().strip()
        return False


def encode_image(img):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# --- Main Execution ---

def main():
    print("\nLoading Dataset...")
    dataset = datasets.load_dataset('DynaMath/DynaMath_Sample', split='sample_variant1')
    print(dataset)

    print("\nInitializing LLM...")
    llm = ChatOpenAI(
        model="qwen3.5",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        max_tokens=2048, 
        temperature=0.7,
        n=4,
        top_p=0.80,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0
        }
    )
    print(llm)

    print("\nEvaluating samples...")
    results = []
    for row in tqdm(dataset):
        # Extract row data
        question = row['question']
        ground_truth = row['ground_truth']
        answer_type = row['answer_type']
        subject = row['subject']
        img = row['decoded_image']

        # Determine instruction suffix
        if answer_type == 'multiple choice':
            inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif answer_type == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."
            
        # Construct multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {question}\n" + GUIDE.format(INST=inst)
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )

        try:
            response = llm.generate([[message]])
            sample_correctness = []
            
            for generation in response.generations[0]:
                is_correct = evaluate_prediction(generation.text, ground_truth, answer_type)
                sample_correctness.append(is_correct)
        except Exception as e:
            print(f"\nInference failed for ID {row['id']}: {e}")
            sample_correctness = [False]
        
        # Calculate metrics
        results.append({
            'id': row['id'],
            'subject': subject,
            'pass@1': 1.0 if sample_correctness[0] else 0.0,
            'pass@4': 1.0 if any(sample_correctness) else 0.0,
            'avg@4': sum(sample_correctness) / len(sample_correctness),
        })

    # Summary Statistics
    print("\n" + "="*30)
    print("RESULTS SUMMARY")
    print("="*30)
    
    df_results = pd.DataFrame(results)
    
    overall_p1 = df_results['pass@1'].mean()
    overall_p4 = df_results['pass@4'].mean()
    overall_avg = df_results['avg@4'].mean()
    
    print(f"Overall: Pass@1 = {overall_p1:.2%} | Pass@4 = {overall_p4:.2%} | Avg@4 = {overall_avg:.2%}")
    print("-" * 30)

    subject_metrics = df_results.groupby('subject')[['pass@1', 'pass@4', 'avg@4']].mean()
    for sub, res in subject_metrics.iterrows():
        print(f"{sub:15}: Pass@1 = {res['pass@1']:.2%} | Pass@4 = {res['pass@4']:.2%} | Avg@4 = {res['avg@4']:.2%}")


if __name__ == "__main__":
    main()