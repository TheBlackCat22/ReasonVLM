import json
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import datasets
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def preprocess(str1):
    str1 = str(str1)
    if 0 <= str1.find("{") < str1.rfind("}"):
        str1 = str1[str1.find("{"): str1.rfind("}") + 1]
    str2 = str1.replace("\\", "")
    str2 = str2.replace("\\n", "\n")
    return str2


def transfer(str1):
    if "\u03c0" in str1:  # Handling pi
        strs = str1.split('\u03c0')
        str1 = strs[0] if strs[0] else '1'
        return float(str1) * np.pi
    else:
        return float(str1)


def parse_answer(answer, answer_type="multiple choice"):
    if answer_type == "float":
        if str(answer).isdigit():
            return True, float(answer)
        else:
            parts = str(answer).split(' ')
            answer = parts[0]
            try:
                answer = transfer(answer)
                return True, answer
            except Exception:
                return False, None
    elif answer_type == "multiple choice":
        if len(str(answer)) == 1:
            return True, str(answer).upper()
        else:
            in_flag = [ch in str(answer).upper() for ch in 'ABCDE']
            if sum(in_flag) == 1:
                for ch in 'ABCDE':
                    if ch in str(answer).upper():
                        return True, ch
            return False, None
    else:
        return True, str(answer)


def evaluate_prediction(pred_json_str, ground_truth, answer_type):
    pred_clean = preprocess(pred_json_str)
    
    try:
        dj = json.loads(pred_clean, strict=False)
        short_answer = dj.get("short answer")
        if short_answer is None:
            return False
            
        succeed, parsed_answer = parse_answer(short_answer, answer_type=answer_type)
        if not succeed:
            return False
            
    except Exception:
        return False

    if answer_type == 'float':
        try:
            diff = float(parsed_answer) - float(ground_truth)
            return abs(diff) <= 0.001
        except Exception:
            return False
    elif answer_type == 'multiple choice':
        return parsed_answer == str(ground_truth).strip().upper()
    else:
        return str(parsed_answer).lower().strip() in str(ground_truth).lower().strip()


def pil_to_base64(img):
    """Converts a PIL Image to a base64 encoded string for LangChain."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def main():    
    llm = ChatOpenAI(
        model="qwen3.5",
        base_url="http://localhost:8000/v1",
        api_key="dummy",
        max_tokens=512,
        temperature=0.0
    )

    print(f"\n\nLoading Dataset")
    dataset = datasets.load_dataset('DynaMath/DynaMath_Sample', split='sample_variant1')
    print(dataset)
    
    GUIDE = """
## Answer Instruction 
Please provide an answer to the question outlined above. Your response should adhere to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:
{
    "solution": "[Detailed step-by-step explanation]",
    "short answer": "[Concise Answer]"
}
"""
    print(f"\n\nEvaluating")

    results = []

    for row in tqdm(dataset):
        question = row['question']
        ground_truth = row['ground_truth']
        answer_type = row['answer_type']
        subject = row['subject']
        img = row['image']

        if answer_type == 'multiple choice':
            inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif answer_type == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

        prompt_text = f"## Question\n {question}\n" + GUIDE.format(INST=inst)
        
        content = [{"type": "text", "text": prompt_text}]
        if img is not None:
            base64_img = pil_to_base64(img)
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })
            
        message = HumanMessage(content=content)

        try:
            response = llm.invoke([message])
            pred_text = response.content
        except Exception as e:
            print(f"Inference failed for ID {row['id']}: {e}")
            pred_text = ""

        is_correct = evaluate_prediction(pred_text, ground_truth, answer_type)
        
        results.append({
            'id': row['id'],
            'subject': subject,
            'is_correct': is_correct
        })

    df_results = pd.DataFrame(results)
    
    print("\n\nResults")
    
    overall_acc = df_results['is_correct'].mean()
    print(f"**Overall Accuracy:** {overall_acc:.2%}")
    
    print("\n**Subject-wise Accuracy:**")
    subject_acc = df_results.groupby('subject')['is_correct'].mean()
    for sub, acc in subject_acc.items():
        print(f"  - {sub}: {acc:.2%}")

if __name__ == "__main__":
    main()