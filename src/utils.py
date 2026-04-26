import json
import numpy as np


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