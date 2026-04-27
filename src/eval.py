import os
import datasets
import argparse
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI

from methods import (
    BaselineMethod,
    SushilOracleMethod,
    SushilMethod,
    AkibMethod,
    VasudevMethod
)
from utils import evaluate_prediction

os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'


def main(method_name):

    print("\nDataset")
    dataset = datasets.load_dataset('DynaMath/DynaMath_Sample', split='sample_variant1')
    print(dataset)


    print("\n\nLLM")
    llm = ChatOpenAI(
        model="qwen3.5",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        max_tokens=2048,
        seed=42, 
        temperature=0.7,
        n=4,
        top_p=0.80,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "guided_json": {
                "type": "object",
                "properties": {
                    "solution": {"type": "string"},
                    "short answer": {"type": "string"}
                },
                "required": ["solution", "short answer"],
                "additionalProperties": False
            }
        }
    )
    print(llm)


    print("\n\nMethod")
    print(method_name)
    if method_name=="baseline":
        method = BaselineMethod()
    elif method_name=="sushiloracle":
        method = SushilOracleMethod()
    elif method_name=="sushil":
        method = SushilMethod()
    elif method_name=="akib":
        method = AkibMethod()
    elif method_name=="vasudev":
        method = VasudevMethod()


    print("\n\nEvaluation")
    results = []
    for row in tqdm(dataset):
        
        question_id = row['id']
        question = row['question']
        ground_truth = row['ground_truth']
        answer_type = row['answer_type']
        subject = row['subject']
        img = row['decoded_image']

        message = method(question_id, question, answer_type, subject, img, llm)

        try:
            response = llm.generate([[message]])
            
            sample_correctness = []
            for generation in response.generations[0]:
                is_correct = evaluate_prediction(generation.text, ground_truth, answer_type)
                sample_correctness.append(is_correct)
        
        except Exception as e:
            print(f"\nInference failed for ID {question_id}: {e}")
            sample_correctness = [False] * 4
        
        results.append({
            'id': question_id,
            'subject': subject,
            'pass@1': 1.0 if sample_correctness[0] else 0.0,
            'pass@4': 1.0 if any(sample_correctness) else 0.0,
            'avg@4': sum(sample_correctness) / len(sample_correctness),
        })


    print("\n\nResults")
    df_results = pd.DataFrame(results)
    
    sub = "Overall"
    print(f"{sub:20}: Pass@1 = {df_results['pass@1'].mean():.2%} | Pass@4 = {df_results['pass@4'].mean():.2%} | Avg@4 = {df_results['avg@4'].mean():.2%}")

    subject_metrics = df_results.groupby('subject')[['pass@1', 'pass@4', 'avg@4']].mean()
    for sub, res in subject_metrics.iterrows():
        print(f"{sub:20}: Pass@1 = {res['pass@1']:.2%} | Pass@4 = {res['pass@4']:.2%} | Avg@4 = {res['avg@4']:.2%}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "sushiloracle", "sushil", "akib", "vasudev"], required=True)
    args = parser.parse_args()

    main(args.method)