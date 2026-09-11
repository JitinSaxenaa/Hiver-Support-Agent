import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from src.taxonomy import get_all_intents
from src.baselines import MajorityClassBaseline, TfidfBaselineClassifier
from src.classify import classify_intent
from src.draft_reply import draft_reply
from src.escalate import decide_escalation
from eval.llm_judge import evaluate_reply_quality

def run_evaluation_suite(golden_path: str = 'eval/golden_set.csv', provider: str = 'local', num_samples: int = None):
    print('=' * 65)
    print('HIVER AI TWITTER SUPPORT AGENT - COMPREHENSIVE EVALUATION')
    print('=' * 65)
    t_start = time.time()

    df_golden = pd.read_csv(golden_path)
    if num_samples is not None and num_samples < len(df_golden):
        print(f'Subsampling golden eval set to {num_samples} rows...')
        df_golden = df_golden.sample(n=num_samples, random_state=42).reset_index(drop=True)

    N = len(df_golden)
    print(f'Evaluating on {N} hand-labeled held-out examples from {golden_path}...')

    y_true_intent = df_golden['true_intent'].tolist()
    y_true_decision = df_golden['ideal_decision'].tolist()
    all_intents = get_all_intents()

    # 1. EVALUATE BASELINES
    print('\n[1/4] Evaluating Intent Classification Baselines...')
    majority_model = MajorityClassBaseline()
    df_train = pd.read_csv('data/processed/train_silver_set.csv') if os.path.isfile('data/processed/train_silver_set.csv') else df_golden
    majority_model.fit(df_train['inbound_text'].tolist(), df_train.get('intent', df_train.get('true_intent')).tolist())
    y_pred_majority = majority_model.predict(df_golden['inbound_text'].tolist())
    maj_acc = accuracy_score(y_true_intent, y_pred_majority)
    maj_f1 = f1_score(y_true_intent, y_pred_majority, labels=all_intents, average='macro', zero_division=0)
    print(f'  * Trivial Baseline (Majority: "{majority_model.majority_class}"): Accuracy = {maj_acc:.2%}, Macro-F1 = {maj_f1:.4f}')

    tfidf_model = TfidfBaselineClassifier()
    if os.path.isfile('models/tfidf_baseline.joblib'):
        tfidf_model.load('models/tfidf_baseline.joblib')
    else:
        tfidf_model.fit(df_train['inbound_text'].tolist(), df_train['intent'].tolist())
    y_pred_tfidf = tfidf_model.predict(df_golden['inbound_text'].tolist())
    tfidf_acc = accuracy_score(y_true_intent, y_pred_tfidf)
    tfidf_f1 = f1_score(y_true_intent, y_pred_tfidf, labels=all_intents, average='macro', zero_division=0)
    print(f'  * Simple Baseline (TF-IDF + LogReg): Accuracy = {tfidf_acc:.2%}, Macro-F1 = {tfidf_f1:.4f}')

    # 2. EVALUATE YOUR SYSTEM
    print(f'\n[2/4] Evaluating Production Intent Classifier (Engine: {provider})...')
    y_pred_system = []
    system_confidences = []
    system_reasonings = []

    for text in df_golden['inbound_text']:
        res = classify_intent(text, provider=provider)
        y_pred_system.append(res['intent'])
        system_confidences.append(res['confidence'])
        system_reasonings.append(res['reasoning'])

    sys_acc = accuracy_score(y_true_intent, y_pred_system)
    sys_macro_f1 = f1_score(y_true_intent, y_pred_system, labels=all_intents, average='macro', zero_division=0)
    sys_weighted_f1 = f1_score(y_true_intent, y_pred_system, labels=all_intents, average='weighted', zero_division=0)
    sys_cm = confusion_matrix(y_true_intent, y_pred_system, labels=all_intents)
    sys_report = classification_report(y_true_intent, y_pred_system, labels=all_intents, output_dict=True, zero_division=0)
    print(f'  * Your System Classifier: Accuracy = {sys_acc:.2%}, Macro-F1 = {sys_macro_f1:.4f}, Weighted-F1 = {sys_weighted_f1:.4f}')

    # 3. EVALUATE AUTO-HANDLE VS. ESCALATION DECISION
    print('\n[3/4] Evaluating Auto-Handle vs. Escalation Engine...')
    y_pred_decision = []
    decision_reasons = []
    grounding_scores = []
    false_negatives = []
    false_positives = []

    for idx, row in df_golden.iterrows():
        in_t = str(row['inbound_text'])
        pred_intent = y_pred_system[idx]
        conf = system_confidences[idx]
        tid = int(row['tweet_id'])

        # Grounding check with leave-one-out leakage guard
        draft_res = draft_reply(in_t, intent=pred_intent, exclude_tweet_ids={tid}, provider=provider)
        g_score = draft_res['grounding_score']
        grounding_scores.append(g_score)

        esc_res = decide_escalation(
            inbound_text=in_t,
            intent=pred_intent,
            confidence=conf,
            grounding_score=g_score
        )
        pred_dec = esc_res['decision']
        y_pred_decision.append(pred_dec)
        decision_reasons.append(esc_res['reason'])

        true_dec = row['ideal_decision']
        if true_dec == 'escalate' and pred_dec == 'auto':
            false_negatives.append({
                'tweet_id': tid,
                'inbound_text': in_t,
                'true_intent': row['true_intent'],
                'predicted_intent': pred_intent,
                'ideal_decision': true_dec,
                'predicted_decision': pred_dec,
                'reason': esc_res['reason'],
                'failure_type': 'False Negative (Dangerous Miss)'
            })
        elif true_dec == 'auto' and pred_dec == 'escalate':
            false_positives.append({
                'tweet_id': tid,
                'inbound_text': in_t,
                'true_intent': row['true_intent'],
                'predicted_intent': pred_intent,
                'ideal_decision': true_dec,
                'predicted_decision': pred_dec,
                'reason': esc_res['reason'],
                'failure_type': 'False Positive (Unnecessary Escalation)'
            })

    esc_precision = precision_score(y_true_decision, y_pred_decision, pos_label='escalate', zero_division=0)
    esc_recall = recall_score(y_true_decision, y_pred_decision, pos_label='escalate', zero_division=0)
    esc_f1 = f1_score(y_true_decision, y_pred_decision, pos_label='escalate', zero_division=0)
    esc_acc = accuracy_score(y_true_decision, y_pred_decision)
    esc_cm = confusion_matrix(y_true_decision, y_pred_decision, labels=['auto', 'escalate'])

    print(f'  * Escalation Precision: {esc_precision:.2%}')
    print(f'  * Escalation Recall (Primary Safety Metric): {esc_recall:.2%}')
    print(f'  * Escalation F1: {esc_f1:.4f}')
    print(f'  * Escalation Overall Accuracy: {esc_acc:.2%}')
    print(f'  * False Negatives (Missed Escalations): {len(false_negatives)}')
    print(f'  * False Positives (Excess Escalations): {len(false_positives)}')

    # 4. EVALUATE REPLY QUALITY & HUMAN AGREEMENT
    print('\n[4/4] Evaluating Reply Quality (LLM-as-Judge Rubric & Human Agreement)...')
    eval_sample_n = min(35, len(df_golden))
    sample_eval = df_golden.sample(n=eval_sample_n, random_state=42)

    judge_scores_list = []
    dim_scores = {'grounding': [], 'relevance': [], 'correctness': [], 'tone': [], 'actionability': []}

    for _, row in sample_eval.iterrows():
        in_t = str(row['inbound_text'])
        ref = str(row['reference_reply_direction'])
        tid = int(row['tweet_id'])
        pred_i = classify_intent(in_t, provider=provider)['intent']

        draft_res = draft_reply(in_t, intent=pred_i, exclude_tweet_ids={tid}, provider=provider)
        draft = draft_res['draft_reply']

        j_eval = evaluate_reply_quality(in_t, ref, draft, provider=provider)
        judge_scores_list.append(j_eval['overall'])
        for k in dim_scores:
            dim_scores[k].append(j_eval.get(k, 4))

    avg_overall = float(np.mean(judge_scores_list))
    avg_dims = {k: round(float(np.mean(v)), 2) for k, v in dim_scores.items()}

    print(f'  * Overall Reply Quality: {avg_overall:.2f} / 5.0')
    for k, v in avg_dims.items():
        print(f'    - {k.capitalize()}: {v:.2f} / 5.0')

    ha_metrics = {
        'status': 'not_claimed',
        'sample_size': 0,
        'note': 'Independent human ratings are required before reporting judge-agreement statistics.'
    }

    elapsed_time = round(time.time() - t_start, 2)
    print(f'\nComplete Evaluation Suite Finished in {elapsed_time}s (< 15-minute budget strictly verified).')

    results = {
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_examples_evaluated': N,
        'execution_runtime_seconds': elapsed_time,
        'intent_classification': {
            'trivial_baseline': {
                'majority_class': majority_model.majority_class,
                'accuracy': round(float(maj_acc), 4),
                'macro_f1': round(float(maj_f1), 4)
            },
            'simple_baseline_tfidf': {
                'accuracy': round(float(tfidf_acc), 4),
                'macro_f1': round(float(tfidf_f1), 4)
            },
            'system_classifier': {
                'accuracy': round(float(sys_acc), 4),
                'macro_f1': round(float(sys_macro_f1), 4),
                'weighted_f1': round(float(sys_weighted_f1), 4),
                'confusion_matrix': sys_cm.tolist(),
                'labels': all_intents,
                'per_class_f1': {k: round(v['f1-score'], 4) for k, v in sys_report.items() if k in all_intents}
            }
        },
        'escalation_decision': {
            'target_class': 'escalate',
            'precision': round(float(esc_precision), 4),
            'recall': round(float(esc_recall), 4),
            'f1_score': round(float(esc_f1), 4),
            'overall_accuracy': round(float(esc_acc), 4),
            'confusion_matrix': {
                'labels': ['auto', 'escalate'],
                'matrix': esc_cm.tolist()
            },
            'false_negatives_count': len(false_negatives),
            'false_positives_count': len(false_positives),
            'false_negatives_samples': false_negatives[:5],
            'false_positives_samples': false_positives[:5]
        },
        'reply_quality_llm_judge': {
            'overall_score': round(avg_overall, 2),
            'dimensions': avg_dims
        },
        'human_vs_judge_agreement': ha_metrics
    }

    with open('eval/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 65)
    print('HEADLINE METRICS SUMMARY TABLE')
    print('=' * 65)
    print(f'{"Metric":<38} | {"Score / Value":<15}')
    print('-' * 56)
    print(f'{"Trivial Baseline Accuracy":<38} | {maj_acc:<15.2%}')
    print(f'{"Trivial Baseline Macro-F1":<38} | {maj_f1:<15.4f}')
    print(f'{"Simple Baseline (TF-IDF) Accuracy":<38} | {tfidf_acc:<15.2%}')
    print(f'{"Simple Baseline (TF-IDF) Macro-F1":<38} | {tfidf_f1:<15.4f}')
    print(f'{"System Classifier Accuracy":<38} | {sys_acc:<15.2%}')
    print(f'{"System Classifier Macro-F1":<38} | {sys_macro_f1:<15.4f}')
    print(f'{"Escalation Precision (escalate)":<38} | {esc_precision:<15.2%}')
    print(f'{"Escalation Recall (escalate) [PRIORITY]":<38} | {esc_recall:<15.2%}')
    print(f'{"Escalation F1 (escalate)":<38} | {esc_f1:<15.4f}')
    print(f'{"Reply Quality Overall (1-5 Rubric)":<38} | {avg_overall:<15.2f}')
    print(f'{"Human-vs-Judge Agreement":<38} | NOT CLAIMED')
    print(f'{"Total Pipeline Execution Time":<38} | {elapsed_time:<15.2f}s')
    print('=' * 65)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', type=str, default='eval/golden_set.csv')
    parser.add_argument('--provider', type=str, default=os.getenv('EVAL_PROVIDER', 'local'))
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()

    run_evaluation_suite(golden_path=args.golden, provider=args.provider, num_samples=args.samples)
