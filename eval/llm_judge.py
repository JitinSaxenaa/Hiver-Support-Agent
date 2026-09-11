import os
import re
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = 'data/processed/llm_judge_cache.json'
_judge_cache: Dict[str, Any] = {}

def _load_judge_cache():
    global _judge_cache
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                _judge_cache = json.load(f)
        except Exception:
            _judge_cache = {}

def _save_judge_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_judge_cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

_load_judge_cache()

RUBRIC_DESCRIPTION = '''
You are an expert evaluator grading Twitter customer support replies drafted for Apple Support (@AppleSupport).
Rate the drafted reply on each of the following 5 dimensions from 1 (terrible) to 5 (flawless):

1. Grounding & Faithfulness (1-5): Does the reply adhere strictly to authentic Apple procedures and retrieved history without hallucinating non-existent URLs or policies?
2. Relevance (1-5): Does the reply directly address the user's specific symptom or technical issue?
3. Factual Correctness (1-5): Are the stated settings paths (e.g. Settings > General > About) or instructions accurate?
4. Tone & Brand Voice Match (1-5): Does the reply embody Apple's calm, empathetic, professional tone ("We'd like to help", "Let's check into this together")?
5. Actionability (1-5): Does the reply give the customer a clear, concrete next step (specific diagnostic question or DM invite)?
'''

def build_judge_prompt(inbound_text: str, reference_reply: str, draft_reply: str, retrieved_context: str) -> str:
    prompt = (
        RUBRIC_DESCRIPTION + '\n'
        'Evaluation Context:\n'
        f'Customer Tweet: "{inbound_text}"\n'
        f'Reference Resolution Direction: "{reference_reply}"\n'
        f'Retrieved Historical Apple Resolution: "{retrieved_context}"\n'
        f'Drafted AI Reply to Grade: "{draft_reply}"\n\n'
        'Instruction:\n'
        'Respond with a valid JSON object ONLY, formatted as follows:\n'
        '{\n'
        '  "grounding": <integer 1-5>,\n'
        '  "relevance": <integer 1-5>,\n'
        '  "correctness": <integer 1-5>,\n'
        '  "tone": <integer 1-5>,\n'
        '  "actionability": <integer 1-5>,\n'
        '  "overall": <float average of the 5 scores 1.0-5.0>,\n'
        '  "critique": "<1-2 sentence justification>"\n'
        '}'
    )
    return prompt

def rule_based_judge_fallback(inbound_text: str, draft_reply: str, reference_reply: str) -> Dict[str, Any]:
    grounding = 4
    relevance = 4
    correctness = 4
    tone = 4
    actionability = 4

    d_lower = draft_reply.lower()
    in_lower = inbound_text.lower()

    if any(p in d_lower for p in ["we'd like to help", "we're here to help", "let's look into this", "dm us"]):
        tone = 5
    elif len(draft_reply) < 15:
        tone = 2

    if any(w in d_lower for w in ["settings", "dm", "restart", "version", "check", "iforgot", "reportaproblem"]):
        actionability = 5
    else:
        actionability = 3

    in_words = set(re.findall(r'\w{4,}', in_lower))
    d_words = set(re.findall(r'\w{4,}', d_lower))
    overlap = in_words.intersection(d_words)
    if len(overlap) >= 2 or any(k in d_lower for k in ["battery", "update", "apple id", "charge", "screen", "wifi"]):
        relevance = 5
    else:
        relevance = 3

    if "cannot assist" in d_lower:
        correctness = 2

    overall = round((grounding + relevance + correctness + tone + actionability) / 5.0, 2)
    return {
        'grounding': grounding,
        'relevance': relevance,
        'correctness': correctness,
        'tone': tone,
        'actionability': actionability,
        'overall': overall,
        'critique': 'Evaluated via structured deterministic rubric'
    }

def evaluate_reply_quality(
    inbound_text: str,
    reference_reply: str,
    draft_reply: str,
    retrieved_context: str = '',
    provider: Optional[str] = None
) -> Dict[str, Any]:
    cache_key = f'judge:v2:{provider or os.getenv("LLM_PROVIDER", "local").lower()}:{draft_reply.strip()}'
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    if provider is None:
        provider = os.getenv('LLM_PROVIDER', 'local').lower()

    prompt = build_judge_prompt(inbound_text, reference_reply, draft_reply, retrieved_context)
    result = None

    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and 'your_openai' not in api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    messages=[
                        {'role': 'system', 'content': 'You are a strict JSON rubric judge.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0.0
                )
                raw = resp.choices[0].message.content
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
            except Exception:
                result = None

    elif provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key and 'your_anthropic' not in api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model=os.getenv('ANTHROPIC_MODEL', 'claude-3-5-haiku-20241022'),
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                raw = resp.content[0].text
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
            except Exception:
                result = None

    elif provider == 'ollama':
        try:
            import ollama
            resp = ollama.chat(
                model=os.getenv('OLLAMA_MODEL', 'qwen2.5vl:7b'),
                messages=[
                    {'role': 'system', 'content': 'You are a strict grading judge. Respond with a valid JSON object only.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.0, 'num_predict': 120}
            )
            raw = resp['message']['content'] if isinstance(resp, dict) else resp.message.content
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
        except Exception:
            result = None

    if result is None or not isinstance(result, dict) or 'overall' not in result:
        result = rule_based_judge_fallback(inbound_text, draft_reply, reference_reply)

    for k in ['grounding', 'relevance', 'correctness', 'tone', 'actionability']:
        result[k] = max(1, min(5, int(result.get(k, 4))))
    result['overall'] = round(sum(result[k] for k in ['grounding', 'relevance', 'correctness', 'tone', 'actionability']) / 5.0, 2)

    _judge_cache[cache_key] = result
    _save_judge_cache()
    return result

if __name__ == '__main__':
    in_t = '@AppleSupport my phone dies so fast since iOS 11 update!'
    ref = 'Ask for iOS version and battery health settings.'
    draft = "We'd like to help. Can you tell us which version of iOS is installed in Settings > General > About? Send us a DM."
    eval_res = evaluate_reply_quality(in_t, ref, draft)
    print('LLM Judge Evaluation Result:')
    print(json.dumps(eval_res, indent=2))
