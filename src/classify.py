import os
import json
import re
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from src.taxonomy import Intent, get_all_intents, get_intent_definitions, get_intent_exemplars, is_valid_intent

load_dotenv()

CACHE_FILE = 'data/processed/llm_classification_cache.json'
_cache: Dict[str, Any] = {}

def _load_cache():
    global _cache
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}

def _save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f'Warning: Could not save cache: {e}')

_load_cache()

def build_classification_prompt(text: str) -> str:
    defs = get_intent_definitions()
    exs = get_intent_exemplars()
    
    prompt = (
        'You are an expert intent classification engine for Apple Support Twitter inquiries.\n'
        'Analyze the customer tweet and classify it into EXACTLY ONE of the following 8 intent categories:\n\n'
    )
    for intent_name, definition in defs.items():
        sample_ex = exs.get(intent_name, [''])[0]
        prompt += f'- "{intent_name}": {definition}\n  Example: "{sample_ex}"\n'
        
    prompt += (
        '\nInstruction:\n'
        'Output a valid JSON object ONLY, with no extra commentary or markdown formatting, containing:\n'
        '{\n'
        '  "intent": "<one of the 8 intent names exactly>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reasoning": "<short 1-sentence justification>"\n'
        '}\n\n'
        f'Customer Tweet to classify:\n"{text}"\n'
    )
    return prompt

def extract_and_repair_json(raw_response: str) -> Dict[str, Any]:
    if not isinstance(raw_response, str):
        return {'intent': 'other', 'confidence': 0.5, 'reasoning': 'Invalid non-string response'}
    
    # Strip markdown code fences if present
    cleaned = raw_response.strip()
    if cleaned.startswith('`'):
        cleaned = re.sub(r'^`[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?`$', '', cleaned)
        cleaned = cleaned.strip()
        
    # Attempt direct json parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and 'intent' in data:
            intent = str(data['intent']).strip().lower()
            if is_valid_intent(intent):
                conf = float(data.get('confidence', 0.8))
                conf = max(0.0, min(1.0, conf))
                return {
                    'intent': intent,
                    'confidence': round(conf, 3),
                    'reasoning': str(data.get('reasoning', 'Extracted from LLM response'))
                }
    except Exception:
        pass
        
    # Regex fallback to find JSON block
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            intent = str(data.get('intent', '')).strip().lower()
            if is_valid_intent(intent):
                conf = float(data.get('confidence', 0.75))
                return {
                    'intent': intent,
                    'confidence': round(max(0.0, min(1.0, conf)), 3),
                    'reasoning': str(data.get('reasoning', 'Regex extracted JSON'))
                }
        except Exception:
            pass
            
    # Substring search for valid intent names
    for valid_i in get_all_intents():
        if valid_i in cleaned.lower():
            return {
                'intent': valid_i,
                'confidence': 0.65,
                'reasoning': f'Recovered intent keyword "{valid_i}" from unstructured output'
            }
            
    return {
        'intent': 'other',
        'confidence': 0.4,
        'reasoning': 'Failed to parse structured intent from response; defaulted to other'
    }

def rule_and_semantic_fallback(text: str) -> Dict[str, Any]:
    t_lower = text.lower()
    if any(w in t_lower for w in ['charged', 'refund', 'subscription', 'purchase', 'receipt', 'bill', 'applecare', '$', 'dollars']):
        return {'intent': 'billing_subscription_store', 'confidence': 0.92, 'reasoning': 'Identified financial charge/billing keywords'}
    if any(w in t_lower for w in ['apple id', 'appleid', 'icloud', 'password', 'passcode', 'verification code', '2fa', 'two factor', 'locked out']):
        return {'intent': 'account_appleid_icloud', 'confidence': 0.91, 'reasoning': 'Identified Apple ID / iCloud authentication keywords'}
    if any(w in t_lower for w in ['battery', 'drain', 'percentage', 'overheat', 'charging', 'charger', 'shut down']):
        return {'intent': 'battery_power_charging', 'confidence': 0.90, 'reasoning': 'Identified battery/power/charging keywords'}
    if any(w in t_lower for w in ['airpod', 'bluetooth', 'speaker', 'audio', 'microphone', 'mic', 'cracked', 'screen unresponsive']):
        return {'intent': 'hardware_display_audio', 'confidence': 0.88, 'reasoning': 'Identified audio/display/hardware keywords'}
    if any(w in t_lower for w in ['no service', 'wifi', 'wi-fi', 'cellular', 'signal', 'lte', 'carrier', 'sim']):
        return {'intent': 'network_connectivity', 'confidence': 0.89, 'reasoning': 'Identified connectivity/carrier keywords'}
    if any(w in t_lower for w in ['ios', 'update', 'updating', '11.', 'crash', 'glitch', 'restart', 'reboot']):
        return {'intent': 'software_update_os', 'confidence': 0.87, 'reasoning': 'Identified OS/system update keywords'}
    if any(w in t_lower for w in ['worst', 'genius bar', 'rude', 'terrible', 'appointment', 'unacceptable', 'disappointed', 'manager']):
        return {'intent': 'general_complaint_store', 'confidence': 0.86, 'reasoning': 'Identified customer store/service complaint keywords'}
    return {'intent': 'other', 'confidence': 0.60, 'reasoning': 'No clear domain keywords matched; classified as other'}

def classify_intent(text: str, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    if not text or not text.strip():
        return {'intent': 'other', 'confidence': 1.0, 'reasoning': 'Empty text'}
        
    cache_key = f'classify:v2:{provider or os.getenv("LLM_PROVIDER", "local").lower()}:{model or ""}:{text.strip()}'
    if cache_key in _cache:
        return _cache[cache_key]
        
    if provider is None:
        provider = os.getenv('LLM_PROVIDER', 'local').lower()
        
    prompt = build_classification_prompt(text)
    result = None
    
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or 'your_openai' in api_key:
            result = rule_and_semantic_fallback(text)
        else:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                m = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {'role': 'system', 'content': 'You are a strict JSON classification engine.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0.0
                )
                raw = resp.choices[0].message.content
                result = extract_and_repair_json(raw)
            except Exception as e:
                print(f'OpenAI error ({e}), falling back...')
                result = rule_and_semantic_fallback(text)

    elif provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key or 'your_anthropic' in api_key:
            result = rule_and_semantic_fallback(text)
        else:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                m = model or os.getenv('ANTHROPIC_MODEL', 'claude-3-5-haiku-20241022')
                resp = client.messages.create(
                    model=m,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                raw = resp.content[0].text
                result = extract_and_repair_json(raw)
            except Exception as e:
                print(f'Anthropic error ({e}), falling back...')
                result = rule_and_semantic_fallback(text)

    elif provider == 'ollama':
        try:
            import ollama
            m = model or os.getenv('OLLAMA_MODEL', 'qwen2.5vl:7b')
            resp = ollama.chat(
                model=m,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful JSON classification engine. Respond with a single valid JSON object only.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.0}
            )
            raw = resp['message']['content'] if isinstance(resp, dict) else resp.message.content
            result = extract_and_repair_json(raw)
        except Exception as e:
            print(f'Ollama error ({e}), falling back...')
            result = rule_and_semantic_fallback(text)
    else:
        result = rule_and_semantic_fallback(text)
        
    if result is None:
        result = rule_and_semantic_fallback(text)
        
    _cache[cache_key] = result
    _save_cache()
    return result

if __name__ == '__main__':
    test_tweets = [
        '@AppleSupport my battery drops from 80% to 10% in 15 minutes on my iPhone 7!',
        '@AppleSupport I was charged .99 for an iCloud subscription I never signed up for.',
        '@AppleSupport Can someone help me unlock my Apple ID? Forgot password and recovery code not sent.'
    ]
    for tw in test_tweets:
        res = classify_intent(tw)
        intent = res['intent']
        conf = res['confidence']
        reason = res['reasoning']
        print(f'Tweet: {tw[:50]}...\n  -> Intent: {intent} (Conf: {conf}) | {reason}\n')
