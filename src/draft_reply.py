import os
import json
import re
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv
from src.retrieve import retrieve_similar_threads

load_dotenv()

CACHE_FILE = 'data/processed/llm_draft_reply_cache.json'
_cache: Dict[str, str] = {}

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
        pass

_load_cache()

def sanitize_historical_reply(text: str) -> str:
    cleaned = re.sub(r'^(?:@\w+\s*)+', '', str(text)).strip()
    cleaned = re.sub(r'https?://\S+', '', cleaned).strip()
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned

def build_rag_prompt(inbound_text: str, intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
    prompt = (
        'You are an AI support agent drafting replies on Twitter for Apple Support (@AppleSupport).\n'
        'Your goal is to compose an authentic, polite, and actionable tweet reply that matches Apple\'s authentic tone.\n'
        'Strictly follow the brand voice and concrete resolution procedures seen in the historical resolved examples below.\n\n'
        f'Customer Inbound Tweet:\n\"{inbound_text}\"\n'
        f'Predicted Intent: {intent}\n\n'
        'Historical Similar Resolved Cases from Apple Support Archives:\n'
    )
    for i, case in enumerate(retrieved_cases, 1):
        sim = case.get('similarity_score', 0.0)
        in_t = case.get('inbound_text', '')
        br_t = sanitize_historical_reply(case.get('brand_reply_text', ''))
        prompt += (
            f'Case {i} (Similarity: {sim:.2f}):\n'
            f'  Customer: \"{in_t}\"\n'
            f'  Official Brand Reply: \"{br_t}\"\n\n'
        )

    prompt += (
        'Guidelines:\n'
        '1. Keep your reply concise (within 280 characters if possible), friendly, and empathetic.\n'
        '2. Use Apple Support\'s signature style: e.g. \"We\'d like to help\", \"Let\'s look into this together\", or \"Send us a DM\".\n'
        '3. Ask for specific diagnostics shown in retrieved cases (e.g. iOS version, device model, or error message) rather than generic platitudes.\n'
        '4. Do not invent non-existent URLs or policies. If recommending a DM, say \"Send us a DM with more details so we can assist.\"\n'
        '5. If the retrieved historical cases are low confidence or do not address the issue, politely acknowledge the issue and ask for clarification.\n\n'
        'Reply directly with the drafted tweet text only. Do not include quotes, hashtags, or meta commentary.'
    )
    return prompt

def heuristic_grounded_draft(inbound_text: str, intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
    if retrieved_cases and retrieved_cases[0].get('similarity_score', 0) >= 0.70:
        top_reply = retrieved_cases[0]['brand_reply_text']
        cleaned_reply = sanitize_historical_reply(top_reply)
        if len(cleaned_reply) > 20:
            return f"We're here to help. {cleaned_reply}"

    if intent == 'battery_power_charging':
        return "We'd like to help get this sorted out. Which device and exact iOS version are you on? Take a look at Settings > Battery to check battery health and app usage."
    elif intent == 'software_update_os':
        return "We want to help ensure your device runs smoothly. Could you let us know your exact device model and iOS version in Settings > General > About?"
    elif intent == 'account_appleid_icloud':
        return "Account security is our top priority. Please visit iforgot.apple.com to reset your credentials, or send us a DM so we can verify your account securely."
    elif intent == 'billing_subscription_store':
        return "We understand unexpected charges are concerning. You can review your purchase history and request refunds at reportaproblem.apple.com, or DM us for assistance."
    elif intent == 'hardware_display_audio':
        return "We'd love to help with your device. Have you tried restarting the device, and does this issue persist across all apps? Send us a DM if you need a Genius Bar visit."
    elif intent == 'network_connectivity':
        return "Let's help get you connected. Have you tried toggling Airplane Mode on and off, or resetting network settings in Settings > General > Reset?"
    elif intent == 'general_complaint_store':
        return "We're sorry to hear about your experience and appreciate you letting us know. Please send us a DM with your appointment details so we can look into this."
    else:
        return "We're here to help! Could you provide a bit more detail about what you're experiencing? You can also send us a DM to troubleshoot together."

def draft_reply(
    inbound_text: str,
    intent: str,
    exclude_tweet_ids: Optional[Set[int]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    retrieved = retrieve_similar_threads(inbound_text, top_k=3, exclude_tweet_ids=exclude_tweet_ids)
    grounding_score = retrieved[0]['similarity_score'] if retrieved else 0.0

    cache_key = f'draft:v2:{provider or os.getenv("LLM_PROVIDER", "local").lower()}:{model or ""}:{inbound_text.strip()}'
    if cache_key in _cache:
        return {
            'draft_reply': _cache[cache_key],
            'retrieved_cases': retrieved,
            'grounding_score': grounding_score,
            'grounding_sufficient': grounding_score >= 0.65
        }

    if provider is None:
        provider = os.getenv('LLM_PROVIDER', 'local').lower()

    prompt = build_rag_prompt(inbound_text, intent, retrieved)
    draft_text = None

    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or 'your_openai' in api_key:
            draft_text = heuristic_grounded_draft(inbound_text, intent, retrieved)
        else:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                m = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {'role': 'system', 'content': 'You are a professional Twitter support agent for Apple.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                draft_text = resp.choices[0].message.content.strip().strip('"')
            except Exception as e:
                draft_text = heuristic_grounded_draft(inbound_text, intent, retrieved)

    elif provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key or 'your_anthropic' in api_key:
            draft_text = heuristic_grounded_draft(inbound_text, intent, retrieved)
        else:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                m = model or os.getenv('ANTHROPIC_MODEL', 'claude-3-5-haiku-20241022')
                resp = client.messages.create(
                    model=m,
                    max_tokens=150,
                    temperature=0.3,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                draft_text = resp.content[0].text.strip().strip('"')
            except Exception as e:
                draft_text = heuristic_grounded_draft(inbound_text, intent, retrieved)

    elif provider == 'ollama':
        try:
            import ollama
            m = model or os.getenv('OLLAMA_MODEL', 'qwen2.5vl:7b')
            resp = ollama.chat(
                model=m,
                messages=[
                    {'role': 'system', 'content': 'You are an Apple Support agent. Reply with the draft tweet only.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.3, 'num_predict': 100}
            )
            raw = resp['message']['content'] if isinstance(resp, dict) else resp.message.content
            draft_text = raw.strip().strip('"')
        except Exception as e:
            draft_text = heuristic_grounded_draft(inbound_text, intent, retrieved)
    else:
        draft_text = heuristic_grounded_draft(inbound_text, intent, retrieved)

    if not draft_text:
        draft_text = heuristic_grounded_draft(inbound_text, intent, retrieved)

    draft_text = re.sub(r'^["\']|["\']$', '', draft_text.strip())

    _cache[cache_key] = draft_text
    _save_cache()

    return {
        'draft_reply': draft_text,
        'retrieved_cases': retrieved,
        'grounding_score': grounding_score,
        'grounding_sufficient': grounding_score >= 0.65
    }

if __name__ == '__main__':
    sample_tweet = '@AppleSupport my battery drops from 80% to 10% in 15 minutes on my iPhone 7!'
    result = draft_reply(sample_tweet, 'battery_power_charging')
    g_score = result['grounding_score']
    d_reply = result['draft_reply']
    print('Draft Reply Generation Test:')
    print('Inbound:', sample_tweet)
    print('Grounding Score:', g_score)
    print('Draft Reply:\n ', d_reply)
