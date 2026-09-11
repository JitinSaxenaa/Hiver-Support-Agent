import re
from typing import Dict, Any, List

HIGH_RISK_INTENTS = {
    'billing_subscription_store',
    'account_appleid_icloud'
}

HIGH_RISK_PATTERNS = [
    r'\b(sue|suing|lawyer|attorney|legal|lawsuit)\b',
    r'\b(fraud|fraudulent|stolen|scam|scammed|hacked|theft)\b',
    r'\b(chargeback|dispute|unauthorized)\b',
    r'\b(furious|disgusted|unacceptable|horrible service|worst service)\b',
    r'\b(cancel my service|cancel my contract|switching to android)\b'
    ,r'\b(personal data|credit card|card number|password|verification code|security code)\b'
    ,r'\b(unsafe|injured|injury|fire|smoke|explod|emergency|self[- ]harm)\b'
]

SENSITIVE_INTENT_TERMS = {
    'account_appleid_icloud',
    'billing_subscription_store',
}

def detect_multiple_issue_domains(text: str) -> bool:
    domains = [
        ('battery', 'charging', 'drain', 'overheat'),
        ('ios', 'update', 'crash', 'restart'),
        ('refund', 'charged', 'subscription', 'billing'),
        ('apple id', 'icloud', 'password', 'verification'),
        ('wifi', 'cellular', 'network', 'signal'),
        ('airpods', 'speaker', 'screen', 'camera'),
    ]
    matched = sum(any(term in text.lower() for term in group) for group in domains)
    return matched >= 2

def detect_anger_and_shouting(text: str) -> bool:
    if not text or len(text.strip()) < 15:
        return False
    cleaned = re.sub(r'@\w+|https?://\S+', '', text).strip()
    words = cleaned.split()
    if len(words) < 3:
        return False
    caps_words = [w for w in words if w.isupper() and len(w) > 2 and w not in {'IOS', 'APPLE', 'IPHONE', 'IPAD', 'MAC'}]
    if len(caps_words) >= 3 or (len(caps_words) / len(words)) >= 0.35:
        return True
    return False

def decide_escalation(
    inbound_text: str,
    intent: str,
    confidence: float = 0.85,
    grounding_score: float = 0.75,
    confidence_threshold: float = 0.70,
    grounding_threshold: float = 0.60
) -> Dict[str, Any]:
    triggers = []
    
    # Factor 1: Intent Risk Tier
    if intent in HIGH_RISK_INTENTS:
        if intent == 'billing_subscription_store':
            triggers.append('High-risk domain: financial billing dispute / refund requires human account access')
        elif intent == 'account_appleid_icloud':
            triggers.append('High-risk domain: Apple ID / iCloud security and credential verification')

    # Factor 2: Lexical Risk and Urgency Signals
    t_lower = inbound_text.lower()
    for pattern in HIGH_RISK_PATTERNS:
        match = re.search(pattern, t_lower)
        if match:
            triggers.append(f'Critical sentiment/legal trigger: \"{match.group(0)}\"')
            break

    # Factor 3: Anger / Shouting Signal
    if detect_anger_and_shouting(inbound_text):
        triggers.append('Elevated customer aggression detected (excessive uppercase/shouting)')

    if intent in SENSITIVE_INTENT_TERMS:
        triggers.append('Sensitive account or payment workflow requires authenticated human handling')

    if detect_multiple_issue_domains(inbound_text):
        triggers.append('Multiple issue domains detected; ambiguous compound request requires human review')

    # Factor 4: Intent Classification Uncertainty
    if confidence < confidence_threshold:
        triggers.append(f'Low intent classification confidence ({confidence:.2f} < {confidence_threshold:.2f})')

    # Factor 5: Weak Historical Grounding Support
    if grounding_score < grounding_threshold:
        triggers.append(f'Insufficient historical grounding ({grounding_score:.2f} < {grounding_threshold:.2f})')

    # Final Decision Synthesis
    if triggers:
        decision = 'escalate'
        reason = 'Escalation triggered by: ' + '; '.join(triggers)
    else:
        decision = 'auto'
        reason = f'Approved for auto-handling: routine technical inquiry with high confidence ({confidence:.2f}) and verified grounding ({grounding_score:.2f})'

    return {
        'decision': decision,
        'reason': reason,
        'triggers': triggers
    }

if __name__ == '__main__':
    test_cases = [
        {
            'text': '@AppleSupport my battery drains quickly on iOS 11',
            'intent': 'battery_power_charging',
            'conf': 0.95,
            'grounding': 0.85
        },
        {
            'text': '@AppleSupport YOU BILLED ME TWICE FOR APPLECARE THIS IS FRAUD I AM CALLING MY LAWYER',
            'intent': 'billing_subscription_store',
            'conf': 0.98,
            'grounding': 0.75
        },
        {
            'text': '@AppleSupport cannot sign in',
            'intent': 'other',
            'conf': 0.52,
            'grounding': 0.45
        }
    ]
    for tc in test_cases:
        t_text = tc['text']
        res = decide_escalation(t_text, tc['intent'], tc['conf'], tc['grounding'])
        dec = res['decision'].upper()
        rea = res['reason']
        print(f'Text: {t_text[:45]}...')
        print(f'  Decision: {dec}')
        print(f'  Reason: {rea}\n')
