import pandas as pd
import numpy as np
import re
import json
import os

# Set seed
SEED = 42
np.random.seed(SEED)

print('Loading grounding corpus...')
df = pd.read_csv('data/processed/grounding_corpus.csv')

# Ensure strict temporal/stratified split
# 20,000 for retrieval corpus index
# 4,800 for training/silver baseline models
# 200 strictly held-out for golden evaluation set
N_TOTAL = len(df)
print(f'Total available grounding corpus rows: {N_TOTAL:,}')

# Reserve the last 2,000 rows as evaluation pool
eval_pool = df.iloc[-2500:].copy().reset_index(drop=True)
train_pool = df.iloc[20000:-2500].copy().reset_index(drop=True)
index_pool = df.iloc[:20000].copy().reset_index(drop=True)

# Save the index pool so retrieve.py strictly indexes ONLY this portion
os.makedirs('data/processed', exist_ok=True)
index_pool.to_csv('data/processed/retrieval_index_corpus.csv', index=False)
print(f'Saved {len(index_pool):,} rows to data/processed/retrieval_index_corpus.csv (Strictly held-out from eval)')

# Regex pattern rules to find candidate exemplars for each intent from eval_pool
rules = {
    'software_update_os': r'\b(ios|update|updated|updating|11\.|11\.0|11\.1|11\.2|crash|crashes|freeze|freezing|glitch|reboot|boot|apple logo|black screen|firmware|itunes error)\b',
    'battery_power_charging': r'\b(battery|drain|drains|draining|dying|charge|charging|charger|percentage|shut down|overheat|hot|cable|lightning)\b',
    'account_appleid_icloud': r'\b(apple id|appleid|icloud|password|passcode|locked|disabled|security question|two factor|2fa|verification code|sign in|login|backup)\b',
    'billing_subscription_store': r'\b(charged|charge|refund|subscription|cancel|receipt|itunes bill|app store bill|purchase|renew|apple music charge|applecare|overcharged|cents|dollars|\$)\b',
    'hardware_display_audio': r'\b(airpods|earphones|headphone|speaker|screen|crack|cracked|broken|sound|audio|microphone|mic|bluetooth|touch|display|lines|sensor)\b',
    'network_connectivity': r'\b(wifi|wi-fi|cellular|lte|4g|service|no service|signal|connecting|disconnect|carrier|sim|network)\b',
    'general_complaint_store': r'\b(genius bar|appointment|store|customer service|worst|unacceptable|useless|terrible|disappointed|manager|staff|rude|hours|waited)\b'
}

golden_rows = []
used_ids = set()

# Targeted stratified collection: 22-26 examples per category, targeting 200 total
target_per_intent = {
    'software_update_os': 25,
    'battery_power_charging': 25,
    'account_appleid_icloud': 25,
    'billing_subscription_store': 25,
    'hardware_display_audio': 25,
    'network_connectivity': 25,
    'general_complaint_store': 25,
    'other': 25
}

# 1. Match candidates for specific intents
for intent, pattern in rules.items():
    matches = eval_pool[eval_pool['inbound_text'].str.contains(pattern, case=False, na=False) & (~eval_pool['inbound_tweet_id'].isin(used_ids))]
    sample_n = min(target_per_intent[intent], len(matches))
    selected = matches.sample(n=sample_n, random_state=SEED)
    for _, row in selected.iterrows():
        used_ids.add(row['inbound_tweet_id'])
        golden_rows.append({
            'inbound_tweet_id': row['inbound_tweet_id'],
            'inbound_text': row['inbound_text'],
            'candidate_intent': intent
        })

# 2. Match candidate 'other' examples (short acknowledgments, vague questions, or unmatched)
other_candidates = eval_pool[~eval_pool['inbound_tweet_id'].isin(used_ids)]
other_sample = other_candidates.sample(n=target_per_intent['other'], random_state=SEED)
for _, row in other_sample.iterrows():
    used_ids.add(row['inbound_tweet_id'])
    golden_rows.append({
        'inbound_tweet_id': row['inbound_tweet_id'],
        'inbound_text': row['inbound_text'],
        'candidate_intent': 'other'
    })

print(f'Collected {len(golden_rows)} candidate rows across all 8 intents.')

# Detailed annotation logic: review each text and assign true_intent, ideal_decision, escalation_reason, reference_reply_direction
labeled_data = []

for item in golden_rows:
    text = item['inbound_text']
    tid = item['inbound_tweet_id']
    t_lower = text.lower()
    
    # 1. Determine True Intent
    if any(w in t_lower for w in ['charged', 'refund', 'subscription', 'purchase', 'receipt', 'bill', 'applecare', '$', 'dollars']):
        intent = 'billing_subscription_store'
    elif any(w in t_lower for w in ['apple id', 'appleid', 'icloud', 'password', 'passcode', 'verification code', '2fa', 'two factor', 'locked out']):
        intent = 'account_appleid_icloud'
    elif any(w in t_lower for w in ['battery', 'drain', 'percentage', 'overheat', 'charging', 'charger', 'shut down']):
        intent = 'battery_power_charging'
    elif any(w in t_lower for w in ['airpod', 'bluetooth', 'speaker', 'audio', 'microphone', 'mic', 'cracked', 'screen unresponsive']):
        intent = 'hardware_display_audio'
    elif any(w in t_lower for w in ['no service', 'wifi', 'wi-fi', 'cellular', 'signal', 'lte', 'carrier', 'sim']):
        intent = 'network_connectivity'
    elif any(w in t_lower for w in ['ios', 'update', 'updating', '11.', 'crash', 'glitch', 'restart', 'reboot']):
        intent = 'software_update_os'
    elif any(w in t_lower for w in ['worst', 'genius bar', 'rude', 'terrible', 'appointment', 'unacceptable', 'disappointed', 'manager']):
        intent = 'general_complaint_store'
    elif len(text.split()) <= 4 or any(w in t_lower for w in ['thanks', 'thank you', 'okay', 'great', 'http', 'dm sent']):
        intent = 'other'
    else:
        intent = item['candidate_intent']
        
    # 2. Determine Ideal Escalation Decision and Reason
    # High risk triggers: billing dispute, account credentials/locked, legal/threats, extreme anger, hardware damage needing physical repair
    is_escalate = False
    reasons = []
    
    if intent == 'billing_subscription_store':
        is_escalate = True
        reasons.append('Financial charge/refund requires private account DM verification')
    elif intent == 'account_appleid_icloud':
        is_escalate = True
        reasons.append('Account access/Apple ID security credentials require secure DM transfer')
    elif any(w in t_lower for w in ['sue', 'lawyer', 'legal', 'fraud', 'stolen', 'worst', 'furious', 'scam', 'ridiculous']):
        is_escalate = True
        reasons.append('High negative sentiment or legal/fraud escalation risk')
    elif any(w in t_lower for w in ['cracked', 'broken', 'water damage', 'hardware repair', 'store appointment']):
        is_escalate = True
        reasons.append('Physical hardware damage requires in-person Genius Bar evaluation')
    elif intent == 'other':
        # Vague messages or acknowledgments don't need tier-2 human escalation, can be auto-handled with clarifying question or closed
        is_escalate = False
        reasons.append('Routine conversational closure or clarifying inquiry')
    else:
        # Technical troubleshooting: standard OS, battery, network diagnostics can be auto-handled
        is_escalate = False
        reasons.append('Standard self-service troubleshooting workflow available')
        
    decision = 'escalate' if is_escalate else 'auto'
    escalation_reason = '; '.join(reasons)
    
    # 3. Reference Reply Direction (key diagnostic elements)
    if intent == 'software_update_os':
        ref_reply = 'Acknowledge issue, inquire exact iOS version and device model, suggest hard restart or update to latest point release.'
    elif intent == 'battery_power_charging':
        ref_reply = 'Ask for device model and iOS version; guide user to Settings > Battery to check battery health and app usage breakdown.'
    elif intent == 'account_appleid_icloud':
        ref_reply = 'Direct customer to iforgot.apple.com; invite them to DM AppleSupport with their Apple ID email for identity verification.'
    elif intent == 'billing_subscription_store':
        ref_reply = 'Direct user to reportaproblem.apple.com to review purchase history and request refund; offer DM for billing assistance.'
    elif intent == 'hardware_display_audio':
        ref_reply = 'Provide basic reset instructions (e.g. unpair/re-pair AirPods, clean port); offer to schedule Genius Bar reservation if unresolved.'
    elif intent == 'network_connectivity':
        ref_reply = 'Recommend toggling Airplane Mode, restarting phone, or resetting Network Settings (Settings > General > Reset).'
    elif intent == 'general_complaint_store':
        ref_reply = 'Express empathy for frustrating experience; offer a direct DM link to collect details and escalate to store management.'
    else:
        ref_reply = 'Politely acknowledge customer message; invite them to provide more details or confirm if further assistance is needed.'
        
    labeled_data.append({
        'tweet_id': tid,
        'inbound_text': text,
        'true_intent': intent,
        'ideal_decision': decision,
        'escalation_reason': escalation_reason,
        'reference_reply_direction': ref_reply
    })

df_golden = pd.DataFrame(labeled_data)
# Exact 200 rows
if len(df_golden) > 200:
    df_golden = df_golden.iloc[:200]
    
out_golden_path = 'eval/golden_set.csv'
os.makedirs('eval', exist_ok=True)
df_golden.to_csv(out_golden_path, index=False)
print(f'Successfully wrote {len(df_golden)} golden evaluation examples to {out_golden_path}.')

# Distribution report
print('\nGolden Set Intent Distribution:')
print(df_golden['true_intent'].value_counts())
print('\nGolden Set Decision Distribution:')
print(df_golden['ideal_decision'].value_counts())
