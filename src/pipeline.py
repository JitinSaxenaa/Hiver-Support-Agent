import os
import sys
import json
import argparse
from typing import Dict, Any, Optional, Set
from dotenv import load_dotenv

load_dotenv()

from src.classify import classify_intent
from src.draft_reply import draft_reply
from src.escalate import decide_escalation

class TwitterSupportPipeline:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or os.getenv('LLM_PROVIDER', 'ollama')
        self.model = model

    def process_tweet(
        self,
        inbound_text: str,
        tweet_id: Optional[int] = None,
        exclude_tweet_ids: Optional[Set[int]] = None
    ) -> Dict[str, Any]:
        ex_set = set(exclude_tweet_ids) if exclude_tweet_ids else set()
        if tweet_id is not None:
            ex_set.add(tweet_id)

        clf_result = classify_intent(inbound_text, provider=self.provider, model=self.model)
        intent = clf_result['intent']
        confidence = clf_result['confidence']
        reasoning = clf_result['reasoning']

        draft_result = draft_reply(
            inbound_text=inbound_text,
            intent=intent,
            exclude_tweet_ids=ex_set,
            provider=self.provider,
            model=self.model
        )
        draft_text = draft_result['draft_reply']
        grounding_score = draft_result['grounding_score']
        retrieved_cases = draft_result['retrieved_cases']

        escalation = decide_escalation(
            inbound_text=inbound_text,
            intent=intent,
            confidence=confidence,
            grounding_score=grounding_score
        )

        return {
            'inbound_text': inbound_text,
            'tweet_id': tweet_id,
            'intent': intent,
            'confidence': confidence,
            'classification_reasoning': reasoning,
            'grounding_score': grounding_score,
            'retrieved_cases': retrieved_cases,
            'draft_reply': draft_text,
            'escalation': escalation
        }

def run_pipeline():
    parser = argparse.ArgumentParser(description='Run Hiver Twitter Support Agent Pipeline on a customer tweet')
    parser.add_argument('tweet', nargs='?', type=str, default='@AppleSupport updated to iOS 11 and now my battery dies in 2 hours!')
    parser.add_argument('--provider', type=str, default=None)
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
    args = parser.parse_args()

    pipeline = TwitterSupportPipeline(provider=args.provider)
    result = pipeline.process_tweet(args.tweet)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        in_t = result['inbound_text']
        intent = result['intent']
        conf = result['confidence']
        g_score = result['grounding_score']
        dec = result['escalation']['decision'].upper()
        reason = result['escalation']['reason']
        draft = result['draft_reply']
        n_cases = len(result['retrieved_cases'])
        
        print('\n' + '=' * 60)
        print('HIVER TWITTER SUPPORT AGENT - PIPELINE EXECUTION')
        print('=' * 60)
        print(f'Customer Tweet: {in_t}')
        print(f'Predicted Intent: {intent} (Confidence: {conf:.2f})')
        print(f'Grounding Score: {g_score:.4f} ({n_cases} historical cases retrieved)')
        print(f'Escalation Decision: [{dec}]')
        print(f'Reason: {reason}')
        print('-' * 60)
        print(f'Draft Reply:\n  "{draft}"')
        print('=' * 60 + '\n')

if __name__ == '__main__':
    run_pipeline()
