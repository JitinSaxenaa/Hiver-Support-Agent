from enum import Enum
from typing import Dict, List, Any

class Intent(str, Enum):
    SOFTWARE_UPDATE_OS = 'software_update_os'
    BATTERY_POWER_CHARGING = 'battery_power_charging'
    ACCOUNT_APPLEID_ICLOUD = 'account_appleid_icloud'
    BILLING_SUBSCRIPTION_STORE = 'billing_subscription_store'
    HARDWARE_DISPLAY_AUDIO = 'hardware_display_audio'
    NETWORK_CONNECTIVITY = 'network_connectivity'
    GENERAL_COMPLAINT_STORE = 'general_complaint_store'
    OTHER = 'other'

TAXONOMY_DEFINITIONS: Dict[Intent, str] = {
    Intent.SOFTWARE_UPDATE_OS: (
        'Issues installing, updating, or running iOS/macOS, including OS crashes, freezing, and version-specific software bugs.'
    ),
    Intent.BATTERY_POWER_CHARGING: (
        'Complaints regarding abnormal battery drainage, device overheating, sudden shutdowns, or charging cable/port failures.'
    ),
    Intent.ACCOUNT_APPLEID_ICLOUD: (
        'Problems with Apple ID authentication, forgotten passwords, two-factor verification, and iCloud backup or storage syncing.'
    ),
    Intent.BILLING_SUBSCRIPTION_STORE: (
        'Disputes regarding unexpected App Store charges, recurring in-app subscription billing, refund requests, or AppleCare fees.'
    ),
    Intent.HARDWARE_DISPLAY_AUDIO: (
        'Physical hardware malfunctions, unresponsive touchscreens, camera/speaker faults, AirPods pairing, or Bluetooth dropouts.'
    ),
    Intent.NETWORK_CONNECTIVITY: (
        'Inability to connect to Wi-Fi networks, cellular data failures, persistent \"No Service\" status, or carrier SIM errors.'
    ),
    Intent.GENERAL_COMPLAINT_STORE: (
        'Customer venting, dissatisfaction with Apple Store retail genius appointments, repair delays, or negative brand sentiment.'
    ),
    Intent.OTHER: (
        'Ambiguous snippets, conversational acknowledgments (e.g., \"thanks\", \"ok\"), image-only tweets, or uninterpretable fragments.'
    ),
}

TAXONOMY_EXEMPLARS: Dict[Intent, List[str]] = {
    Intent.SOFTWARE_UPDATE_OS: [
        '@AppleSupport #ios11 is very glitchy and phone usage has become a pain pls fix the OS and release an update soon',
        '@AppleSupport every time I type the letter \"I\" this pops up [?]. How do I fix this keyboard bug?',
        '@AppleSupport Updated to 11.1 and now my phone constantly restarts itself every 10 minutes.'
    ],
    Intent.BATTERY_POWER_CHARGING: [
        '@AppleSupport please fix battery life on ios 11.0.2 iphone 6s. Draining 50% in an hour without use!',
        '@AppleSupport my iPhone 7 battery is dying within 3 hours and the phone gets burning hot when plugged in.',
        '@AppleSupport phone wont charge past 80% with official charger since yesterday.'
    ],
    Intent.ACCOUNT_APPLEID_ICLOUD: [
        '@AppleSupport my Apple ID has been locked for security reasons and the recovery email is not arriving.',
        '@AppleSupport iCloud says my backup failed because storage is full even though I purchased the 50GB plan.',
        '@AppleSupport cannot sign into my iCloud account on my new iPad, keeps asking for verification code.'
    ],
    Intent.BILLING_SUBSCRIPTION_STORE: [
        '@AppleSupport I was charged .99 twice for Apple Music subscription this month. I need a refund.',
        '@AppleSupport how do I cancel a subscription I was charged for through iTunes on my child account?',
        '@AppleSupport just got billed for an in-app purchase I never authorized. Please help reverse this charge.'
    ],
    Intent.HARDWARE_DISPLAY_AUDIO: [
        '@AppleSupport my AirPods disconnect every 30 seconds during phone calls on iPhone X.',
        '@AppleSupport half of my screen is completely unresponsive to touch after dropping it on carpet.',
        '@AppleSupport speaker produces a distorted crackling noise whenever playing audio or video.'
    ],
    Intent.NETWORK_CONNECTIVITY: [
        '@AppleSupport my phone keeps saying No Service and searching for cellular network even after restart.',
        '@AppleSupport iPhone 8 will not connect to home WiFi network, keeps claiming incorrect password.',
        '@AppleSupport LTE data has stopped working completely since this morning while voice calls still work.'
    ],
    Intent.GENERAL_COMPLAINT_STORE: [
        '@AppleSupport your Genius Bar appointment system is terrible. Waited 2 hours past my scheduled time.',
        '@AppleSupport customer service at Regent Street was utterly unhelpful and rude today.',
        '@AppleSupport 10 years of buying Apple products and this customer experience is the absolute worst.'
    ],
    Intent.OTHER: [
        '@AppleSupport Thanks.',
        '@AppleSupport Okay sounds good.',
        '@AppleSupport https://t.co/exampleLink'
    ]
}

# Explicit Out-of-Scope boundaries (Required by brief)
OUT_OF_SCOPE_DECLARATION = {
    'multi_intent_resolution': (
        'Messages containing compound inquiries across multiple distinct domains (e.g. \"battery drains fast and also cancel my subscription\") '
        'are categorized by their primary actionable clause rather than multi-label decomposed.'
    ),
    'non_english_tweets': (
        'Tweets written primarily in languages other than English are filtered out during ingestion.'
    ),
    'multimodal_content': (
        'Tweets consisting solely of images, screenshots, or videos without accompanying textual diagnostic information are routed to other.'
    ),
    'private_dm_threads': (
        'Twitter Direct Messages (DMs) requiring customer credential verification or private identity exchanges are escalated rather than automated.'
    )
}

def get_all_intents() -> List[str]:
    return [i.value for i in Intent]

def get_intent_definitions() -> Dict[str, str]:
    return {k.value: v for k, v in TAXONOMY_DEFINITIONS.items()}

def get_intent_exemplars() -> Dict[str, List[str]]:
    return {k.value: v for k, v in TAXONOMY_EXEMPLARS.items()}

def is_valid_intent(intent_str: str) -> bool:
    return intent_str in get_all_intents()
