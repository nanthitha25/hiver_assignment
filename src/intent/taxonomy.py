"""Intent taxonomy, definitions, and canonical prototype examples for @AppleSupport."""

from typing import Dict, List
from src.models import AppleIntentEnum

INTENT_DESCRIPTIONS: Dict[AppleIntentEnum, str] = {
    AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING: (
        "Operating system bugs, iOS/macOS update issues, crashing/freezing apps, "
        "connectivity dropouts (Wi-Fi, Bluetooth, Cellular), boot loops, and system glitches."
    ),
    AppleIntentEnum.HARDWARE_AND_BATTERY: (
        "Physical hardware defects, rapid battery drain, degraded battery health, "
        "charging port failure, cracked displays, speaker/mic distortion, and overheating."
    ),
    AppleIntentEnum.ACCOUNT_BILLING_ICLOUD: (
        "Apple ID security, password resets, 2FA lockout, App Store billing/subscription charges, "
        "refund requests, and iCloud storage synchronization issues."
    ),
    AppleIntentEnum.HOW_TO_CONFIGURATION: (
        "General usage guidance, device setup, feature configuration (AirDrop, Apple Pay, Face ID), "
        "data transfer between devices, and standard operating procedures."
    ),
    AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS: (
        "Ambiguous statements, emotional rants without actionable technical context, "
        "memes, greetings, non-Apple topics, and unintelligible messages."
    ),
}

# Canonical seed prototypes used for semantic centroid embedding calculation
INTENT_PROTOTYPES: Dict[AppleIntentEnum, List[str]] = {
    AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING: [
        "My iPhone is stuck on the Apple logo after the new iOS update.",
        "Apps keep crashing and freezing on my iPad since this morning.",
        "Wi-Fi and Bluetooth disconnect constantly on my MacBook Pro.",
        "My screen is completely frozen and touch does not respond at all.",
        "The phone keeps restarting on its own every 5 minutes in a boot loop.",
        "Cannot update iOS because it says an error occurred checking for update.",
        "Notifications are not showing up on my lock screen after updating.",
        "Safari keeps crashing every time I open a new tab.",
    ],
    AppleIntentEnum.HARDWARE_AND_BATTERY: [
        "My battery health dropped to 78% and the phone dies within two hours.",
        "My phone is not charging at all when plugged into the charger cable.",
        "I dropped my iPhone and the front glass screen is shattered cracked.",
        "The bottom speaker is distorted and makes crackling static noise.",
        "My iPhone gets burning hot and overheats while browsing.",
        "The power button is stuck and will not click anymore.",
        "Camera shows a black screen and flash does not work.",
        "Headphone jack or lightning port is loose and wobbly.",
    ],
    AppleIntentEnum.ACCOUNT_BILLING_ICLOUD: [
        "My Apple ID has been locked for security reasons and I cannot reset password.",
        "I was charged $9.99 for an App Store subscription I already canceled.",
        "My iCloud storage says full even though I deleted thousands of photos.",
        "Two-factor verification code is not being received on my phone.",
        "Need a refund for an accidental in-app purchase my child made.",
        "Cannot sign into my iTunes account due to verification failed error.",
        "Payment method was declined when trying to update billing information.",
        "How do I remove an old device from my iCloud account?",
    ],
    AppleIntentEnum.HOW_TO_CONFIGURATION: [
        "How do I transfer all my photos from iPhone to my Windows computer?",
        "How do I set up Apple Pay on my new Apple Watch?",
        "How can I backup my iPhone to iCloud or to my Mac?",
        "How do I enable Night Shift or True Tone display in settings?",
        "Where can I find the serial number and IMEI number on my device?",
        "How do I turn on AirDrop to share files with another iPhone?",
        "Can I pair two pairs of AirPods to one iPad at the same time?",
        "How do I restore my contacts from a previous backup?",
    ],
    AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS: [
        "Why does this always happen to me smh worst day ever",
        "Apple is the absolute worst company in history lol",
        "Hello is anyone there? Please respond",
        "Help me please I have an issue",
        "Just testing to see if this bot actually replies",
        "Can you order a pizza for me right now?",
        "Nice weather today outside isn't it",
        "idk what is happening anymore whatever",
    ],
}
