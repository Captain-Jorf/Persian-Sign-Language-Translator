from __future__ import annotations

# MediaPipe Hands returns 21 landmarks per hand, each with x/y/z.
HAND_LANDMARK_COUNT = 21
POINT_DIMS = 3
HAND_VECTOR_SIZE = HAND_LANDMARK_COUNT * POINT_DIMS
TWO_HAND_VECTOR_SIZE = HAND_VECTOR_SIZE * 2

DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_MIN_CONFIDENCE = 0.65

LEFT_HAND = "Left"
RIGHT_HAND = "Right"

# First public target. Keep this list boring and useful; impressive labels do not
# help a deaf person at a hospital desk.
DEFAULT_PSL_WORDS = [
    "سلام",
    "خداحافظ",
    "بله",
    "نه",
    "لطفا",
    "متشکرم",
    "ببخشید",
    "کمک",
    "آب",
    "غذا",
    "خانه",
    "مدرسه",
    "بیمارستان",
    "دکتر",
    "دارو",
    "درد",
    "خوب",
    "بد",
    "من",
    "تو",
    "او",
    "ما",
    "شما",
    "امروز",
    "فردا",
    "دیروز",
    "صبح",
    "شب",
    "کجا",
    "کی",
    "چرا",
    "چطور",
    "چه",
    "قیمت",
    "اتوبوس",
    "تاکسی",
    "کار",
    "پول",
    "شماره",
    "نام",
    "خانواده",
    "دوست",
    "پدر",
    "مادر",
    "کودک",
    "خرید",
    "سریع",
    "آرام",
    "خطر",
    "تمام",
]
