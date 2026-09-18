"""
Clinical Knowledge Base, Symptom NLP Parser, and Medical Triage Engine.
Provides comprehensive clinical data, home remedies, Ayurvedic guidance,
dietary instructions, specialist recommendations, and symptom synonyms for all diseases.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Any

# Map dataset raw names to clean, professional clinical nomenclature
DISEASE_NAME_MAP = {
    "(vertigo) Paroymsal  Positional Vertigo": "Vertigo (BPPV)",
    "Acne": "Acne",
    "Allergy": "Allergy",
    "Arthritis": "Arthritis",
    "Bronchial Asthma": "Bronchial Asthma",
    "Cervical spondylosis": "Cervical Spondylosis",
    "Chicken pox": "Chickenpox",
    "Common Cold": "Common Cold & Flu",
    "Dengue": "Dengue",
    "Diabetes": "Diabetes",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids (Piles)",
    "Drug Reaction": "Drug Reaction",
    "Fungal infection": "Fungal Infection",
    "GERD": "GERD (Acid Reflux)",
    "Gastroenteritis": "Gastroenteritis (Food Poisoning / Stomach Infection)",
    "Hypertension": "Hypertension (High Blood Pressure)",
    "Hyperthyroidism": "Hyperthyroidism",
    "Hypoglycemia": "Hypoglycemia (Low Blood Sugar)",
    "Hypothyroidism": "Hypothyroidism",
    "Impetigo": "Impetigo",
    "Jaundice": "Jaundice",
    "Malaria": "Malaria",
    "Migraine": "Migraine",
    "Osteoarthristis": "Osteoarthritis",
    "Peptic ulcer diseae": "Peptic Ulcer Disease",
    "Psoriasis": "Psoriasis",
    "Typhoid": "Typhoid Fever",
    "Urinary tract infection": "Urinary Tract Infection (UTI)",
    "Varicose veins": "Varicose Veins",
    "hepatitis A": "Hepatitis A",
}

# Reverse lookup dictionary
REVERSE_DISEASE_NAME_MAP = {v: k for k, v in DISEASE_NAME_MAP.items()}

# Verified symptom profiles per disease from the clinical dataset
DISEASE_SYMPTOM_PROFILES: Dict[str, List[str]] = {
    "Acne": ["blackheads", "pus_filled_pimples", "scurring", "skin_rash"],
    "Allergy": ["chills", "continuous_sneezing", "shivering", "watering_from_eyes"],
    "Arthritis": ["movement_stiffness", "muscle_weakness", "painful_walking", "stiff_neck", "swelling_joints"],
    "Bronchial Asthma": ["breathlessness", "cough", "family_history", "fatigue", "high_fever", "mucoid_sputum"],
    "Cervical Spondylosis": ["back_pain", "dizziness", "loss_of_balance", "neck_pain", "weakness_in_limbs"],
    "Chickenpox": ["fatigue", "headache", "high_fever", "itching", "lethargy", "loss_of_appetite", "malaise", "mild_fever", "red_spots_over_body", "skin_rash", "swelled_lymph_nodes"],
    "Common Cold & Flu": ["chest_pain", "chills", "congestion", "continuous_sneezing", "cough", "fatigue", "headache", "high_fever", "loss_of_smell", "malaise", "muscle_pain", "phlegm", "redness_of_eyes", "runny_nose", "sinus_pressure", "swelled_lymph_nodes", "throat_irritation"],
    "Dengue": ["back_pain", "chills", "fatigue", "headache", "high_fever", "joint_pain", "loss_of_appetite", "malaise", "muscle_pain", "nausea", "pain_behind_the_eyes", "red_spots_over_body", "skin_rash", "vomiting"],
    "Diabetes": ["blurred_and_distorted_vision", "excessive_hunger", "fatigue", "increased_appetite", "irregular_sugar_level", "lethargy", "obesity", "polyuria", "restlessness", "weight_loss"],
    "Drug Reaction": ["burning_micturition", "itching", "skin_rash", "spotting_ urination", "stomach_pain"],
    "Fungal Infection": ["dischromic _patches", "itching", "nodal_skin_eruptions", "skin_rash"],
    "GERD (Acid Reflux)": ["acidity", "chest_pain", "cough", "stomach_pain", "ulcers_on_tongue", "vomiting"],
    "Gastroenteritis (Food Poisoning / Stomach Infection)": ["dehydration", "diarrhoea", "sunken_eyes", "vomiting"],
    "Hemorrhoids (Piles)": ["bloody_stool", "constipation", "irritation_in_anus", "pain_during_bowel_movements", "pain_in_anal_region"],
    "Hepatitis A": ["abdominal_pain", "dark_urine", "diarrhoea", "joint_pain", "loss_of_appetite", "mild_fever", "muscle_pain", "nausea", "vomiting", "yellowing_of_eyes", "yellowish_skin"],
    "Hypertension (High Blood Pressure)": ["chest_pain", "dizziness", "headache", "lack_of_concentration", "loss_of_balance"],
    "Hyperthyroidism": ["abnormal_menstruation", "diarrhoea", "excessive_hunger", "fast_heart_rate", "fatigue", "irritability", "mood_swings", "muscle_weakness", "restlessness", "sweating", "weight_loss"],
    "Hypoglycemia (Low Blood Sugar)": ["anxiety", "blurred_and_distorted_vision", "drying_and_tingling_lips", "excessive_hunger", "fatigue", "headache", "irritability", "nausea", "palpitations", "slurred_speech", "sweating", "vomiting"],
    "Hypothyroidism": ["abnormal_menstruation", "brittle_nails", "cold_hands_and_feets", "depression", "dizziness", "enlarged_thyroid", "fatigue", "irritability", "lethargy", "mood_swings", "puffy_face_and_eyes", "swollen_extremeties", "weight_gain"],
    "Impetigo": ["blister", "high_fever", "red_sore_around_nose", "skin_rash", "yellow_crust_ooze"],
    "Jaundice": ["abdominal_pain", "dark_urine", "fatigue", "high_fever", "itching", "vomiting", "weight_loss", "yellowish_skin"],
    "Malaria": ["chills", "diarrhoea", "headache", "high_fever", "muscle_pain", "nausea", "sweating", "vomiting"],
    "Migraine": ["acidity", "blurred_and_distorted_vision", "depression", "excessive_hunger", "headache", "indigestion", "irritability", "stiff_neck", "visual_disturbances"],
    "Osteoarthritis": ["hip_joint_pain", "joint_pain", "knee_pain", "neck_pain", "painful_walking", "swelling_joints"],
    "Peptic Ulcer Disease": ["abdominal_pain", "indigestion", "internal_itching", "loss_of_appetite", "passage_of_gases", "vomiting"],
    "Psoriasis": ["inflammatory_nails", "joint_pain", "silver_like_dusting", "skin_peeling", "skin_rash", "small_dents_in_nails"],
    "Typhoid Fever": ["abdominal_pain", "belly_pain", "chills", "constipation", "diarrhoea", "fatigue", "headache", "high_fever", "nausea", "toxic_look_(typhos)", "vomiting"],
    "Urinary Tract Infection (UTI)": ["bladder_discomfort", "burning_micturition", "continuous_feel_of_urine", "foul_smell_of urine"],
    "Varicose Veins": ["bruising", "cramps", "fatigue", "obesity", "prominent_veins_on_calf", "swollen_blood_vessels", "swollen_legs"],
    "Vertigo (BPPV)": ["headache", "loss_of_balance", "nausea", "spinning_movements", "unsteadiness", "vomiting"]
}

# Comprehensive NLP Synonym mapping for all 130+ symptoms (English + Native Kannada ಕನ್ನಡ + Transliterated Kannada)
SYMPTOM_SYNONYMS: Dict[str, List[str]] = {
    "itching": [
        "ತುರಿಕೆ", "ತುರಿಸುವುದು", "ನವೆ", "ಚರ್ಮದ ತುರಿಕೆ", "ಮೈ ತುರಿಕೆ", "ತುರಿಸುತ್ತಿದೆ", "ತುರಿಕೆ ಇದೆ", "ಮೈ ನವೆ",
        "turike", "thurike", "nave", "charmada turike", "mai turike",
        "itching", "itch", "itchy", "scratching", "pruritus", "skin itching", "itchiness", "itchy skin", "itching all over"
    ],
    "skin_rash": [
        "ದದ್ದು", "ದದ್ದುಗಳು", "ಚರ್ಮದ ದದ್ದು", "ಕೆಂಪು ದದ್ದು", "ಗುಳ್ಳೆಗಳು", "ಚರ್ಮ ಕೆಂಪಾಗುವುದು", "ಅಲರ್ಜಿ ದದ್ದು", "ಚರ್ಮದ ಗುಳ್ಳೆಗಳು",
        "daddu", "daddugalu", "kempu daddu", "charmada daddu", "skin rash", "rash", "rashes",
        "red marks", "breakout", "skin irritation", "erythema", "skin eruptions", "red rash", "skin rashes", "body rash"
    ],
    "nodal_skin_eruptions": [
        "ಗಂಟು ಗುಳ್ಳೆಗಳು", "ಚರ್ಮದ ಗಂಟುಗಳು", "ಗಂಟುಗಳು", "nodal eruptions", "gantu gullegalu",
        "nodal skin eruptions", "skin eruptions", "bumps on skin", "nodules", "lumps on skin", "skin bumps"
    ],
    "continuous_sneezing": [
        "ಸತತ ಸೀನು", "ಸೀನುವುದು", "ಸೀನುಗಳು", "ಸೀನು", "ಸೀನು ಬರುತ್ತಿದೆ", "seenu", "seenuvudu", "satata seenu", "sneezing",
        "continuous sneezing", "sneeze", "constant sneezing", "sneezing fits"
    ],
    "shivering": [
        "ನಡುಕ", "ಮೈ ನಡುಗುವುದು", "ದೇಹ ನಡುಕ", "ನಡುಗುತ್ತಿದೆ", "naduka", "mai nadukuvudu", "deha naduka",
        "shivering", "shiver", "trembling", "body shaking", "rigors", "shivers", "violent shivering"
    ],
    "chills": [
        "ಚಳಿ", "ವಿಪರೀತ ಚಳಿ", "ಚಳಿ ಆಗುವುದು", "ಮೈ ಚಳಿ", "ಚಳಿಯಾಗುತ್ತಿದೆ", "ತೀವ್ರ ಚಳಿ", "chali", "chali aaguvudu", "mai chali", "vipareetha chali",
        "chills", "feeling cold", "cold chills", "chilly", "goosebumps", "cold feeling", "shivering with cold"
    ],
    "joint_pain": [
        "ಕೀಲು ನೋವು", "ಕೀಲುಗಳ ನೋವು", "ಸಂಧಿವಾತ ನೋವು", "ಕೀಲು ಬೇನೆ", "ಕೈಕಾಲು ಕೀಲು ನೋವು", "keelu novu", "keelugala novu", "sandhivata",
        "joint pain", "pain in joints", "knee pain", "knees hurt", "arthralgia", "elbow pain", "finger joint pain", "wrist pain"
    ],
    "stomach_pain": [
        "ಹೊಟ್ಟೆ ನೋವು", "ಹೊಟ್ಟೆ ಶೂಲೆ", "ಹೊಟ್ಟೆ ಬೇನೆ", "ಹೊಟ್ಟೆ ಬೇನೆಯಾಗುವುದು", "ಹೊಟ್ಟೆ ಸೆಳೆತ", "ಹೊಟ್ಟೆ ಚುಚ್ಚುವುದು", "ಹೊಟ್ಟೆ ಕಡಿಯುವುದು",
        "hotte novu", "hotte shule", "hotte bene", "hotte seleta", "hottenovu",
        "stomach pain", "stomach ache", "tummy ache", "belly ache", "pain in stomach", "gastric pain", "stomach cramps", "gut pain", "burning stomach"
    ],
    "acidity": [
        "ಆಸಿಡಿಟಿ", "ಎದೆ ಉರಿ", "ಹುಳಿ ತೇಗು", "ಪಿತ್ತ", "ಹೊಟ್ಟೆ ಉರಿ", "ಗ್ಯಾಸ್ಟ್ರಿಕ್", "ಹುಳಿತೇಗು",
        "acidity", "ede uri", "huli tegu", "pittha", "hotte uri", "gastric", "acid reflux",
        "heartburn", "burning chest", "sour burps", "acid regurgitation", "hyperacidity", "burning in food pipe", "acid problem"
    ],
    "ulcers_on_tongue": [
        "ನಾಲಿಗೆ ಹುಣ್ಣು", "ಬಾಯಿ ಹುಣ್ಣು", "ನಾಲಿಗೆಯಲ್ಲಿ ಹುಣ್ಣುಗಳು", "ಬಾಯಿಯಲ್ಲಿ ಹುಣ್ಣು", "nalige hunnu", "baayi hunnu",
        "ulcers on tongue", "tongue ulcers", "mouth ulcers", "canker sores", "sores in mouth", "tongue sore"
    ],
    "muscle_wasting": [
        "ಸ್ನಾಯು ಕ್ಷೀಣತೆ", "ಸ್ನಾಯು ನಷ್ಟ", "snayu kshinathe",
        "muscle wasting", "muscle loss", "loss of muscle mass", "shrinking muscles", "muscle atrophy"
    ],
    "vomiting": [
        "ವಾಂತಿ", "ವಾಂತಿಯಾಗುವುದು", "ವಾಂತಿ ಬರುತ್ತಿದೆ", "ಓಕರಿಕೆ", "ವಾಂತಿ ಮಾಡುವುದು", "ವಾಂತಿ ಬೇಧಿ",
        "vaanti", "vanti", "okarke", "vaanthi", "vanti baruttide",
        "vomiting", "vomit", "throwing up", "puking", "emesis", "heaving", "food throwing", "vomitted", "barfing"
    ],
    "burning_micturition": [
        "ಉರಿ ಮೂತ್ರ", "ಮೂತ್ರದಲ್ಲಿ ಉರಿ", "ಮೂತ್ರ ವಿಸರ್ಜನೆ ವೇಳೆ ಉರಿ", "ಮೂತ್ರ ಸುಡುವುದು", "ಮೂತ್ರ ಮಾಡುವಾಗ ನೋವು", "ಉರಿಮೂತ್ರ",
        "uri moothra", "uri mootra", "moothradalli uri", "mootra uriyuvudu", "urimoothra",
        "burning micturition", "burning urination", "pain when peeing", "burning urine", "painful urination", "dysuria", "burning pee", "burning sensation when urinating", "pain in urination"
    ],
    "spotting_urination": [
        "ಮೂತ್ರದಲ್ಲಿ ರಕ್ತದ ಕಲೆ", "ಮೂತ್ರದ ಹನಿಗಳು", "moothradalli raktha",
        "spotting urination", "blood in urine drops", "spotting urine", "scanty dark urine drops"
    ],
    "fatigue": [
        "ಸುಸ್ತು", "ದಣಿವು", "ಆಯಾಸ", "ನಿಶ್ಯಕ್ತಿ", "ದೇಹದ ಆಯಾಸ", "ಸುಸ್ತಾಗುತ್ತಿದೆ", "ಅತಿಯಾದ ದಣಿವು", "ತ್ರಾಣವಿಲ್ಲದಿರುವುದು", "ಶಕ್ತಿ ಇಲ್ಲ",
        "susthu", "sustu", "danivu", "aayasa", "nishakthi", "traana illa",
        "fatigue", "tired", "tiredness", "exhaustion", "no energy", "feeling drained", "burnout", "extreme weakness", "lack of energy", "weariness", "feeling weak", "weakness", "weak", "body weakness", "exhausted"
    ],
    "weight_gain": [
        "ತೂಕ ಹೆಚ್ಚಾಗುವುದು", "ದೇಹದ ತೂಕ ಏರಿಕೆ", "ದಪ್ಪಗಾಗುವುದು", "tooka hechhaguvudu",
        "weight gain", "gaining weight", "putting on weight", "unexplained weight gain", "getting fat"
    ],
    "anxiety": [
        "ಆತಂಕ", "ಗಾಬರಿ", "ಭಯ", "ಚಿಂತೆ", "ನೆಮ್ಮದಿಯಿಲ್ಲ", "aatanka", "gabari", "chinte",
        "anxiety", "anxious", "nervous", "nervousness", "panic", "feeling uneasy", "worrying too much", "fearfulness"
    ],
    "cold_hands_and_feets": [
        "ಕೈ ಕಾಲು ತಣ್ಣಗಾಗುವುದು", "ತಣ್ಣನೆಯ ಕೈ ಕಾಲುಗಳು", "ಕೈಕಾಲು ತಣ್ಣಗಾಗಿದೆ", "kai kaalu tannagaguvudu",
        "cold hands and feets", "cold hands", "cold feet", "freezing extremities", "chilled palms", "cold feet and hands"
    ],
    "mood_swings": [
        "ಮನಸ್ಥಿತಿ ಬದಲಾವಣೆ", "ಕೋಪ ತಾಪ", "mood swings", "sudden mood changes", "emotional instability", "irritability swings"
    ],
    "weight_loss": [
        "ತೂಕ ಇಳಿಕೆ", "ತೂಕ ಕಡಿಮೆಯಾಗುವುದು", "ಸಣ್ಣಗಾಗುವುದು", "ತೂಕ ಇಳಿಯುವುದು", "tooka ilike", "tooka kadime", "sannagaguvudu",
        "weight loss", "losing weight", "rapid weight loss", "unintentional weight loss", "slimming rapidly"
    ],
    "restlessness": [
        "ಚಡಪಡಿಕೆ", "ಅಸಮಾಧಾನ", "ನೆಮ್ಮದಿಯಿಲ್ಲದಿರುವುದು", "chadapadike", "asamaadhana",
        "restlessness", "restless", "unable to sit still", "agitation", "fidgeting"
    ],
    "lethargy": [
        "ಜಡತ್ವ", "ಆಲಸ್ಯ", "ಮಂಪರು", "jadatwa", "aalasya", "mamparu",
        "lethargy", "lethargic", "sluggishness", "feeling sluggish", "laziness", "lack of motivation", "drowsy feeling"
    ],
    "patches_in_throat": [
        "ಗಂಟಲಿನಲ್ಲಿ ಬಿಳಿ ಕಲೆಗಳು", "ಗಂಟಲು ಪ್ಯಾಚ್", "gantalinalli machhe",
        "patches in throat", "white patches in throat", "throat spots", "throat coating", "tonsil patches"
    ],
    "irregular_sugar_level": [
        "ಸಕ್ಕರೆ ಮಟ್ಟದಲ್ಲಿ ಏರಿಳಿತ", "ಅನಿಯಮಿತ ಶುಗರ್", "ಬ್ಲಡ್ ಶುಗರ್ ಏರಿಕೆ", "sugar yerilitha",
        "irregular sugar level", "fluctuating blood sugar", "high sugar", "unstable glucose", "sugar spikes"
    ],
    "cough": [
        "ಕೆಮ್ಮು", "ಒಣ ಕೆಮ್ಮು", "ಕಫದ ಕೆಮ್ಮು", "ವಿಪರೀತ ಕೆಮ್ಮು", "ಕೆಮ್ಮುವುದು", "ಕೆಮ್ಮು ಬರುತ್ತಿದೆ", "ಕೆಮ್ಮು ಇದೆ", "ತೀವ್ರ ಕೆಮ್ಮು",
        "kemmu", "ona kemmu", "kaphada kemmu", "kemmu baruttide", "kemmu ide",
        "cough", "coughing", "dry cough", "wet cough", "phlegmy cough", "hacking cough", "persistent cough", "caugh", "throat coughing"
    ],
    "high_fever": [
        "ಜ್ವರ", "ವಿಪರೀತ ಜ್ವರ", "ಹೆಚ್ಚಿನ ಜ್ವರ", "ಬಿಸಿ ಮೈ", "ತೀವ್ರ ಜ್ವರ", "ಮೈ ಬಿಸಿ", "ಜ್ವರ ಬಂದಿದೆ", "ಜ್ವರ ಇದೆ", "ಕಾಯಿಲೆ",
        "jwara", "jvara", "thivra jwara", "fever", "vipareetha jwara", "mai bisi", "jwara bandide", "jwara ide",
        "high fever", "high temperature", "temperature", "hot body", "pyrexia", "feverish", "burning with fever", "spiking fever", "having fever", "running temperature", "running fever"
    ],
    "sunken_eyes": [
        "ಗುಳಿಬಿದ್ದ ಕಣ್ಣುಗಳು", "ಕುಗ್ಗಿದ ಕಣ್ಣುಗಳು", "ಕಣ್ಣು ಒಳಗೆ ಹೋಗಿದೆ", "guli bidda kannugalu",
        "sunken eyes", "deep hollow eyes", "hollowed eyes", "dark sunken eyes", "eyes sunken in"
    ],
    "breathlessness": [
        "ಉಸಿರಾಟದ ತೊಂದರೆ", "ಉಬ್ಬಸ", "ಉಸಿರು ಕಟ್ಟುವುದು", "ದಮ್ಮು", "ಉಸಿರಾಡಲು ಕಷ್ಟ", "ಉಸಿರಾಟ ಕಷ್ಟ", "ಉಸಿರು ಸಿಗುತ್ತಿಲ್ಲ",
        "usiratada thondare", "ubbaasa", "usiru kattuvudu", "dammu", "usiradalu kashta",
        "breathlessness", "shortness of breath", "cant breathe", "difficulty breathing", "dyspnea", "gasping for air", "winded", "labored breathing", "trouble breathing"
    ],
    "sweating": [
        "ಬೆವರುವುದು", "ಅತಿಯಾದ ಬೆವರು", "ರಾತ್ರಿ ಬೆವರು", "ಬೆವರು", "ಬೆವರು ಬರುತ್ತಿದೆ", "bevaruvudu", "athiyaada bevaru", "bevaru",
        "sweating", "night sweats", "perspiring", "excessive sweat", "profuse sweating", "sweaty body", "sweat"
    ],
    "dehydration": [
        "ನಿರ್ಜಲೀಕರಣ", "ಬಾಯಾರಿಕೆ", "ದೇಹದಲ್ಲಿ ನೀರಿನ ಕೊರತೆ", "ಬಾಯಿ ಒಣಗುವುದು", "ಅತಿಯಾದ ಬಾಯಾರಿಕೆ", "nirjaleekarana", "bayarike", "baayi onaguvudu",
        "dehydration", "dehydrated", "very thirsty", "extreme thirst", "dry mouth and throat", "lack of water in body", "water loss"
    ],
    "indigestion": [
        "ಅಜೀರ್ಣ", "ಹೊಟ್ಟೆ ಉಬ್ಬರ", "ಜೀರ್ಣವಾಗದಿರುವುದು", "ಅಜೀರ್ಣತೆ", "ajeerna", "hotte ubbara",
        "indigestion", "bloating", "upset stomach", "dyspepsia", "food not digesting", "heavy stomach", "gas problem"
    ],
    "headache": [
        "ತಲೆನೋವು", "ತಲೆ ನೋವು", "ತಲೆ ಭಾರ", "ವಿಪರೀತ ತಲೆನೋವು", "ತಲೆ ಸಿಡಿಯುವುದು", "ಅರೆತಲೆನೋವು", "ತಲೆ ಕೆರೆತ", "ತಲೆ ಸಿಡಿತ",
        "talenovu", "tale novu", "tale novvu", "tale bhaara", "tale sidi", "aretalenovu",
        "headache", "head pain", "head hurts", "throbbing head", "head heavy", "migraine ache", "forehead ache", "cephalalgia", "severe headache", "pain in head"
    ],
    "yellowish_skin": [
        "ಹಳದಿ ಚರ್ಮ", "ಚರ್ಮ ಹಳದಿಯಾಗುವುದು", "ಕಾಮಾಲೆ ಚರ್ಮ", "ಕಾಮಾಲೆ", "haladi charma", "haladi mai", "kaamale", "kamale",
        "yellowish skin", "yellow skin", "jaundice skin", "icterus", "skin looking yellow", "pale yellow skin"
    ],
    "dark_urine": [
        "ಗಾಢ ಹಳದಿ ಮೂತ್ರ", "ಕಡು ಮೂತ್ರ", "ಕಪ್ಪನೆಯ ಮೂತ್ರ", "ಹಳದಿ ಮೂತ್ರ", "gaadha haladi moothra", "kadu mootra", "haladi moothra",
        "dark urine", "brown urine", "tea colored urine", "deep yellow urine", "dark colored pee"
    ],
    "nausea": [
        "ವಾಕರಿಕೆ", "ವಾಂತಿ ಬಂದಂತಾಗುವುದು", "ತಲೆ ತಿರುಗಿದಂತಾಗುವುದು", "ಹೊಟ್ಟೆ ತೊಳಸುವುದು", "vaakarike", "vakarike", "hotte tholasuvudu",
        "nausea", "feeling sick", "queasy", "nauseous", "feeling like vomiting", "sick to stomach", "urge to vomit"
    ],
    "loss_of_appetite": [
        "ಹಸಿವಿಲ್ಲದಿರುವುದು", "ಹಸಿವು ಇಲ್ಲ", "ಊಟ ಸೇರದಿರುವುದು", "ಹಸಿವಿಲ್ಲ", "hasivu illa", "hasivilladiruvudu", "ootha seradhe", "ootha seralla",
        "loss of appetite", "no appetite", "not hungry", "reduced eating", "anorexia", "poor appetite", "dont feel like eating"
    ],
    "pain_behind_the_eyes": [
        "ಕಣ್ಣಿನ ಹಿಂಭಾಗದ ನೋವು", "ಕಣ್ಣು ನೋವು", "ಕಣ್ಣಿನ ನೋವು", "kannina himbhagada novu", "kannu novu",
        "pain behind the eyes", "eye socket pain", "retro orbital pain", "eye pain on moving", "behind eyes aching"
    ],
    "back_pain": [
        "ಬೆನ್ನು ನೋವು", "ಸೊಂಟ ನೋವು", "ಕೆಳಬೆನ್ನು ನೋವು", "ಬೆನ್ನು ಮೂಳೆ ನೋವು", "bennu novu", "sonta novu", "sonta novvu",
        "back pain", "back ache", "lower back pain", "upper back pain", "spine pain", "lumbago", "back hurts"
    ],
    "constipation": [
        "ಮಲಬದ್ಧತೆ", "ಮಲ ಕಟ್ಟಿಕೊಳ್ಳುವುದು", "ಮಲ ವಿಸರ್ಜನೆ ಕಷ್ಟ", "malabaddhate", "mala kattu",
        "constipation", "constipated", "hard stool", "difficulty passing stool", "irregular bowel"
    ],
    "abdominal_pain": [
        "ಹೊಟ್ಟೆ ನೋವು", "ಹೊಟ್ಟೆ ಸೆಳೆತ", "ಕಿಬ್ಬೊಟ್ಟೆ ನೋವು", "ಹೊಟ್ಟೆ ಶೂಲೆ", "hotte novu", "kibbotte novu", "hotte seleta",
        "abdominal pain", "belly pain", "stomach cramps", "lower abdominal pain", "cramping belly", "gut cramps", "tummy pain"
    ],
    "diarrhoea": [
        "ಭೇದಿ", "ಅತಿಸಾರ", "ಬೇಧಿ", "ಹೊಟ್ಟೆ ತೊಳೆಸುವ ಭೇದಿ", "ನೀರು ಭೇದಿ", "ಲೂಸ್ ಮೋಷನ್", "ಹೊಟ್ಟೆ ಕೆಡುವುದು", "ಮಲ ತೆಳುವಾಗುವುದು",
        "bhedi", "bedhi", "atisaara", "neeru bhedi", "loose motion", "loose motions",
        "diarrhoea", "diarrhea", "watery stool", "loose stools", "frequent motions", "running stomach", "watery motions", "dysentery", "loose stool", "stomach loose"
    ],
    "mild_fever": [
        "ಸ್ವಲ್ಪ ಜ್ವರ", "ಸಣ್ಣ ಜ್ವರ", "ಮಂದ ಜ್ವರ", "ಕಡಿಮೆ ಜ್ವರ", "sanna jwara", "swalpa jwara",
        "mild fever", "low fever", "slight fever", "low grade fever", "warm body"
    ],
    "yellow_urine": [
        "ಹಳದಿ ಮೂತ್ರ", "ಅತಿಯಾದ ಹಳದಿ ಮೂತ್ರ", "haladi moothra",
        "yellow urine", "bright yellow urine", "dark yellow pee"
    ],
    "yellowing_of_eyes": [
        "ಕಣ್ಣುಗಳು ಹಳದಿಯಾಗುವುದು", "ಹಳದಿ ಕಣ್ಣು", "ಕಾಮಾಲೆ ಕಣ್ಣುಗಳು", "ಕಣ್ಣು ಹಳದಿ", "kannugalu haladi", "haladi kannu", "kaamale kannu",
        "yellowing of eyes", "yellow eyes", "scleral icterus", "whites of eyes turned yellow"
    ],
    "acute_liver_failure": [
        "ಯಕೃತ್ ವೈಫಲ್ಯ", "ಲಿವರ್ ತೊಂದರೆ", "acute liver failure", "liver failure", "hepatic dysfunction"
    ],
    "fluid_overload": [
        "ದೇಹದಲ್ಲಿ ನೀರು ತುಂಬುವುದು", "ದ್ರವ ಶೇಖರಣೆ", "fluid overload", "water retention", "edema", "body swelling with fluid"
    ],
    "swelling_of_stomach": [
        "ಹೊಟ್ಟೆ ಊತ", "ಹೊಟ್ಟೆ ಊದಿಕೊಳ್ಳುವುದು", "ಹೊಟ್ಟೆ ಬಲೂನಿನಂತಾಗುವುದು", "hotte ootha",
        "swelling of stomach", "stomach distension", "ascites", "swollen abdomen", "belly swelling"
    ],
    "swelled_lymph_nodes": [
        "ಕುತ್ತಿಗೆಯಲ್ಲಿ ಗಂಟು ಊತ", "ದುಗ್ಧರಸ ಗ್ರಂಥಿ ಊತ", "swelled lymph nodes", "swollen glands", "swollen lymph nodes", "neck swelling", "gland swelling", "lymphadenopathy"
    ],
    "malaise": [
        "ಮೈ ಅಸ್ವಸ್ಥತೆ", "ದೇಹ ಸೌಖ್ಯವಿಲ್ಲದಿರುವುದು", "ಅನಾರೋಗ್ಯ", "malaise", "general discomfort", "feeling unwell", "body unwell", "overall sickness"
    ],
    "blurred_and_distorted_vision": [
        "ಮಸುಕಾದ ದೃಷ್ಟಿ", "ಕಣ್ಣು ಮಸುಕು", "ಕಣ್ಣು ಸರಿಯಾಗಿ ಕಾಣಿಸದಿರುವುದು", "ದೃಷ್ಟಿ ಮಂದವಾಗುವುದು", "masukaada drushti", "kannu masuku",
        "blurred and distorted vision", "blurred vision", "blurry vision", "distorted vision", "foggy vision", "hazy sight"
    ],
    "phlegm": [
        "ಕಫ", "ಲೋಳೆ", "ಗಂಟಲಿನಲ್ಲಿ ಕಫ", "ಬಿಳಿ ಕಫ", "ಕಫ ಬರುತ್ತಿದೆ", "kapha", "kapa", "lole", "gantalinalli kapha",
        "phlegm", "mucus", "sputum", "coughing mucus", "thick phlegm", "catarrh"
    ],
    "throat_irritation": [
        "ಗಂಟಲು ನೋವು", "ಗಂಟಲು ಕೆರೆತ", "ಗಂಟಲು ಕಿರಿಕಿರಿ", "ಗಂಟಲು ಊತ", "ಗಂಟಲು ಕೆರೆದುಕೊಳ್ಳುವುದು", "ನುಂಗಲು ಕಷ್ಟ", "ಗಂಟಲು ಬೇನೆ", "ಗಂಟಲು ಕಟ್ಟುವುದು",
        "gantalu novu", "gantlu novu", "gantalu kereta", "gantalu keretha", "gantalu kirikiri", "nungalu kashta", "gantalu bene",
        "throat irritation", "sore throat", "throat pain", "scratchy throat", "raw throat", "pain swallowing", "itchy throat", "throat infection", "pain in throat"
    ],
    "redness_of_eyes": [
        "ಕಣ್ಣು ಕೆಂಪಾಗುವುದು", "ಕೆಂಪು ಕಣ್ಣುಗಳು", "ಕಣ್ಣು ಕೆಂಪು", "kannu kempaguvudu", "kempu kannu",
        "redness of eyes", "red eyes", "bloodshot eyes", "pink eye", "eye inflammation"
    ],
    "sinus_pressure": [
        "ಸೈನಸ್ ನೋವು", "ಹಣೆ ನೋವು", "ಹಣೆ ಭಾರ", "ಮೂಗಿನ ಸೈನಸ್ ಒತ್ತಡ", "sinus novu", "hane novu", "hane bhaara",
        "sinus pressure", "sinus pain", "facial pressure", "forehead pressure", "nasal sinus blockage"
    ],
    "runny_nose": [
        "ನೆಗಡಿ", "ಮೂಗು ಸೋರುವುದು", "ಮೂಗಿನಲ್ಲಿ ನೀರು", "ಶೀತ", "ನೆಗಡಿ ಇದೆ", "ಮೂಗು ಸೋರುತ್ತಿದೆ",
        "negadi", "negedi", "sheetha", "shitha", "moogu soruvudu", "moogu neeru",
        "runny nose", "running nose", "dripping nose", "nasal discharge", "watery nose", "rhinorrhea", "cold in nose", "sniffles"
    ],
    "congestion": [
        "ಮೂಗು ಕಟ್ಟುವಿಕೆ", "ಮೂಗು ಕಟ್ಟುವುದು", "ಮೂಗು ಕಟ್ಟಿದೆ", "ಎದೆ ಕಟ್ಟುವುದು", "ಕಟ್ಟುವುದು", "moogu kattu", "moogu kattuvike", "moogu kattide", "ede kattu",
        "congestion", "blocked nose", "stuffy nose", "nasal blockage", "clogged nose", "stuffy nasal passages", "nose block", "chest congestion"
    ],
    "chest_pain": [
        "ಎದೆ ನೋವು", "ಎದೆ ಬಿಗಿತ", "ಎದೆ ಭಾರ", "ಎದೆಯಲ್ಲಿ ನೋವು", "ಎದೆ ಚುಚ್ಚುವುದು", "ede novu", "ede bigitha", "ede bhaara", "ede chuchuvudu",
        "chest pain", "pain in chest", "heart pain", "chest tightness", "chest pressure", "angina", "sternum pain"
    ],
    "weakness_in_limbs": [
        "ಕೈಕಾಲುಗಳಲ್ಲಿ ಶಕ್ತಿ ಇಲ್ಲದಿರುವುದು", "ಕೈಕಾಲು ನಿಶ್ಯಕ್ತಿ", "ಕೈ ಕಾಲು ಬಲಹೀನತೆ", "kaikaalu nishakthi",
        "weakness in limbs", "weak arms", "weak legs", "loss of limb strength", "limbs feel heavy"
    ],
    "fast_heart_rate": [
        "ಹೃದಯ ಬಡಿತ ಹೆಚ್ಚಾಗುವುದು", "ಗುಂಡಿಗೆ ಬಡಿತ", "ಹೃದಯ ವೇಗವಾಗಿ ಬಡಿಯುವುದು", "hrudaya baditha",
        "fast heart rate", "tachycardia", "racing heart", "rapid pulse", "quick heartbeat"
    ],
    "pain_during_bowel_movements": [
        "ಮಲ ವಿಸರ್ಜನೆ ವೇಳೆ ನೋವು", "ಶೌಚಕ್ಕೆ ಹೋಗುವಾಗ ನೋವು", "pain during bowel movements", "painful defecation", "pain passing stool", "anal pain on stool"
    ],
    "pain_in_anal_region": [
        "ಗುದದ್ವಾರದ ನೋವು", "ಮೂಲವ್ಯಾಧಿ ನೋವು", "anal pain", "pain in anal region", "rectal pain", "pain in bottom", "pain around anus"
    ],
    "bloody_stool": [
        "ಮಲದಲ್ಲಿ ರಕ್ತ", "ರಕ್ತ ಭೇದಿ", "ರಕ್ತ ಬೀಳುವುದು", "maladalli raktha", "raktha bhedi",
        "bloody stool", "blood in stool", "rectal bleeding", "red blood in feces", "hematochezia"
    ],
    "irritation_in_anus": [
        "ಗುದದ್ವಾರದಲ್ಲಿ ತುರಿಕೆ", "ಮೂಲವ್ಯಾಧಿ ಉರಿ", "irritation in anus", "anal itching", "pruritus ani", "burning in anus"
    ],
    "neck_pain": [
        "ಕುತ್ತಿಗೆ ನೋವು", "ಕುತ್ತಿಗೆ ಬಿಗಿತ", "ಕುತ್ತಿಗೆ ಬೇನೆ", "kuttige novu", "kuttige bigitha",
        "neck pain", "neck ache", "stiff neck pain", "cervical pain", "sore neck"
    ],
    "dizziness": [
        "ತಲೆಸುತ್ತು", "ತಲೆ ತಿರುಗುವುದು", "ತಲೆತಿರುಗುವಿಕೆ", "ಮೈ ಮರೆವು", "talesuthu", "tale tiruguvudu", "talesuttu",
        "dizziness", "dizzy", "giddy", "head spinning", "lightheaded", "lightheadedness", "fainting sensation"
    ],
    "cramps": [
        "ಸ್ನಾಯು ಸೆಳೆತ", "ಸೆಳೆತ", "ಕಾಲು ಸೆಳೆತ", "snayu seleta", "kaalu seleta",
        "cramps", "muscle cramps", "leg cramps", "calf cramps", "spasms"
    ],
    "bruising": [
        "ಚರ್ಮದ ಮೇಲೆ ರಕ್ತಗಟ್ಟುವಿಕೆ", "ನೀಲಿ ಕಲೆ", "bruising", "easy bruising", "blue marks on skin", "ecchymosis", "skin bruises"
    ],
    "obesity": [
        "ಸ್ಥೂಲಕಾಯ", "ಅತಿಯಾದ ತೂಕ", "ದಪ್ಪ ಶರೀರ", "obesity", "overweight", "excessive body fat", "morbid obesity"
    ],
    "swollen_legs": [
        "ಕಾಲುಗಳಲ್ಲಿ ಊತ", "ಕಾಲು ಊತ", "ಪಾದಗಳ ಊತ", "kaalugalalli ootha", "kaalu ootha",
        "swollen legs", "leg swelling", "swollen feet", "pedal edema", "puffy legs"
    ],
    "swollen_blood_vessels": [
        "ಉಬ್ಬಿದ ರಕ್ತನಾಳಗಳು", "ಸಿರೆಗಳ ಊತ", "swollen blood vessels", "enlarged veins", "visible blue veins", "bulging veins"
    ],
    "puffy_face_and_eyes": [
        "ಮುಖ ಮತ್ತು ಕಣ್ಣುಗಳ ಊತ", "ಮುಖ ಊದಿಕೊಳ್ಳುವುದು", "puffy face and eyes", "puffy face", "facial swelling", "swollen eyelids", "puffy eyes"
    ],
    "enlarged_thyroid": [
        "ಥೈರಾಯ್ಡ್ ಊತ", "ಗಳಗಂಡ", "enlarged thyroid", "goiter", "swollen neck front", "thyroid swelling"
    ],
    "brittle_nails": [
        "ಒಡೆಯುವ ಉಗುರುಗಳು", "ಬಲಹೀನ ಉಗುರು", "brittle nails", "breaking nails", "fragile nails", "cracking nails"
    ],
    "swollen_extremeties": [
        "ಕೈ ಕಾಲುಗಳಲ್ಲಿ ಊತ", "ಅಂಗಾಂಗಗಳ ಊತ", "swollen extremeties", "swollen hands and feet", "extremity swelling"
    ],
    "excessive_hunger": [
        "ಅತಿಯಾದ ಹಸಿವು", "ಸತತ ಹಸಿವು", "excessive hunger", "always hungry", "polyphagia", "increased appetite", "constant hunger"
    ],
    "drying_and_tingling_lips": [
        "ತುಟಿ ಒಣಗುವುದು ಮತ್ತು ಜುಮುಗುಡುವುದು", "drying and tingling lips", "tingling lips", "dry numb lips", "lip numbness"
    ],
    "slurred_speech": [
        "ಮಾತು ತೊದಲುವಿಕೆ", "ಮಾತನಾಡಲು ಕಷ್ಟ", "slurred speech", "difficulty speaking", "garbled speech", "incoherent speech"
    ],
    "knee_pain": [
        "ಮೊಣಕಾಲು ನೋವು", "ಕಾಲು ನೋವು", "ಮಂಡಿ ನೋವು", "monakalu novu", "mandi novu", "kaalu novu",
        "knee pain", "pain in knees", "knee joint ache", "knees hurting"
    ],
    "hip_joint_pain": [
        "ಸೊಂಟದ ಕೀಲು ನೋವು", "ಹಿಪ್ ನೋವು", "hip joint pain", "pain in hip", "hip ache"
    ],
    "muscle_weakness": [
        "ಸ್ನಾಯು ದೌರ್ಬಲ್ಯ", "ಸ್ನಾಯುಗಳಲ್ಲಿ ಶಕ್ತಿ ಇಲ್ಲ", "muscle weakness", "weak muscles", "loss of muscle strength", "myasthenia"
    ],
    "stiff_neck": [
        "ಕುತ್ತಿಗೆ ಬಿಗಿತ", "ಕುತ್ತಿಗೆ ತಿರುಗಿಸಲು ಕಷ್ಟ", "stiff neck", "neck stiffness", "inability to bend neck", "rigid neck"
    ],
    "swelling_joints": [
        "ಕೀಲುಗಳಲ್ಲಿ ಊತ", "ಕೀಲು ಊತ", "ಊದಿಕೊಂಡ ಕೀಲುಗಳು", "keelugalalli ootha", "keelu ootha",
        "swelling joints", "swollen joints", "joint swelling", "puffy joints"
    ],
    "movement_stiffness": [
        "ಚಲನೆಯಲ್ಲಿ ಬಿಗಿತ", "ಮೈ ಬಿಗಿತ", "movement stiffness", "stiff body", "stiffness in morning", "joint rigidity"
    ],
    "spinning_movements": [
        "ತಲೆ ಗಿರ್ರನೆ ತಿರುಗುವುದು", "ಗಿರ್ರನೆ ತಿರುಗುವ ಅನುಭವ", "tale girrane tiruguvudu",
        "spinning movements", "room spinning", "vertigo sensation", "spinning head"
    ],
    "loss_of_balance": [
        "ಸಮತೋಲನ ತಪ್ಪುವುದು", "ತೂರಾಡುವುದು", "ನಡೆಯಲು ಕಷ್ಟ", "samatholana thappuvudu",
        "loss of balance", "unsteadiness", "off balance", "imbalance walking", "stumbling", "poor coordination"
    ],
    "unsteadiness": [
        "ಅಸ್ಥಿರತೆ", "ಕಾಲು ತೂರಾಡುವುದು", "unsteadiness", "wobbly walking", "shaky footing", "instability"
    ],
    "weakness_of_one_body_side": [
        "ದೇಹದ ಒಂದು ಬದಿಯ ದೌರ್ಬಲ್ಯ", "ಒಂದು ಬದಿ ಪಾರ್ಶ್ವವಾಯು", "weakness of one body side", "hemiparesis", "one side weak", "paralysis of one side"
    ],
    "loss_of_smell": [
        "ವಾಸನೆ ತಿಳಿಯದಿರುವುದು", "ವಾಸನೆ ಗ್ರಹಿಕೆ ನಷ್ಟ", "loss of smell", "anosmia", "cannot smell", "lost sense of smell"
    ],
    "bladder_discomfort": [
        "ಮೂತ್ರಕೋಶದ ನೋವು", "ಮೂತ್ರಕೋಶದ ಅಸ್ವಸ್ಥತೆ", "ಮೂತ್ರದ ಒತ್ತಡ", "moothrakoshada novu",
        "bladder discomfort", "pelvic discomfort", "full bladder pressure", "bladder ache", "bladder pain"
    ],
    "foul_smell_ofurine": [
        "ದುರ್ವಾಸನೆಯ ಮೂತ್ರ", "ಮೂತ್ರದಲ್ಲಿ ಕೆಟ್ಟ ವಾಸನೆ", "foul smell of urine", "foul_smell_of urine", "smelly urine", "malodorous urine", "stinky pee"
    ],
    "continuous_feel_of_urine": [
        "ಸತತ ಮೂತ್ರ ವಿಸರ್ಜನೆಯ ಭಾವನೆ", "ಮೂತ್ರದ ತೀವ್ರತೆ", "continuous feel of urine", "frequent urge to pee", "urgency urination", "feeling like peeing continuously"
    ],
    "passage_of_gases": [
        "ವಾಯು ಪ್ರಕೋಪ", "ಗ್ಯಾಸ್ ಬಿಡುವುದು", "ತೇಗು", "passage of gases", "flatulence", "excessive gas", "burping and farting", "passing wind"
    ],
    "internal_itching": [
        "ಆಂತರಿಕ ತುರಿಕೆ", "internal itching", "deep itching", "internal prickling sensation"
    ],
    "toxic_look_(typhos)": [
        "ವಿಷಪೂರಿತ ನೋಟ", "ದಣಿದ ಮುಖ", "ಟೈಫಾಯ್ಡ್ ಆಲಸ್ಯ", "toxic look (typhos)", "typhoid look", "toxic appearance", "severely sick appearance", "dull lethargic appearance"
    ],
    "depression": [
        "ಖಿನ್ನತೆ", "ಮನಸ್ಸಿನ ಬೇಸರ", "ತೀವ್ರ ದುಃಖ", "depression", "depressed mood", "feeling very low", "sadness", "hopelessness"
    ],
    "irritability": [
        "ಕಿರಿಕಿರಿ", "ಸಿಡುಕುತನ", "ಬೇಗ ಕೋಪ ಬರುವುದು", "irritability", "irritable", "cranky", "getting angry easily", "short tempered"
    ],
    "muscle_pain": [
        "ಮೈಕೈ ನೋವು", "ಮೈ ನೋವು", "ಮೈಕೈನೋವು", "ದೇಹದ ನೋವು", "ಸ್ನಾಯು ನೋವು", "ಅಂಗಾಂಗ ನೋವು", "ಮೈ ಕೈ ನೋಯುತ್ತಿದೆ",
        "mai kai novu", "maikai novu", "mai novu", "dehada novu", "snayu novu", "anganga novu", "maikainovu",
        "muscle pain", "body ache", "body pain", "bodypain", "myalgia", "sore muscles", "whole body hurts", "body aches", "bodyache"
    ],
    "altered_sensorium": [
        "ಪ್ರಜ್ಞಾಹೀನತೆ", "ಗೊಂದಲ", "altered sensorium", "confusion", "delirium", "disorientation", "altered consciousness"
    ],
    "red_spots_over_body": [
        "ದೇಹದ ಮೇಲೆ ಕೆಂಪು ಕಲೆಗಳು", "ಕೆಂಪು ಚುಕ್ಕೆಗಳು", "ದಡಾರ ಕಲೆಗಳು", "kempu chukkegalu", "kempu kale",
        "red spots over body", "petechiae", "red dots on skin", "measles rash", "purpura", "red skin spots"
    ],
    "belly_pain": [
        "ಹೊಟ್ಟೆ ನೋವು", "ಕಿಬ್ಬೊಟ್ಟೆ ನೋವು", "hotte novu", "kibbotte novu", "belly pain", "lower stomach ache", "abdomen hurts", "gut pain"
    ],
    "abnormal_menstruation": [
        "ಅಸಹಜ ಮುಟ್ಟು", "ಅನಿಯಮಿತ ಮುಟ್ಟಿನ ಸಮಸ್ಯೆ", "abnormal menstruation", "irregular periods", "heavy menstrual bleeding", "missed periods"
    ],
    "dischromic_patches": [
        "ಚರ್ಮದ ಬಣ್ಣ ಬದಲಾವಣೆ ಕಲೆಗಳು", "ಬಿಳಿ ಕಲೆಗಳು", "dischromic _patches", "dischromic patches", "discolored patches", "skin discoloration", "tinea versicolor", "white skin patches"
    ],
    "watering_from_eyes": [
        "ಕಣ್ಣಿನಲ್ಲಿ ನೀರು ಬರುವುದು", "ಕಣ್ಣೀರು", "kanninalli neeru",
        "watering from eyes", "watery eyes", "excessive tears", "lacrimation", "eyes watering"
    ],
    "increased_appetite": [
        "ಹೆಚ್ಚಿದ ಹಸಿವು", "increased appetite", "huge appetite", "eating frequently"
    ],
    "polyuria": [
        "ಅತಿಯಾದ ಮೂತ್ರ ವಿಸರ್ಜನೆ", "ಸತತ ಮೂತ್ರ", "polyuria", "frequent urination", "peeing a lot", "excessive urination", "urination at night"
    ],
    "family_history": [
        "ವಂಶಪಾರಂಪರ್ಯ", "ಕುಟುಂಬದ ಇತಿಹಾಸ", "family history", "hereditary", "genetics", "runs in family"
    ],
    "mucoid_sputum": [
        "ಲೋಳೆಯುಕ್ತ ಕಫ", "ಬಿಳಿ ಕಫ", "mucoid sputum", "thick white phlegm", "sticky sputum", "mucus cough"
    ],
    "rusty_sputum": [
        "ಕಂದು ಬಣ್ಣದ ಕಫ", "ರಕ್ತಮಿಶ್ರಿತ ಕಫ", "rusty sputum", "brownish phlegm", "blood tinged sputum"
    ],
    "lack_of_concentration": [
        "ಏಕಾಗ್ರತೆಯ ಕೊರತೆ", "ಗಮನ ಕೇಂದ್ರೀಕರಿಸಲು ಕಷ್ಟ", "lack of concentration", "brain fog", "cannot focus", "difficulty concentrating"
    ],
    "visual_disturbances": [
        "ದೃಷ್ಟಿ ದೋಷ", "ಕಣ್ಣಿನ ಮಿಂಚು", "visual disturbances", "aura", "flashing lights", "blind spots in vision"
    ],
    "coma": [
        "ಕೋಮಾ", "ಪ್ರಜ್ಞಾಹೀನ ಸ್ಥಿತಿ", "coma", "unconscious", "unresponsive", "passed out deeply"
    ],
    "stomach_bleeding": [
        "ಹೊಟ್ಟೆಯಲ್ಲಿ ರಕ್ತಸ್ರಾವ", "ವಾಂತಿಯಲ್ಲಿ ರಕ್ತ", "stomach bleeding", "vomiting blood", "black tarry stool", "hematemesis", "melena"
    ],
    "distention_of_abdomen": [
        "ಹೊಟ್ಟೆ ಉಬ್ಬರ", "ಉಬ್ಬಿದ ಹೊಟ್ಟೆ", "distention of abdomen", "swollen belly", "bloated abdomen", "abdominal swelling"
    ],
    "blood_in_sputum": [
        "ಕಫದಲ್ಲಿ ರಕ್ತ", "ರಕ್ತ ಕೆಮ್ಮು", "blood in sputum", "coughing blood", "hemoptysis", "red blood in phlegm"
    ],
    "prominent_veins_on_calf": [
        "ಹಿಂಗಾಲಿನಲ್ಲಿ ಉಬ್ಬಿದ ರಕ್ತನಾಳಗಳು", "ಉಬ್ಬಿದ ಸಿರೆಗಳು", "prominent veins on calf", "spider veins", "twisted veins on legs", "engorged calf veins"
    ],
    "palpitations": [
        "ಎದೆ ಬಡಿತ ಹೆಚ್ಚಾಗುವುದು", "ಗುಂಡಿಗೆ ಢವಢವ", "palpitations", "heart racing", "fluttering heart", "skipped heartbeats", "pounding heart"
    ],
    "painful_walking": [
        "ನಡೆಯುವಾಗ ನೋವು", "ಕುಂಟುವುದು", "painful walking", "limping", "hurts to walk", "difficulty walking due to pain"
    ],
    "pus_filled_pimples": [
        "ಕೀವು ತುಂಬಿದ ಮೊಡವೆಗಳು", "ಮೊಡವೆ ಕೀವು", "pus filled pimples", "acne pustules", "pus pimples", "zits with pus", "boils on face"
    ],
    "blackheads": [
        "ಕಪ್ಪು ಚುಕ್ಕೆಗಳು", "ಬ್ಲ್ಯಾಕ್‌ಹೆಡ್ಸ್", "blackheads", "clogged pores", "comedones", "open comedones"
    ],
    "scurring": [
        "ಕಲೆಗಳು", "ಮೊಡವೆ ಕಲೆಗಳು", "scurring", "acne scars", "scarring", "pockmarks"
    ],
    "skin_peeling": [
        "ಚರ್ಮ ಸುಲಿಯುವುದು", "ಸಿಪ್ಪೆ ಏಳುವುದು", "skin peeling", "flaking skin", "desquamation", "peeling epidermis"
    ],
    "silver_like_dusting": [
        "ಬೆಳ್ಳಿಯಂತಹ ಚರ್ಮದ ಪುಡಿ", "ಚರ್ಮದ ಪದರ", "silver like dusting", "silvery scales", "psoriasis scales", "white flaky crusts"
    ],
    "small_dents_in_nails": [
        "ಉಗುರುಗಳಲ್ಲಿ ಸಣ್ಣ ಗುಳಿಗಳು", "small dents in nails", "nail pitting", "pitted fingernails"
    ],
    "inflammatory_nails": [
        "ಉಗುರುಗಳ ಊತ", "ಉಗುರು ಸುತ್ತು", "inflammatory nails", "nail inflammation", "swollen nail beds", "paronychia"
    ],
    "blister": [
        "ಗುಳ್ಳೆ", "ನೀರು ಗುಳ್ಳೆಗಳು", "ಬೊಕ್ಕೆಗಳು", "ಬೊಕ್ಕೆ", "gulle", "neeru gulle", "bokkegalu", "blister", "blisters", "skin vesicles", "fluid filled blisters", "bullae"
    ],
    "red_sore_around_nose": [
        "ಮೂಗಿನ ಸುತ್ತ ಕೆಂಪು ಹುಣ್ಣು", "red sore around nose", "sores near nostrils", "crusty nose sores"
    ],
    "yellow_crust_ooze": [
        "ಹಳದಿ ಕವಚದ ದ್ರವ ಸೋರುವಿಕೆ", "yellow crust ooze", "honey colored crust", "oozing golden crust", "impetigo crust"
    ]
}

# Red flag symptoms that require urgent / emergency medical care
RED_FLAGS = {
    "chest_pain",
    "breathlessness",
    "blood_in_sputum",
    "stomach_bleeding",
    "altered_sensorium",
    "coma",
    "weakness_of_one_body_side",
    "slurred_speech",
    "acute_liver_failure",
    "sunken_eyes",
}

# Rich Medical Knowledge Base for all 30 diseases
DISEASE_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "Common Cold & Flu": {
        "display_name": "Common Cold & Viral Flu",
        "description": "A viral infection of the upper respiratory tract causing nasal congestion, runny nose, throat irritation, cough, mild-to-high fever, chills, and fatigue.",
        "primary_symptoms": ["Runny nose", "Congestion", "Cough", "Throat irritation", "Sneezing", "Fever", "Chills", "Fatigue"],
        "home_remedies": [
            "Steam inhalation with 2 drops of eucalyptus oil or ajwain twice daily for 10 minutes.",
            "Gargle with warm salt water (1/2 tsp turmeric + 1/2 tsp rock salt) 3 times a day.",
            "Drink hot ginger-lemon-honey water 2-3 times daily to soothe throat lining.",
            "Ensure 8-9 hours of sound sleep and stay in a warm, well-ventilated room."
        ],
        "ayurvedic": [
            "Tulsi (Holy Basil) + Ginger + Black Pepper Kadha boiled in water twice daily.",
            "Sitopaladi Churna (1/2 tsp) mixed with pure honey twice a day for cough and congestion.",
            "Golden Milk (warm cow milk with 1/4 tsp organic turmeric and a pinch of black pepper) at bedtime."
        ],
        "diet_do": ["Warm vegetable soups, moong dal khichdi, hot herbal broths, fresh citrus fruits, warm water."],
        "diet_dont": ["Ice-cold beverages, dairy ice creams, deep-fried snacks, heavy cheese, refrigerated leftovers."],
        "precautions": [
            "Use a clean tissue or elbow while coughing and wash hands regularly.",
            "Avoid direct exposure to cold air drafts and air conditioners set below 24°C.",
            "Seek immediate medical help if fever crosses 102°F or breathlessness develops."
        ],
        "specialist": "General Physician / ENT Specialist"
    },
    "Typhoid Fever": {
        "display_name": "Typhoid Fever",
        "description": "A bacterial infection caused by Salmonella typhi, transmitted via contaminated food or water, causing sustained step-ladder fever, headache, stomach pain, extreme weakness, and gastrointestinal upset.",
        "primary_symptoms": ["High fever", "Chills", "Headache", "Fatigue", "Abdominal pain", "Nausea", "Vomiting", "Diarrhea or Constipation"],
        "home_remedies": [
            "Drink plenty of boiled and cooled water with Electrolytes / ORS throughout the day to prevent dehydration.",
            "Apply cool wet cloth compresses to forehead and limbs to control high fever spikes.",
            "Take strict bed rest for at least 7 to 10 days to aid body recovery."
        ],
        "ayurvedic": [
            "Giloy (Guduchi) Kwath (20ml twice daily) to build immunity and manage chronic pyrexia.",
            "Sudarshan Vati or Mahasudarshan Churna with lukewarm water (under practitioner guidance).",
            "Boiled water infused with dry ginger (Sonth) and coriander seeds (Dhaniya)."
        ],
        "diet_do": ["Very soft boiled rice, moong dal soup, boiled potatoes, soft bananas, coconut water, thin kanji."],
        "diet_dont": ["Raw salads, unpeeled fruits, spicy curries, high-fiber raw vegetables, oily fried foods, unpasteurized milk."],
        "precautions": [
            "Always drink purified/boiled water and eat only freshly cooked hot meals.",
            "Never stop prescribed antibiotics mid-way; complete the full medical course.",
            "Maintain strict hand hygiene after using the restroom."
        ],
        "specialist": "General Physician / Infectious Disease Specialist"
    },
    "Gastroenteritis (Food Poisoning / Stomach Infection)": {
        "display_name": "Gastroenteritis / Acute Food Poisoning",
        "description": "Inflammation of the stomach and intestines caused by food/water bacterial toxins or viral infection, leading to rapid vomiting, watery loose motions (diarrhoea), abdominal cramps, nausea, and dehydration.",
        "primary_symptoms": ["Vomiting", "Diarrhoea (Loose motion)", "Dehydration", "Sunken eyes", "Abdominal cramps", "Nausea", "Dry mouth"],
        "home_remedies": [
            "Drink Oral Rehydration Salts (ORS) or electrolyte water after EVERY loose stool to replace lost electrolytes.",
            "Sip tender coconut water, pomegranate juice, or rice congee (kanji) with a pinch of rock salt.",
            "Follow the BRAT diet (Bananas, Rice, Applesauce, Toast) once vomiting eases.",
            "Avoid solid food during active nausea; resume with small sips of clear fluids."
        ],
        "ayurvedic": [
            "Musta (Nutgrass) + Sonth (Dry ginger) boiled water for calming intestinal spasms.",
            "Kutajarishta (15ml with equal water twice daily after food) for soothing loose motions.",
            "Fresh churned buttermilk (Takra) tempered with roasted cumin (Jeera) and rock salt."
        ],
        "diet_do": ["Boiled rice with light curd, steamed mashed apples, barley water, clear broth, plain crackers."],
        "diet_dont": ["Dairy milk, cheese, spicy food, raw street food, aerated sodas, caffeine, sugary desserts."],
        "precautions": [
            "Watch out for severe dehydration: sunken eyes, dry tongue, no urine output for >6 hours.",
            "Wash hands thoroughly before handling food.",
            "Seek immediate emergency care if stools contain blood or high fever accompanies diarrhea."
        ],
        "specialist": "Gastroenterologist / General Physician"
    },
    "Dengue": {
        "display_name": "Dengue Fever",
        "description": "A mosquito-borne viral infection causing sudden high fever, intense retro-orbital eye pain, severe joint and muscle aches ('breakbone fever'), and characteristic skin rash with risk of low platelet count.",
        "primary_symptoms": ["High fever", "Pain behind eyes", "Muscle pain", "Joint pain", "Chills", "Skin rash", "Red spots", "Nausea"],
        "home_remedies": [
            "Aggressive oral hydration with 3-4 liters of fluid (tender coconut water, ORS, fresh fruit juices).",
            "Sponge the body with lukewarm water for fever control.",
            "Strict physical rest to preserve platelet energy."
        ],
        "ayurvedic": [
            "Papaya leaf extract (Carica papaya) 15-20ml twice daily to support healthy platelet count.",
            "Giloy (Guduchi) juice (20ml) with lukewarm water in the morning.",
            "Wheatgrass juice and pomegranate juice to boost hemoglobin and vitality."
        ],
        "diet_do": ["Coconut water, kiwi, dragon fruit, pomegranate, light vegetable soups, steamed porridge."],
        "diet_dont": ["Aspirin/Ibuprofen (strictly avoid as they increase bleeding risk), dark meat, heavy fried meals."],
        "precautions": [
            "Daily Complete Blood Count (CBC) monitoring is essential to track platelet count and hematocrit.",
            "Use mosquito repellents and bed nets to avoid spreading infection.",
            "Emergency warning signs: bleeding from gums/nose, persistent vomiting, severe abdominal pain."
        ],
        "specialist": "General Physician / Hematologist"
    },
    "Malaria": {
        "display_name": "Malaria",
        "description": "A parasitic infection transmitted by the bite of infected Anopheles mosquitoes, characterized by cyclic high fever with violent shivering, chills, profuse sweating, headache, and fatigue.",
        "primary_symptoms": ["High fever with chills", "Violent shivering", "Profuse sweating", "Headache", "Nausea", "Muscle pain", "Diarrhea"],
        "home_remedies": [
            "Keep warm during the cold/shivering phase with blankets; switch to light cotton clothing during the sweat phase.",
            "Sip warm water with electrolytes to replenish salt loss from intense sweating.",
            "Complete physical bed rest in a clean, quiet room."
        ],
        "ayurvedic": [
            "Sudarshan Ghanvati (1-2 tablets twice daily under medical guidance).",
            "Chirayata (Swertia chirata) and Giloy decoction for balancing Pitta-Kapha fever cycles.",
            "Tulsi leaf juice mixed with black pepper powder and honey."
        ],
        "diet_do": ["Warm moong dal khichdi, boiled apple, coconut water, fresh orange juice, steamed vegetables."],
        "diet_dont": ["Heavy spicy food, deep-fried snacks, sour pickles, unhygienic cold foods."],
        "precautions": [
            "Get a blood smear or rapid antigen test (Malarial Parasite MP test) confirmed immediately.",
            "Take prescribed antimalarial regimen strictly on time without missing doses.",
            "Use window screens and mosquito nets."
        ],
        "specialist": "General Physician / Infectious Disease Specialist"
    },
    "GERD (Acid Reflux)": {
        "display_name": "GERD (Gastroesophageal Reflux Disease)",
        "description": "A chronic digestive disorder where stomach acid repeatedly flows back into the esophagus, causing burning chest pain (heartburn), acidity, regurgitation, and throat irritation.",
        "primary_symptoms": ["Acidity", "Heartburn / Burning chest", "Stomach pain", "Cough", "Vomiting", "Tongue ulcers", "Sour burps"],
        "home_remedies": [
            "Chew 1 teaspoon of fennel seeds (saunf) after every major meal to stimulate digestion.",
            "Elevate the head of your bed by 6 inches to prevent nighttime acid backflow.",
            "Eat smaller, frequent meals instead of 2 large heavy meals, and finish dinner 3 hours before sleep."
        ],
        "ayurvedic": [
            "Amla (Indian Gooseberry) powder or juice in the morning on an empty stomach.",
            "Yashtimadhu (Licorice root) powder (1/2 tsp) with warm water to coat gastric mucosa.",
            "Shatavari Churna or Kamadudha Ras (Mukta yukta) for soothing excess Pitta acid."
        ],
        "diet_do": ["Oatmeal, cold skimmed milk, bananas, cucumbers, melons, steamed vegetables, ginger tea."],
        "diet_dont": ["Spicy chili curries, citrus vinegar, tomatoes, raw onions, garlic, chocolate, caffeine, alcohol."],
        "precautions": [
            "Do not lie down or recline immediately after eating.",
            "Wear loose, comfortable clothing around the abdomen.",
            "Consult a gastroenterologist if swallowing becomes painful or weight loss occurs."
        ],
        "specialist": "Gastroenterologist"
    },
    "Bronchial Asthma": {
        "display_name": "Bronchial Asthma",
        "description": "A chronic inflammatory condition of the airways resulting in episodic breathlessness, chest tightness, wheezing, and persistent coughing.",
        "primary_symptoms": ["Breathlessness", "Cough", "Mucoid sputum", "High fever / Chest tightness", "Fatigue"],
        "home_remedies": [
            "Sit upright and practice relaxed pursed-lip breathing during mild tightness.",
            "Drink warm ginger-clove tea to help dilate constricted airways.",
            "Use a HEPA air purifier in the bedroom and maintain humidity between 30-50%."
        ],
        "ayurvedic": [
            "Vasaka (Adhatoda vasica) syrup or leaf decoction for bronchodilation and phlegm clearance.",
            "Trikatu Churna (Sonth, Maricha, Pippali) with pure honey twice daily.",
            "Gentle chest massage with warm mustard oil infused with rock salt (Saindhava Lavana)."
        ],
        "diet_do": ["Warm light soups, steamed vegetables, turmeric honey water, garlic-infused warm water."],
        "diet_dont": ["Refrigerated cold drinks, curds at night, preserved fermented food, sulfur-dried fruits."],
        "precautions": [
            "Always carry your prescribed fast-acting rescue inhaler (Salbutamol) at all times.",
            "Avoid known allergen triggers: smoke, dust mites, pet dander, sudden temperature shifts.",
            "Seek emergency ER care immediately if inhaler fails to relieve breathlessness."
        ],
        "specialist": "Pulmonologist / Chest Physician"
    },
    "Diabetes": {
        "display_name": "Diabetes Mellitus",
        "description": "A metabolic disorder characterized by elevated blood glucose levels due to insufficient insulin secretion or insulin resistance, causing frequent urination, excessive thirst, and unexplained weight changes.",
        "primary_symptoms": ["Polyuria (Frequent urination)", "Excessive hunger", "Weight loss", "Blurred vision", "Fatigue", "Restlessness"],
        "home_remedies": [
            "Drink methi (fenugreek) water in the morning: soak 1 tbsp seeds in a glass of water overnight.",
            "Engage in a 30-40 minute brisk walk or aerobic activity daily to improve insulin sensitivity.",
            "Include cinnamon powder (1/4 tsp) in daily oatmeal or warm water."
        ],
        "ayurvedic": [
            "Jamun seed powder (1/2 tsp) + Karela (Bitter gourd) juice on an empty stomach.",
            "Gudmar (Gymnema sylvestre - 'sugar destroyer') churna twice daily with warm water.",
            "Nisha Amalaki (Turmeric + Amla combination) for protecting microvascular health."
        ],
        "diet_do": ["Whole grains (barley, quinoa, millet), leafy greens, bitter gourd, cucumbers, flaxseeds, sprouts."],
        "diet_dont": ["Refined white sugar, sweets, sweetened sodas, white bread, processed bakery items, fruit juices."],
        "precautions": [
            "Monitor fasting and post-prandial blood sugar levels regularly with a glucometer.",
            "Inspect feet daily for cuts, blisters, or numbness (diabetic foot care).",
            "Follow prescribed oral hypoglycemic agents or insulin strictly."
        ],
        "specialist": "Endocrinologist / Diabetologist"
    },
    "Hypertension (High Blood Pressure)": {
        "display_name": "Hypertension",
        "description": "A chronic medical condition in which blood pressure in the arteries is persistently elevated, increasing risk of cardiovascular events, stroke, and kidney damage.",
        "primary_symptoms": ["Headache", "Dizziness", "Lack of concentration", "Chest pain", "Loss of balance", "Palpitations"],
        "home_remedies": [
            "Follow the DASH diet and reduce daily sodium (salt) intake below 2 grams (half teaspoon).",
            "Practice 15 minutes of deep diaphragmatic breathing (Pranayama / Anulom Vilom) twice daily.",
            "Consume 1 clove of raw crushed garlic with warm water in the morning."
        ],
        "ayurvedic": [
            "Arjuna bark decoction (Arjuna Ksheerapaka) for strengthening heart muscles and vascular tone.",
            "Sarpagandha Vati (under strict Ayurvedic doctor supervision).",
            "Ashwagandha and Shankhpushpi for calming stress-induced hypertension."
        ],
        "diet_do": ["Potassium-rich foods (bananas, spinach, coconut water), oats, walnuts, beetroot juice, pomegranate."],
        "diet_dont": ["Excess table salt, salted pickles, canned processed soups, potato chips, red meat, alcohol, smoking."],
        "precautions": [
            "Log blood pressure readings twice weekly at the same time of day.",
            "Do not stop blood pressure medication abruptly as rebound hypertension can occur.",
            "Seek immediate medical attention if systolic BP exceeds 180 mmHg or chest pain occurs."
        ],
        "specialist": "Cardiologist / General Physician"
    },
    "Migraine": {
        "display_name": "Migraine Headache",
        "description": "A neurological condition causing intense, throbbing, pulsating headache typically on one side of the head, accompanied by visual disturbances (aura), nausea, and light/sound sensitivity.",
        "primary_symptoms": ["Throbbing headache", "Visual disturbances", "Blurred vision", "Nausea", "Stiff neck", "Irritability", "Excessive hunger"],
        "home_remedies": [
            "Rest in a pitch-dark, quiet room with an ice pack applied across the forehead or base of neck.",
            "Stay well-hydrated and avoid skipping regular meal schedules.",
            "Gently massage temples with peppermint essential oil diluted in coconut oil."
        ],
        "ayurvedic": [
            "Nasya therapy with Anu Taila (2 drops in each nostril in morning on empty stomach).",
            "Brahmi and Shankhpushpi tea to calm neurological hyperactivity.",
            "Pathyadi Kwath with lukewarm water for chronic vascular headaches."
        ],
        "diet_do": ["Magnesium-rich foods (almonds, pumpkin seeds, dark leafy greens), ginger water, fresh fruits."],
        "diet_dont": ["Aged cheese, chocolate, monosodium glutamate (MSG), processed cured meats, red wine, artificial sweeteners."],
        "precautions": [
            "Maintain a headache diary to identify individual triggers (bright lights, screen glare, lack of sleep).",
            "Avoid prolonged continuous screen time; follow the 20-20-20 rule.",
            "Consult a neurologist if migraine attacks occur more than twice a week."
        ],
        "specialist": "Neurologist"
    },
    "Allergy": {
        "display_name": "Allergic Rhinitis & Hypersensitivity",
        "description": "An immune system reaction to typically harmless environmental allergens (pollen, dust mites, mold, pet dander), causing continuous sneezing, watery eyes, runny nose, and chills.",
        "primary_symptoms": ["Continuous sneezing", "Watering from eyes", "Chills", "Shivering", "Runny nose", "Nasal congestion"],
        "home_remedies": [
            "Perform daily saline nasal irrigation (Neti pot with sterile saline water) to wash out allergen particles.",
            "Apply a cool compress over closed eyes to soothe burning and itching.",
            "Keep home windows closed during high pollen count hours."
        ],
        "ayurvedic": [
            "Haridra Khanda (1 tsp with warm milk twice daily) for systemic anti-allergic action.",
            "Tulsi, ginger, and black pepper herbal infusion with honey.",
            "Pratimarsha Nasya with pure sesame oil or cow's ghee."
        ],
        "diet_do": ["Warm cooked foods, turmeric, ginger, garlic, hot soups, vitamin C rich fruits (amla, oranges)."],
        "diet_dont": ["Cold foods, yogurt at night, chilled ice creams, fermented foods, stale leftovers."],
        "precautions": [
            "Wash bed linens and pillow covers weekly in hot water (>60°C) to kill dust mites.",
            "Wear a face mask when dusting, vacuuming, or gardening.",
            "Keep emergency antihistamines or Epipen ready if severe anaphylactic history exists."
        ],
        "specialist": "Allergist / Immunologist / ENT"
    },
    "Urinary Tract Infection (UTI)": {
        "display_name": "Urinary Tract Infection (UTI)",
        "description": "A bacterial infection in any part of the urinary system (kidneys, bladder, or urethra), characterized by burning sensation during urination, pelvic discomfort, frequent urination urge, and cloudy/foul-smelling urine.",
        "primary_symptoms": ["Burning micturition", "Bladder discomfort", "Continuous feel of urine", "Foul smell of urine", "Mild fever"],
        "home_remedies": [
            "Drink 3 to 4 liters of clean water daily to flush bacteria out of the urinary tract.",
            "Drink pure unsweetened cranberry juice to prevent bacteria from adhering to the bladder wall.",
            "Apply a warm heating pad to lower abdomen to soothe bladder cramping."
        ],
        "ayurvedic": [
            "Chandanadi Vati or Gokshuradi Guggulu with water for clearing urinary tract inflammation.",
            "Punarnavarishta (15ml with equal water twice daily after meals).",
            "Coriander seed (Dhaniya) cold infusion: soak 2 tbsp crushed seeds in water overnight, strain and drink."
        ],
        "diet_do": ["Watermelon, barley water, coconut water, cucumbers, probiotic yogurt, plenty of filtered water."],
        "diet_dont": ["Coffee, black tea, alcohol, spicy pepper foods, artificial sweeteners, carbonated sodas."],
        "precautions": [
            "Never hold urine for long durations; empty bladder completely when needed.",
            "Wipe from front to back after bowel movements to prevent bacterial spread.",
            "Consult a physician for urine routine and culture test before taking antibiotics."
        ],
        "specialist": "Urologist / Nephrologist"
    },
    "Fungal Infection": {
        "display_name": "Fungal Skin Infection (Tinea / Ringworm / Candidiasis)",
        "description": "A fungal overgrowth affecting keratinized skin tissue, leading to circular red itchy patches, nodal eruptions, peeling skin, and discolored patches in moist skin folds.",
        "primary_symptoms": ["Itching", "Skin rash", "Nodal skin eruptions", "Dischromic patches"],
        "home_remedies": [
            "Keep the affected area meticulously clean, dry, and ventilated at all times.",
            "Apply diluted tea tree oil (2 drops mixed in 1 tsp coconut oil) to affected skin twice daily.",
            "Use antifungal dusting powder (Clotrimazole) in skin folds, groin, and underarms."
        ],
        "ayurvedic": [
            "Neem leaf decoction wash for the affected skin areas.",
            "Khadirarishta (15ml with equal water twice daily) as a premier blood and skin purifier.",
            "Gandhak Rasayana tablets with lukewarm water."
        ],
        "diet_do": ["Neem, bitter gourd, turmeric, garlic, light home-cooked meals, green leafy vegetables."],
        "diet_dont": ["Excess sugar, pastries, fermented batter, sour curds, stale cheese, alcohol."],
        "precautions": [
            "Wear loose, breathable 100% cotton clothing; avoid synthetic tight underwear.",
            "Do not share personal towels, soaps, combs, or bedsheets with family members.",
            "Complete full course of topical antifungal cream for 2 weeks even after rash clears."
        ],
        "specialist": "Dermatologist"
    },
    "Chickenpox": {
        "display_name": "Chickenpox (Varicella)",
        "description": "A highly contagious viral infection caused by the Varicella-Zoster virus, causing an intensely itchy blister-like rash, mild to high fever, headache, and fatigue.",
        "primary_symptoms": ["Itching", "Skin rash", "Red spots over body", "Mild fever", "High fever", "Fatigue", "Headache"],
        "home_remedies": [
            "Add neem leaves to bath water or take an oatmeal bath to relieve intense itching.",
            "Apply calamine lotion gently with cotton balls onto skin spots to cool itching.",
            "Trim fingernails short and wear soft cotton gloves at night to prevent scratching scars."
        ],
        "ayurvedic": [
            "Neem leaf paste applied externally over crusted lesions.",
            "Mahamanjisthadi Kwath (15ml twice daily) to purify blood and accelerate skin healing.",
            "Sandalwood (Chandan) paste mixed with rose water applied gently for cooling sensation."
        ],
        "diet_do": ["Hydrating fluids, soft khichdi, tender coconut water, mashed fruits, vegetable broths."],
        "diet_dont": ["Spicy foods, citrus juices if mouth sores exist, hard crunchy chips, salty fried snacks."],
        "precautions": [
            "Isolate the patient until all blisters have completely dried and formed crusts/scabs (approx 7-10 days).",
            "Do NOT give Aspirin to children with chickenpox (risk of life-threatening Reye's syndrome).",
            "Consult a doctor immediately if blisters develop near the eyes or secondary bacterial infection occurs."
        ],
        "specialist": "Pediatrician / Dermatologist / General Physician"
    },
    "Jaundice": {
        "display_name": "Jaundice (Hyperbilirubinemia)",
        "description": "A yellowish pigmentation of the skin and whites of the eyes caused by elevated bilirubin levels due to liver inflammation, biliary obstruction, or red blood cell breakdown.",
        "primary_symptoms": ["Yellowish skin", "Yellowing of eyes", "Dark urine", "Vomiting", "Fatigue", "Abdominal pain", "Weight loss"],
        "home_remedies": [
            "Drink fresh sugarcane juice prepared hygienically with a dash of lime to provide quick liver energy.",
            "Drink radish leaf juice (freshly extracted) in the morning for liver detox.",
            "Ensure complete physical rest to allow liver cell regeneration."
        ],
        "ayurvedic": [
            "Bhumyamalaki (Phyllanthus niruri) powder (1/2 tsp) with warm water twice daily.",
            "Liv-52 or Punarnava Mandur tablets to support hepatocyte recovery.",
            "Arogyavardhini Vati under qualified Ayurvedic supervision."
        ],
        "diet_do": ["Boiled rice with light moong dal, coconut water, barley water, papaya, boiled sweet potatoes."],
        "diet_dont": ["Oils, butter, ghee, fried foods, spices, alcohol (strictly 0%), processed packaged foods."],
        "precautions": [
            "Monitor serum bilirubin and liver function tests (SGOT, SGPT, ALP) weekly.",
            "Ensure all drinking water is thoroughly boiled.",
            "Consult a hepatologist/gastroenterologist for ultrasound evaluation of liver and gall bladder."
        ],
        "specialist": "Gastroenterologist / Hepatologist"
    },
    "Hepatitis A": {
        "display_name": "Hepatitis A",
        "description": "A highly contagious viral liver infection spread through contaminated food or water, causing acute jaundice, dark urine, pale stools, nausea, vomiting, and fatigue.",
        "primary_symptoms": ["Yellowish skin", "Yellowing of eyes", "Dark urine", "Nausea", "Vomiting", "Abdominal pain", "Mild fever", "Diarrhea"],
        "home_remedies": [
            "Strict bed rest during the acute symptomatic phase.",
            "Small, frequent meals rich in natural simple carbohydrates to maintain glycogen stores.",
            "Sip clean boiled water and coconut water continuously."
        ],
        "ayurvedic": [
            "Katuki (Picrorhiza kurroa) powder with honey for restoring healthy bile flow.",
            "Bhumyamalaki and Kalmegh (Andrographis paniculata) decoctions for antiviral liver protection.",
            "Draksharishta (15ml with water after food) to combat debility."
        ],
        "diet_do": ["Steamed rice, boiled carrots, raisins, sugarcane juice, coconut water, thin kanji."],
        "diet_dont": ["Any fatty or oily food, meat, heavy dairy, alcohol, unhygienic raw street foods."],
        "precautions": [
            "Strict hand hygiene after toilet use and before meal prep to prevent transmission to others.",
            "Disinfect household utensils and surfaces.",
            "Avoid taking unnecessary paracetamol or hepatotoxic drugs without physician clearance."
        ],
        "specialist": "Hepatologist / Gastroenterologist"
    },
    "Arthritis": {
        "display_name": "Arthritis (Inflammatory Joint Disease)",
        "description": "Inflammation of one or more joints causing morning stiffness, joint swelling, tenderness, restricted movement, and painful walking.",
        "primary_symptoms": ["Joint pain", "Swelling joints", "Movement stiffness", "Painful walking", "Muscle weakness", "Stiff neck"],
        "home_remedies": [
            "Alternate warm compresses (to relax stiff joints) and cold packs (to reduce active joint swelling).",
            "Engage in gentle low-impact exercises like swimming, water aerobics, and joint mobility stretches.",
            "Add 1 tsp turmeric powder and 1/2 tsp ginger to daily cooking."
        ],
        "ayurvedic": [
            "Shallaki (Boswellia serrata) and Guggulu formulations (Yograj Guggulu) for reducing joint inflammation.",
            "Nirgundi Taila or Mahanarayan Taila gentle warm oil massage over painful joints.",
            "Rasnadi Kwath decoction with warm water twice daily."
        ],
        "diet_do": ["Omega-3 rich foods (walnuts, flaxseeds, chia seeds), berries, leafy greens, cherries, ginger tea."],
        "diet_dont": ["Nightshade vegetables in excess if sensitive (brinjal, tomatoes), refined sugars, red meat, saturated fats."],
        "precautions": [
            "Maintain a healthy body weight to minimize load and mechanical stress on weight-bearing joints.",
            "Avoid high-impact jumping or running during active flare-ups.",
            "Consult a rheumatologist for inflammatory markers (ESR, CRP, RA factor)."
        ],
        "specialist": "Rheumatologist / Orthopedic Specialist"
    },
    "Osteoarthritis": {
        "display_name": "Osteoarthritis (Degenerative Joint Disease)",
        "description": "A degenerative 'wear-and-tear' joint disorder where cartilage protective cushioning breaks down, causing severe knee/hip pain, bone friction, stiffness, and cracking sounds.",
        "primary_symptoms": ["Knee pain", "Hip joint pain", "Neck pain", "Joint pain", "Painful walking", "Swelling joints"],
        "home_remedies": [
            "Use knee braces or walking aids during long walks to unload joint stress.",
            "Apply warm sesame oil followed by a warm heating pad for 15 minutes twice a day.",
            "Strengthen quadriceps muscles with seated leg-lift exercises daily."
        ],
        "ayurvedic": [
            "Janu Basti (warm medicated oil pooling over knee joints) under Ayurvedic therapist care.",
            "Lakshadi Guggulu and Asthisanghata formulations for bone and cartilage support.",
            "Ashwagandha churna (1 tsp) with warm milk at night to strengthen muscular support."
        ],
        "diet_do": ["Calcium and Vitamin D rich foods (fortified dairy, sesame seeds, ragi/millet, green leafy vegetables)."],
        "diet_dont": ["Fried snacks, cold refrigerated drinks, excessively sour foods that aggravate Vata."],
        "precautions": [
            "Avoid sitting cross-legged on the floor or deep squatting.",
            "Wear cushioned orthopedic footwear with proper arch support.",
            "Consult an orthopedic specialist for X-ray assessment of joint space narrowing."
        ],
        "specialist": "Orthopedic Surgeon / Physiotherapist"
    },
    "Cervical Spondylosis": {
        "display_name": "Cervical Spondylosis",
        "description": "Age-related wear and tear affecting the cervical spinal disks in your neck, causing persistent neck pain, stiffness, radiating pain to arms, dizziness, and balance instability.",
        "primary_symptoms": ["Neck pain", "Back pain", "Dizziness", "Loss of balance", "Weakness in limbs"],
        "home_remedies": [
            "Perform gentle isometric neck strengthening exercises 3 times daily.",
            "Use a contoured orthopedic cervical pillow to maintain neutral spinal alignment during sleep.",
            "Apply moist heat to the neck muscles for 15 minutes before stretching."
        ],
        "ayurvedic": [
            "Greeva Basti (retaining warm medicated oil on cervical spine).",
            "Mahanarayan Taila or Ksheerabala Taila gentle neck massage.",
            "Trayodashanga Guggulu with lukewarm water."
        ],
        "diet_do": ["Warm nourishing soups, sesame seeds, almonds, dates, cow's ghee in moderation, milk."],
        "diet_dont": ["Dry snacks, stale fast food, cold carbonated drinks, gas-producing beans in excess."],
        "precautions": [
            "Maintain correct ergonomics: keep computer monitors at eye level; avoid 'text neck' looking down at phones.",
            "Never perform sudden violent neck jerking or cracking.",
            "Consult an orthopedic/spine specialist if numbness or tingling shoots down fingers."
        ],
        "specialist": "Orthopedic / Spine Specialist / Physiotherapist"
    },
    "Peptic Ulcer Disease": {
        "display_name": "Peptic Ulcer Disease",
        "description": "Sores or open lesions that develop on the inside lining of the stomach and upper small intestine, causing burning stomach pain, indigestion, vomiting, bloating, and loss of appetite.",
        "primary_symptoms": ["Abdominal pain", "Indigestion", "Vomiting", "Loss of appetite", "Passage of gases", "Internal itching"],
        "home_remedies": [
            "Drink cabbage juice (freshly pressed) 100ml daily; rich in glutamine which heals gastric mucosa.",
            "Sip lukewarm chamomile tea or licorice tea between meals.",
            "Eat small bland meals at fixed hours; never stay on an empty stomach for prolonged periods."
        ],
        "ayurvedic": [
            "Yashtimadhu (Licorice) + Shatavari root churna with milk.",
            "Kamadudha Ras (Mukta yukta) for reducing excess digestive fire (Tikshnagni).",
            "Aloe vera gel (2 tbsp pure inner leaf) on an empty stomach."
        ],
        "diet_do": ["Cooked oatmeal, bananas, melons, steamed squash, tender coconut water, almond milk."],
        "diet_dont": ["Chili powder, black pepper, coffee, alcohol, smoking, NSAID painkiller pills (like Ibuprofen/Diclofenac)."],
        "precautions": [
            "Check with a doctor for Helicobacter pylori bacterial testing.",
            "Seek immediate emergency care if vomit contains coffee-ground blood or stools appear black/tarry.",
            "Avoid late night eating."
        ],
        "specialist": "Gastroenterologist"
    },
    "Hemorrhoids (Piles)": {
        "display_name": "Hemorrhoids (Piles)",
        "description": "Swollen veins in the lowest part of your rectum and anus, causing rectal pain, anal itching, bleeding during bowel movements, and painful defecation.",
        "primary_symptoms": ["Pain in anal region", "Irritation in anus", "Pain during bowel movements", "Bloody stool", "Constipation"],
        "home_remedies": [
            "Take a warm Sitz bath for 15-20 minutes in a tub of warm water after bowel movements.",
            "Apply aloe vera gel or witch hazel extract gently to the anal opening for soothing relief.",
            "Increase dietary fiber and drink 3 liters of water to keep stools soft and effortless."
        ],
        "ayurvedic": [
            "Triphala Churna (1 tsp with warm water at bedtime) to regulate bowel movements naturally.",
            "Abhayarishta (20ml with equal water after lunch and dinner).",
            "Kasisadi Taila local application to reduce hemorrhoidal swelling."
        ],
        "diet_do": ["Psyllium husk (Isabgol), figs, prunes, papaya, flaxseeds, oats, leafy vegetables, lentils."],
        "diet_dont": ["Spicy chili curries, bakery bread, fast food, red meat, alcohol, sitting on toilet bowl >5 minutes."],
        "precautions": [
            "Never strain or push forcefully during bowel movements.",
            "Avoid sitting on hard surfaces for long continuous hours; use a donut cushion.",
            "Consult a colorectal surgeon or proctologist for clinical grading."
        ],
        "specialist": "Proctologist / General Surgeon"
    },
    "Varicose Veins": {
        "display_name": "Varicose Veins",
        "description": "Enlarged, swollen, and twisting veins, often appearing blue or dark purple on the calves and legs due to weakened valve function, causing aching pain, cramps, and swelling.",
        "primary_symptoms": ["Prominent veins on calf", "Swollen legs", "Swollen blood vessels", "Cramps", "Bruising", "Fatigue", "Obesity"],
        "home_remedies": [
            "Wear graduated compression stockings during the day when standing or walking.",
            "Elevate your legs above heart level for 15 minutes 3-4 times daily to facilitate venous return.",
            "Perform regular calf pump exercises (raising heels up and down) while seated or standing."
        ],
        "ayurvedic": [
            "Sahacharadi Taila gentle upward massage from ankles towards knees (never massage directly on bulges).",
            "Kaishore Guggulu for improving microcirculation.",
            "Manjistha decoction for blood purification and vascular wall health."
        ],
        "diet_do": ["Flavonoid-rich foods (berries, citrus fruits, bell peppers), high-fiber foods, buckwheat, chia seeds."],
        "diet_dont": ["High sodium foods, canned goods, processed sausages, alcohol, tight waistbands."],
        "precautions": [
            "Avoid standing or sitting in one position for more than 45 minutes continuously.",
            "Maintain an optimal body mass index (BMI).",
            "Consult a vascular surgeon if skin discoloration, ulcers, or severe swelling develops."
        ],
        "specialist": "Vascular Surgeon"
    },
    "Vertigo (BPPV)": {
        "display_name": "Vertigo (Benign Paroxysmal Positional Vertigo)",
        "description": "A disorder arising from the inner ear characterized by sudden episodes of a spinning sensation (the room spinning around you), loss of balance, unsteadiness, and nausea triggered by head position changes.",
        "primary_symptoms": ["Spinning movements", "Loss of balance", "Unsteadiness", "Nausea", "Vomiting", "Headache"],
        "home_remedies": [
            "Perform the Epley Maneuver or Brandt-Daroff exercises under guided medical demonstration.",
            "Change head positions slowly; pause a moment on the edge of the bed before standing up in the morning.",
            "Sleep with your head elevated on 2 pillows."
        ],
        "ayurvedic": [
            "Ashwagandha and Brahmi Ghrita to nourish nervous system pathways.",
            "Saraswatarishta (15ml with water after meals) for neural balance.",
            "Ginger tea to reduce associated nausea and vestibular spinning."
        ],
        "diet_do": ["Hydrating fluids, whole grains, nuts, ginger, vitamin D and B12 rich foods."],
        "diet_dont": ["Excess caffeine, high sodium foods, alcohol, sudden head jerks, bright flickering lights."],
        "precautions": [
            "Sit down immediately when dizzy to prevent falls and head trauma.",
            "Avoid driving or climbing ladders while vertigo episodes are active.",
            "Consult an ENT / Neurotologist for canalith repositioning."
        ],
        "specialist": "ENT Specialist / Neurologist"
    },
    "Hyperthyroidism": {
        "display_name": "Hyperthyroidism",
        "description": "An overactive thyroid gland producing excessive thyroid hormones (T3/T4), accelerating metabolism and causing rapid heartbeat, unexplained weight loss, heat intolerance, and mood swings.",
        "primary_symptoms": ["Fast heart rate", "Weight loss", "Mood swings", "Restlessness", "Excessive hunger", "Diarrhoea", "Muscle weakness", "Abnormal menstruation"],
        "home_remedies": [
            "Practice cooling breathing exercises (Sheetali and Sheetkari Pranayama) to regulate body heat.",
            "Drink fresh lemon balm tea to help calm an overstimulated nervous system.",
            "Ensure regular 8-hour sleep schedules to reduce adrenal strain."
        ],
        "ayurvedic": [
            "Shankhpushpi and Brahmi herbal preparations for calming autonomic overactivity.",
            "Kanchnar Guggulu (under Ayurvedic physician supervision).",
            "Amalaki Rasayana for reducing excess bodily Pitta."
        ],
        "diet_do": ["Cruciferous vegetables (broccoli, cauliflower, cabbage - contain natural goitrogens that temper excess thyroid hormone), berries, oats."],
        "diet_dont": ["Excessive iodized salt, seaweed (kelp), energy drinks, high caffeine, spicy heating foods."],
        "precautions": [
            "Get thyroid function tests (Free T3, Free T4, TSH) done every 6 to 8 weeks.",
            "Take prescribed antithyroid medications (Methimazole/PTU) diligently.",
            "Seek emergency care if heartbeat feels irregularly rapid or severe fever/agitation occurs (Thyroid Storm)."
        ],
        "specialist": "Endocrinologist"
    },
    "Hypothyroidism": {
        "display_name": "Hypothyroidism",
        "description": "An underactive thyroid gland unable to produce sufficient thyroid hormone, causing slowed metabolism, weight gain, fatigue, puffy face, dry skin, and cold intolerance.",
        "primary_symptoms": ["Weight gain", "Puffy face and eyes", "Lethargy", "Swollen extremeties", "Brittle nails", "Mood swings", "Abnormal menstruation", "Enlarged thyroid"],
        "home_remedies": [
            "Consume 2 Brazil nuts daily for natural selenium which supports thyroid hormone conversion.",
            "Engage in 30 minutes of aerobic exercise daily to stimulate sluggish metabolism.",
            "Drink warm coriander seed water in the morning on an empty stomach."
        ],
        "ayurvedic": [
            "Kanchnar Guggulu (2 tablets twice daily) for balancing thyroid gland enlargement.",
            "Ashwagandha root powder (1/2 tsp with warm milk) to stimulate thyroid secretion.",
            "Triphala and Guggulu to clear sluggish Kapha and metabolic toxins (Ama)."
        ],
        "diet_do": ["Iodized salt in moderation, coconut oil cooking, roasted pumpkin seeds, chia seeds, fresh vegetables, lentils."],
        "diet_dont": ["Excessive raw cruciferous vegetables (steam them instead), soy products, processed sugars, refined wheat."],
        "precautions": [
            "Take thyroid hormone replacement (Levothyroxine) on an empty stomach with plain water 1 hour before breakfast.",
            "Do not consume calcium or iron supplements within 4 hours of thyroid medication.",
            "Retest TSH levels every 3-6 months."
        ],
        "specialist": "Endocrinologist"
    },
    "Hypoglycemia (Low Blood Sugar)": {
        "display_name": "Hypoglycemia (Acute Low Blood Glucose)",
        "description": "A sudden drop in blood glucose levels below 70 mg/dL, causing trembling, palpitations, sweating, dizziness, tingling lips, blurred vision, and confusion.",
        "primary_symptoms": ["Sweating", "Palpitations", "Anxiety", "Excessive hunger", "Drying and tingling lips", "Blurred vision", "Fatigue", "Nausea"],
        "home_remedies": [
            "Follow the 15-15 Rule: Consume 15 grams of fast-acting simple carbohydrates (3-4 glucose tablets, half cup fruit juice, or 1 tbsp honey/sugar).",
            "Re-check blood glucose after 15 minutes; if still below 70 mg/dL, repeat with another 15g carbs.",
            "Once glucose stabilizes, eat a small snack with complex carbs and protein (crackers with peanut butter)."
        ],
        "ayurvedic": [
            "Eat small, frequent balanced meals containing whole grains and ghee to maintain steady blood glucose.",
            "Amla with honey in the morning for metabolic equilibrium.",
            "Ashwagandha to stabilize neuro-endocrine stress response."
        ],
        "diet_do": ["Complex carbohydrates with protein (oats with nuts, whole grain toast, lentils, brown rice, seeds)."],
        "diet_dont": ["Skipping meals, prolonged fasting without supervision, alcohol on empty stomach."],
        "precautions": [
            "Always carry glucose sweets, candy, or glucose sachets in your pocket/bag.",
            "Wear a medical alert bracelet if diabetic.",
            "If the person becomes unconscious or unable to swallow, administer emergency Glucagon and call ambulance immediately."
        ],
        "specialist": "Endocrinologist / Diabetologist"
    },
    "Drug Reaction": {
        "display_name": "Adverse Drug Reaction / Medication Allergy",
        "description": "An allergic or toxic adverse response of the body to a pharmaceutical medication, causing acute skin rashes, intense itching, burning urination, or gastrointestinal distress.",
        "primary_symptoms": ["Itching", "Skin rash", "Stomach pain", "Burning micturition", "Spotting urination"],
        "home_remedies": [
            "Immediately identify and discontinue the suspected culprit medication (after contacting your doctor).",
            "Apply cold compresses or calamine lotion to soothe itching skin.",
            "Drink plenty of water to assist kidneys in clearing drug metabolites."
        ],
        "ayurvedic": [
            "Manjisthadi Kwath for clearing systemic blood toxins.",
            "Sandalwood paste application on rash lesions.",
            "Coriander seed cold infusion to reduce burning sensations."
        ],
        "diet_do": ["Bland home-cooked meals, coconut water, fresh fruits, light soups."],
        "diet_dont": ["Spicy food, alcohol, self-prescribed over-the-counter painkillers or herbal cocktails."],
        "precautions": [
            "Seek IMMEDIATE emergency medical care if swelling of lips/tongue, wheezing, or difficulty breathing develops (Anaphylaxis).",
            "Carry a written medical alert card listing known drug allergies.",
            "Inform all future healthcare providers before receiving new prescriptions."
        ],
        "specialist": "Allergist / Immunologist / Dermatologist"
    },
    "Psoriasis": {
        "display_name": "Psoriasis",
        "description": "An autoimmune skin disease that speeds up the life cycle of skin cells, causing cells to build up rapidly on the surface and form thick, silvery scales and itchy, dry red patches.",
        "primary_symptoms": ["Skin rash", "Skin peeling", "Silver like dusting", "Small dents in nails", "Inflammatory nails", "Joint pain"],
        "home_remedies": [
            "Keep skin constantly moisturized with virgin coconut oil, aloe vera, or thick ointment right after bathing.",
            "Expose affected skin to gentle early morning sunlight for 10-15 minutes (Natural UV phototherapy).",
            "Add Epsom salt or colloidal oatmeal to lukewarm bathwater."
        ],
        "ayurvedic": [
            "Takradhara (pouring medicated buttermilk on forehead or skin) under Ayurvedic panchakarma.",
            "Khadira (Acacia catechu) and Neem preparations to purify blood and balance Pitta-Kapha.",
            "Psorolin ointment / Wrightia tinctoria (Swetha Kutaja) oil applied topically."
        ],
        "diet_do": ["Anti-inflammatory diet: turmeric, green vegetables, omega-3 flaxseeds, pumpkin seeds, carrots."],
        "diet_dont": ["Alcohol, smoking, red meat, dairy in excess, junk food, emotional stress."],
        "precautions": [
            "Never scratch or forcefully peel silvery scales (Koebner phenomenon causes new lesions).",
            "Use mild fragrance-free soaps and moisturizers.",
            "Consult a dermatologist for targeted topical, biological, or phototherapy treatments."
        ],
        "specialist": "Dermatologist / Rheumatologist"
    },
    "Acne": {
        "display_name": "Acne Vulgaris",
        "description": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells, causing whiteheads, blackheads, pus-filled pimples, and scarring.",
        "primary_symptoms": ["Pus filled pimples", "Skin rash", "Blackheads", "Scurring (Acne scars)"],
        "home_remedies": [
            "Wash face twice daily with a gentle, non-comedogenic foaming cleanser; avoid harsh scrubbing.",
            "Apply a small dab of diluted tea tree oil (5% concentration) to active pimples.",
            "Apply fresh aloe vera gel to reduce redness and promote scar healing."
        ],
        "ayurvedic": [
            "Neem and turmeric face mask mixed with rose water.",
            "Khadirarishta and Manjistha for internal systemic blood purification.",
            "Lodhra and Chandana (Sandalwood) paste for cooling inflamed pustules."
        ],
        "diet_do": ["Zinc and Vitamin A rich foods, plenty of water (8-10 glasses), green tea, fresh salads."],
        "diet_dont": ["High glycemic index foods, dairy milk in excess, whey protein supplements, deep fried oily snacks."],
        "precautions": [
            "Never pick, pop, or squeeze pimples to prevent bacterial spread and permanent scarring.",
            "Change pillowcases twice weekly and clean phone screens regularly.",
            "Consult a dermatologist for topical retinoids, salicylic acid, or benzoyl peroxide formulations."
        ],
        "specialist": "Dermatologist"
    },
    "Impetigo": {
        "display_name": "Impetigo (Contagious Bacterial Skin Infection)",
        "description": "A highly contagious bacterial skin infection (Staphylococcus/Streptococcus) causing red sores around the nose and mouth that rupture, ooze for a few days, and form characteristic honey-colored yellow crusts.",
        "primary_symptoms": ["Red sore around nose", "Yellow crust ooze", "Blister", "Skin rash", "High fever"],
        "home_remedies": [
            "Gently soak sores in warm soapy water to loosen and remove crusts before applying antibiotic ointment.",
            "Cover sores loosely with a sterile bandage to prevent spreading bacteria through touch.",
            "Wash patient's clothes, towels, and bed linen in hot water separately."
        ],
        "ayurvedic": [
            "Neem decoction wash for gentle antiseptic cleansing of sores.",
            "Turmeric and honey paste applied locally for antimicrobial action.",
            "Gandhak Rasayana tablets with water."
        ],
        "diet_do": ["Immunity-boosting fruits, warm vegetable soups, coconut water, turmeric milk."],
        "diet_dont": ["Refined sugars, unhygienic foods, shared utensils."],
        "precautions": [
            "Do not touch or scratch the sores; wash hands immediately if accidental contact occurs.",
            "Keep children home from school/daycare until 24 hours after starting prescription antibiotic treatment.",
            "Consult a doctor for prescription topical mupirocin or oral antibiotics."
        ],
        "specialist": "Dermatologist / Pediatrician"
    }
}


def parse_symptoms(text: str, valid_features: List[str]) -> Tuple[Dict[str, str], List[int]]:
    """
    Advanced NLP parser that matches natural language symptom text
    (English, Native Kannada ಕನ್ನಡ, and Romanized Kannada)
    against the 132 valid model features.
    """
    if not text:
        return {}, [0] * len(valid_features)

    # Normalize text
    cleaned = text.lower()
    cleaned = re.sub(r"[,/;&+।]", " , ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Pre-clean punctuation while strictly preserving Kannada Unicode block (\u0C80-\u0CFF)
    alphanumeric_clean = re.sub(r"[^a-zA-Z0-9\s_\u0C80-\u0CFF]", " ", cleaned)
    alphanumeric_clean = f" {alphanumeric_clean} "

    matched: Dict[str, str] = {}

    # 1. Match multi-word and Kannada synonyms first (ordered by phrase length descending)
    for symptom, phrases in SYMPTOM_SYNONYMS.items():
        col = f"has_{symptom}"
        if col not in valid_features:
            continue
        for phrase in sorted(phrases, key=len, reverse=True):
            p_clean = phrase.lower().strip()
            # For Kannada Unicode strings or general phrases, check substring with boundary safety
            if any('\u0c80' <= ch <= '\u0cff' for ch in p_clean):
                if p_clean in alphanumeric_clean:
                    matched[symptom] = col
                    break
            else:
                pattern = r"\b" + re.escape(p_clean) + r"\b"
                if re.search(pattern, alphanumeric_clean):
                    matched[symptom] = col
                    break

    # 2. Match exact feature tokens from valid_features
    for col in valid_features:
        sym = col.replace("has_", "")
        token_phrase = sym.replace("_", " ")
        if token_phrase in alphanumeric_clean and sym not in matched:
            matched[sym] = col

    # 3. Handle spoken disease names and colloquial combos in English and Kannada
    # Direct spoken disease name mappings to primary symptoms
    disease_spoken_map = {
        ("ಕಾಮಾಲೆ", "kaamale", "kamale", "jaundice"): ["yellowish_skin", "yellowing_of_eyes", "dark_urine", "fatigue"],
        ("ಮಲೇರಿಯಾ", "malaria", "maleriya"): ["high_fever", "chills", "shivering", "sweating", "muscle_pain"],
        ("ಡೆಂಗ್ಯೂ", "dengue", "dengu"): ["high_fever", "joint_pain", "headache", "pain_behind_the_eyes", "skin_rash"],
        ("ಟೈಫಾಯ್ಡ್", "typhoid", "typhos", "taayfaayid"): ["high_fever", "headache", "abdominal_pain", "fatigue", "chills", "nausea"],
        ("ಅಸ್ತಮಾ", "asthma", "asthama", "ದಮ್ಮು"): ["breathlessness", "cough", "mucoid_sputum", "fatigue"],
        ("ಮೈಗ್ರೇನ್", "migraine", "migren", "ಅರೆತಲೆನೋವು"): ["headache", "visual_disturbances", "blurred_and_distorted_vision", "acidity"],
        ("ಫುಡ್ ಪಾಯಿಸನಿಂಗ್", "food poisoning", "ಗ್ಯಾಸ್ಟ್ರೋ"): ["vomiting", "diarrhoea", "dehydration", "stomach_pain"],
        ("ಅಲರ್ಜಿ", "allergy", "alerji"): ["continuous_sneezing", "watering_from_eyes", "shivering", "chills"],
        ("ಮೂಲವ್ಯಾಧಿ", "piles", "hemorrhoids", "ಪೈಲ್ಸ್"): ["constipation", "pain_during_bowel_movements", "bloody_stool", "pain_in_anal_region"],
        ("ಸಂಧಿವಾತ", "arthritis", "ಆರ್ಥ್ರೈಟಿಸ್"): ["joint_pain", "swelling_joints", "painful_walking", "movement_stiffness"],
        ("ಚಿಕನ್‌ಪಾಕ್ಸ್", "chickenpox", "ಅಮ್ಮ"): ["skin_rash", "red_spots_over_body", "itching", "high_fever"],
        ("ಯೂರಿನ್ ಇನ್‌ಫೆಕ್ಷನ್", "uti", "ಮೂತ್ರ ಸೋಂಕು"): ["burning_micturition", "bladder_discomfort", "continuous_feel_of_urine"],
        ("ಡಯಾಬಿಟಿಸ್", "diabetes", "ಮಧುಮೇಹ", "ಶುಗರ್"): ["polyuria", "excessive_hunger", "fatigue", "weight_loss"],
        ("ಹೈಪರ್ ಟೆನ್ಷನ್", "hypertension", "ಬಿಪಿ", "ರಕ್ತದೊತ್ತಡ"): ["headache", "dizziness", "loss_of_balance", "chest_pain"],
        ("ಫಂಗಲ್", "fungal", "ಶಿಲೀಂಧ್ರ"): ["itching", "skin_rash", "nodal_skin_eruptions"]
    }

    for triggers, sym_list in disease_spoken_map.items():
        if any(tr in alphanumeric_clean for tr in triggers):
            for s in sym_list:
                col_name = f"has_{s}"
                if col_name in valid_features:
                    matched[s] = col_name

    # Cold / ನೆಗಡಿ
    if any(w in alphanumeric_clean for w in ["cold", "ನೆಗಡಿ", "ಶೀತ", "negadi", "sheetha"]) and not any(k in matched for k in ["runny_nose", "congestion", "chills"]):
        if "has_congestion" in valid_features:
            matched["congestion"] = "has_congestion"
        if "has_chills" in valid_features:
            matched["chills"] = "has_chills"
        if "has_runny_nose" in valid_features:
            matched["runny_nose"] = "has_runny_nose"

    # Body pain / ಮೈಕೈ ನೋವು
    if any(bp in alphanumeric_clean for bp in ["bodypain", "body pain", "body ache", "bodyaches", "myalgia", "body hurts", "ಮೈಕೈ ನೋವು", "ಮೈ ನೋವು", "ಮೈಕೈನೋವು", "mai kai novu", "maikainovu", "mai novu"]):
        if "has_muscle_pain" in valid_features:
            matched["muscle_pain"] = "has_muscle_pain"

    # Loose motion / ವಾಂತಿ ಬೇಧಿ / ಭೇದಿ
    if any(lm in alphanumeric_clean for lm in ["loose motion", "loose motions", "watery motion", "food poisoning", "ಭೇದಿ", "ಬೇಧಿ", "ಲೂಸ್ ಮೋಷನ್", "ವಾಂತಿ ಬೇಧಿ", "bhedi", "bedhi"]):
        if "has_diarrhoea" in valid_features:
            matched["diarrhoea"] = "has_diarrhoea"
        if ("vomit" in alphanumeric_clean or "ವಾಂತಿ" in alphanumeric_clean or "vaanti" in alphanumeric_clean) and "has_vomiting" in valid_features:
            matched["vomiting"] = "has_vomiting"

    # Bladder pain / ಉರಿ ಮೂತ್ರ / ಮೂತ್ರಕೋಶದ ನೋವು
    if any(bp in alphanumeric_clean for bp in ["bladder pain", "bladder ache", "bladder discomfort", "pelvic discomfort", "ಉರಿ ಮೂತ್ರ", "ಮೂತ್ರಕೋಶದ ನೋವು", "uri moothra"]):
        if "has_bladder_discomfort" in valid_features and "bladder_discomfort" not in matched:
            matched["bladder_discomfort"] = "has_bladder_discomfort"

    # Acidity / ಗ್ಯಾಸ್ಟ್ರಿಕ್
    if any(ac in alphanumeric_clean for ac in ["acidity", "ಆಸಿಡಿಟಿ", "ಎದೆ ಉರಿ", "ಗ್ಯಾಸ್ಟ್ರಿಕ್", "acid reflux", "ede uri"]):
        if "has_acidity" in valid_features:
            matched["acidity"] = "has_acidity"

    vector = [1 if col in matched.values() else 0 for col in valid_features]
    return matched, vector


# ---------------------------------------------------------
# Multi-Factor Clinical Parameter Knowledge Matrices
# ---------------------------------------------------------

DISEASE_CHRONICITY = {
    "Common Cold & Flu": "acute",
    "Gastroenteritis (Food Poisoning / Stomach Infection)": "acute",
    "Dengue": "acute",
    "Malaria": "acute",
    "Allergy": "acute",
    "Drug Reaction": "acute",
    "Chickenpox": "acute",
    "Migraine": "acute_episodic",
    "Vertigo (BPPV)": "acute_episodic",
    "Hypoglycemia (Low Blood Sugar)": "acute_metabolic",
    "Typhoid Fever": "subacute",
    "Urinary Tract Infection (UTI)": "subacute",
    "Impetigo": "subacute",
    "Jaundice": "subacute",
    "Bronchial Asthma": "chronic_with_flare",
    "GERD (Acid Reflux)": "chronic",
    "Peptic Ulcer Disease": "chronic",
    "Osteoarthritis": "chronic",
    "Cervical Spondylosis": "chronic",
    "Arthritis": "chronic",
    "Psoriasis": "chronic",
    "Fungal Infection": "chronic",
    "Acne": "chronic",
    "Hemorrhoids (Piles)": "chronic",
    "Hypertension (High Blood Pressure)": "chronic",
    "Diabetes": "chronic",
    "Hypothyroidism": "chronic",
    "Hyperthyroidism": "chronic",
}

COMORBIDITY_RISK_MAP = {
    "Diabetes": {
        "Urinary Tract Infection (UTI)": 1.55,
        "Fungal Infection": 1.60,
        "Hypoglycemia (Low Blood Sugar)": 1.95,
        "Diabetes": 1.85,
        "Gastroenteritis (Food Poisoning / Stomach Infection)": 1.30,
        "Impetigo": 1.25,
        "Hypertension (High Blood Pressure)": 1.35,
    },
    "Hypertension": {
        "Hypertension (High Blood Pressure)": 1.90,
        "Vertigo (BPPV)": 1.40,
        "Migraine": 1.30,
        "Cervical Spondylosis": 1.20,
    },
    "Asthma": {
        "Bronchial Asthma": 2.10,
        "Allergy": 1.45,
        "Common Cold & Flu": 1.30,
    },
    "Heart Disease": {
        "Hypertension (High Blood Pressure)": 1.60,
        "Bronchial Asthma": 1.35,
        "Vertigo (BPPV)": 1.35,
    },
    "Kidney Disease": {
        "Urinary Tract Infection (UTI)": 1.65,
        "Hypertension (High Blood Pressure)": 1.45,
        "Diabetes": 1.35,
    },
    "Thyroid Disorder": {
        "Hypothyroidism": 1.95,
        "Hyperthyroidism": 1.95,
        "Migraine": 1.25,
    }
}


def compute_clinical_factor_multiplier(
    disease_raw: str,
    patient_age: int = 28,
    patient_gender: str = "Male",
    days: int = 3,
    severity: int = 5,
    existing_conditions: List[str] = None
) -> Tuple[float, List[str]]:
    """
    Computes an epidemiological prior multiplier and clinical rationales
    based on patient age, gender, duration, severity, and pre-existing conditions.
    """
    if existing_conditions is None:
        existing_conditions = ["None"]

    standard_name = DISEASE_NAME_MAP.get(disease_raw, disease_raw)
    multiplier = 1.0
    reasons = []

    # 1. Comorbidity Multiplier
    active_conds = [c for c in existing_conditions if c and c != "None"]
    for cond in active_conds:
        if cond in COMORBIDITY_RISK_MAP:
            mapping = COMORBIDITY_RISK_MAP[cond]
            if standard_name in mapping:
                factor = mapping[standard_name]
                multiplier *= factor
                reasons.append(f"Pre-existing {cond} elevates clinical likelihood of {standard_name} (×{factor:.2f})")

    # 2. Duration / Chronicity Alignment
    chronicity = DISEASE_CHRONICITY.get(standard_name, "acute")
    if days <= 3:
        # Acute presentation
        if chronicity in ["acute", "acute_episodic", "acute_metabolic"]:
            multiplier *= 1.35
            reasons.append(f"Acute onset ({days} days) strongly supports rapid-presentation condition ({standard_name})")
        elif chronicity == "chronic":
            multiplier *= 0.65
    elif 4 <= days <= 14:
        # Subacute presentation
        if chronicity == "subacute":
            multiplier *= 1.45
            reasons.append(f"Duration of {days} days matches classic subacute progression of {standard_name}")
        elif chronicity == "chronic":
            multiplier *= 1.15
        elif chronicity == "acute":
            multiplier *= 0.95
    else:
        # Chronic presentation (> 14 days)
        if chronicity == "chronic":
            multiplier *= 1.70
            reasons.append(f"Prolonged persistence ({days} days) strongly points to chronic/degenerative condition ({standard_name})")
        elif chronicity == "acute":
            multiplier *= 0.35
            reasons.append(f"Timeline ({days} days) is unusually prolonged for typical self-limiting acute {standard_name}")

    # 3. Discomfort Severity Alignment (1-10)
    if severity >= 7:
        high_severity_diseases = [
            "Dengue", "Typhoid Fever", "Malaria", 
            "Gastroenteritis (Food Poisoning / Stomach Infection)", 
            "Peptic Ulcer Disease", "Migraine", "Bronchial Asthma"
        ]
        if standard_name in high_severity_diseases:
            multiplier *= 1.30
            reasons.append(f"High severity rating ({severity}/10) aligns with acute clinical distress of {standard_name}")
    elif severity <= 3:
        mild_diseases = ["Allergy", "Acne", "Common Cold & Flu", "Fungal Infection"]
        if standard_name in mild_diseases:
            multiplier *= 1.20

    # 4. Age Demographic Factor
    if patient_age >= 50:
        elderly_diseases = ["Osteoarthritis", "Cervical Spondylosis", "Hypertension (High Blood Pressure)", "Diabetes", "Arthritis"]
        if standard_name in elderly_diseases:
            multiplier *= 1.35
            reasons.append(f"Patient age ({patient_age} yrs) matches prime demographic profile for {standard_name}")
    elif patient_age <= 25:
        young_diseases = ["Acne", "Allergy", "Common Cold & Flu", "Chickenpox", "Migraine"]
        if standard_name in young_diseases:
            multiplier *= 1.25
            reasons.append(f"Younger patient age ({patient_age} yrs) correlates with high incidence of {standard_name}")

    # 5. Gender Demographic Factor
    if patient_gender == "Female":
        female_biased = ["Urinary Tract Infection (UTI)", "Hypothyroidism", "Migraine"]
        if standard_name in female_biased:
            multiplier *= 1.15
            reasons.append(f"Epidemiological prevalence of {standard_name} is higher in females")

    return max(0.05, multiplier), reasons


def predict_clinical_comprehensive(
    model,
    le,
    vector: List[int],
    matched_symptoms: List[str],
    patient_age: int = 28,
    patient_gender: str = "Male",
    days: int = 3,
    severity: int = 5,
    existing_conditions: List[str] = None
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Comprehensive multi-parameter clinical prediction engine.
    Combines:
    1. XGBoost ML multi-class inference probabilities
    2. Clinical symptom F1 overlap scores
    3. Bayesian patient-parameter multipliers (Duration, Severity, Age, Gender, Comorbidities)
    Returns:
    - Sorted list of (disease_name, calibrated_confidence_percentage)
    - List of clinical reasoning explanations for the top diagnosis
    """
    if existing_conditions is None:
        existing_conditions = ["None"]

    # 1. Base ML model probabilities
    try:
        raw_probs = model.predict_proba([vector])[0]
    except Exception:
        raw_probs = np.ones(len(le.classes_)) / len(le.classes_)

    classes = le.classes_
    num_classes = len(classes)

    # 2. Clinical Symptom Overlap Score
    overlap_scores = np.zeros(num_classes)
    active_set = set(matched_symptoms)

    if active_set:
        for idx, disease_name in enumerate(classes):
            profile = set(DISEASE_SYMPTOM_PROFILES.get(disease_name, []))
            if profile:
                matched_count = len(active_set.intersection(profile))
                precision = matched_count / len(active_set) if active_set else 0
                recall = matched_count / len(profile) if profile else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                overlap_scores[idx] = (precision * 0.65) + (f1_score * 0.35)

    if overlap_scores.sum() > 0:
        norm_overlap = overlap_scores / overlap_scores.sum()
    else:
        norm_overlap = raw_probs

    # 3. Compute Patient Parameter Multipliers for all candidate diseases
    clinical_multipliers = np.ones(num_classes)
    all_reasons = {}

    for idx, disease_name in enumerate(classes):
        mult, reasons = compute_clinical_factor_multiplier(
            disease_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            days=days,
            severity=severity,
            existing_conditions=existing_conditions
        )
        clinical_multipliers[idx] = mult
        all_reasons[disease_name] = reasons

    # 4. Multi-Factor Fusion (ML 30% + Symptom Overlap 40% + Patient Factors 30%)
    base_evidence = (0.35 * raw_probs) + (0.45 * norm_overlap)
    weighted_evidence = base_evidence * clinical_multipliers

    norm_final = weighted_evidence / weighted_evidence.sum()
    sorted_idx = norm_final.argsort()[::-1]

    results = []
    for idx in sorted_idx:
        results.append((classes[idx], float(norm_final[idx] * 100)))

    top_disease = classes[sorted_idx[0]]
    top_reasons = all_reasons.get(top_disease, [])

    return results, top_reasons


# Backward compatibility wrapper
def predict_clinical_hybrid(model, le, vector: List[int], matched_symptoms: List[str]) -> List[Tuple[str, float]]:
    results, _ = predict_clinical_comprehensive(model, le, vector, matched_symptoms)
    return results


def assess_urgency_comprehensive(
    active_symptoms: List[str],
    days: int,
    severity: int,
    confidence: float,
    patient_age: int = 28,
    existing_conditions: List[str] = None
) -> Tuple[str, str, List[str], int, str]:
    """
    Comprehensive clinical triage engine assessing patient vulnerability and urgency.
    Returns:
    - Urgency Level ('EMERGENCY', 'See Doctor Immediately', 'See Doctor Soon', 'Monitor 2-3 Days', 'Self-Care')
    - Action Advice string
    - Triage Reasons list
    - Vulnerability Score (0-100)
    - Vulnerability Tier ('Low Risk', 'Moderate Risk', 'Elevated Risk', 'High Critical Risk')
    """
    if existing_conditions is None:
        existing_conditions = ["None"]

    active_conds = [c for c in existing_conditions if c and c != "None"]
    reasons = []

    # Calculate Patient Vulnerability Index (0-100)
    vuln_score = 0
    # 1. Severity contribution (up to 40 pts)
    vuln_score += int(severity * 4)
    # 2. Duration contribution (up to 25 pts)
    vuln_score += min(25, int(days * 1.8))
    # 3. Comorbidity contribution (up to 25 pts)
    if active_conds:
        vuln_score += min(25, len(active_conds) * 12)
    # 4. Age factor (pediatric or elderly, up to 10 pts)
    if patient_age >= 60 or patient_age <= 6:
        vuln_score += 10

    vuln_score = min(100, vuln_score)

    if vuln_score >= 75:
        vuln_tier = "High Critical Risk"
    elif vuln_score >= 50:
        vuln_tier = "Elevated Risk"
    elif vuln_score >= 30:
        vuln_tier = "Moderate Risk"
    else:
        vuln_tier = "Low Risk"

    # Check for life-threatening Red Flags
    detected_red_flags = [s for s in active_symptoms if s in RED_FLAGS]
    if detected_red_flags:
        rf_names = [s.replace("_", " ").title() for s in detected_red_flags]
        return (
            "EMERGENCY",
            "Critical life-threatening symptom(s) detected. Transport patient immediately to the nearest Hospital Emergency Room.",
            [f"🚨 Critical Red Flag: {', '.join(rf_names)}", f"Patient Vulnerability Index: {vuln_score}/100"],
            100,
            "High Critical Risk"
        )

    # Escalated triage for High Comorbidity + High Severity / Duration
    if active_conds and (severity >= 7 or days >= 7):
        reasons.append(f"Pre-existing {', '.join(active_conds)} significantly amplifies acute complication risk")
        if severity >= 8 or days >= 10:
            return (
                "See Doctor Immediately",
                f"Due to pre-existing {', '.join(active_conds)} combined with high discomfort ({severity}/10) lasting {days} days, immediate clinical consultation and lab tests are required.",
                reasons + [f"Comorbidity present: {', '.join(active_conds)}", f"High severity ({severity}/10)", f"Duration: {days} days"],
                vuln_score,
                vuln_tier
            )
        else:
            return (
                "See Doctor Soon",
                f"Patient with {', '.join(active_conds)} should be evaluated by a healthcare professional within 24 hours to prevent secondary complications.",
                reasons + [f"Chronic comorbidity: {', '.join(active_conds)}", f"Symptom duration {days} days"],
                vuln_score,
                vuln_tier
            )

    # Clinical Threshold: If symptoms persist for 5-6+ days, escalate immediately to "See Doctor Immediately"
    if days >= 5:
        return (
            "See Doctor Immediately",
            f"Symptoms persisting for {days} days (>= 5 days threshold) require immediate professional medical evaluation and diagnostic testing. Do not rely solely on home care.",
            [f"🚨 Persistent symptom duration ({days} days >= 5 days threshold)", f"Patient Vulnerability Index: {vuln_score}/100 ({vuln_tier})"],
            max(vuln_score, 70),
            "Elevated Risk" if vuln_score < 75 else "High Critical Risk"
        )

    if severity >= 6:
        return (
            "See Doctor Soon",
            "Consult a physician within 24-48 hours for definitive diagnosis and targeted prescription due to elevated discomfort.",
            [f"Symptoms persisting for {days} day(s)", f"Discomfort level {severity}/10"],
            vuln_score,
            vuln_tier
        )

    if confidence < 35.0:
        return (
            "Monitor 2-3 Days",
            "Low model certainty with provided symptoms. Monitor closely and consult a clinic if symptoms persist.",
            [f"Prediction confidence is {confidence:.1f}%", f"Current duration: {days} days"],
            vuln_score,
            vuln_tier
        )

    if days <= 2 and severity <= 4 and not active_conds:
        return (
            "Self-Care",
            "Symptoms appear mild and recent in a patient with no high-risk comorbidities. Follow home remedies and hydration.",
            ["Recent onset (<= 2 days)", "Mild discomfort level", "No active comorbidities"],
            vuln_score,
            vuln_tier
        )

    return (
        "Monitor 2-3 Days",
        "Rest well, maintain hydration, and apply the recommended home remedies below. Re-evaluate if symptoms do not improve.",
        ["Moderate symptom profile", f"Duration: {days} days", f"Severity: {severity}/10"],
        vuln_score,
        vuln_tier
    )


# Backward compatibility wrapper
def assess_urgency(active_symptoms: List[str], days: int, severity: int, confidence: float) -> Tuple[str, str, List[str]]:
    urgency, advice, reasons, _, _ = assess_urgency_comprehensive(active_symptoms, days, severity, confidence)
    return urgency, advice, reasons


def get_comorbidity_tailored_precautions(existing_conditions: List[str], disease_name: str) -> List[str]:
    """
    Returns specific, actionable safety guidelines customized for the patient's pre-existing conditions.
    """
    if not existing_conditions or "None" in existing_conditions:
        return []

    tailored = []
    active_conds = [c for c in existing_conditions if c and c != "None"]

    if "Diabetes" in active_conds:
        tailored.append("🩺 **Diabetic Safety Alert:** Infections and fever increase insulin resistance. Check capillary blood glucose every 4-6 hours. Avoid home remedies with honey, jaggery, or high-sugar syrups.")
    if "Hypertension" in active_conds:
        tailored.append("💓 **Hypertension Caution:** Avoid excessive sodium intake or high-salt oral hydration without monitoring BP. Avoid herbal stimulants (e.g. licorice/mulethi) that can elevate blood pressure.")
    if "Asthma" in active_conds:
        tailored.append("🫁 **Asthma Bronchial Alert:** Keep rescue bronchodilator inhaler (e.g. Salbutamol) readily accessible. Avoid cold air, strong eucalyptus smoke/steam, or dust exposure.")
    if "Heart Disease" in active_conds:
        tailored.append("❤️ **Cardiovascular Protocol:** Monitor resting pulse rate. Do not consume excess fluid volume rapidly without consulting your cardiologist. Seek urgent care if chest heaviness occurs.")
    if "Kidney Disease" in active_conds:
        tailored.append("🧪 **Renal Precautions:** Strictly avoid over-the-counter NSAID pain killers (Ibuprofen, Diclofenac). Adhere to your nephrologist's daily fluid and potassium limit.")
    if "Thyroid Disorder" in active_conds:
        tailored.append("🦋 **Thyroid Management:** Take morning thyroid medication on an empty stomach with plain water, separated from herbal preparations by at least 2 hours.")

    return tailored



def get_remedies_for_disease(disease_name: str) -> Dict[str, Any]:
    """
    Returns full clinical and home remedy guidance for any disease name.
    """
    standard_name = DISEASE_NAME_MAP.get(disease_name, disease_name)
    if standard_name in DISEASE_KNOWLEDGE:
        return DISEASE_KNOWLEDGE[standard_name]

    for key, data in DISEASE_KNOWLEDGE.items():
        if key.lower() == disease_name.lower() or data.get("display_name", "").lower() == disease_name.lower():
            return data

    return {
        "display_name": disease_name,
        "description": f"Clinical condition identified as {disease_name}.",
        "primary_symptoms": [],
        "home_remedies": [
            "Drink plenty of boiled and filtered water to maintain optimal hydration.",
            "Ensure 8 hours of restful sleep and avoid physical or mental exertion.",
            "Eat light, easily digestible home-cooked meals (e.g. khichdi, clear soups)."
        ],
        "ayurvedic": [
            "Drink warm herbal infusion with ginger, tulsi, and black pepper.",
            "Take warm golden turmeric milk before bedtime.",
            "Consult a certified Ayurvedic physician for personalized Dosha analysis."
        ],
        "diet_do": ["Fresh fruits, warm vegetables, moong dal soup, plenty of clean fluids."],
        "diet_dont": ["Heavy fried food, excess sugar, cold drinks, unhygienic raw food."],
        "precautions": [
            "Monitor body temperature and symptoms daily.",
            "Consult a licensed doctor for accurate diagnostic tests.",
            "Do not self-prescribe antibiotics or strong medications."
        ],
        "specialist": "General Physician"
    }


# =========================================================================
# KANNADA (ಕನ್ನಡ) CLINICAL KNOWLEDGE BASE & TRANSLATION ENGINE
# =========================================================================

# Clear, layman-friendly symptom names in Kannada with English subtitles
SYMPTOM_NAMES_KN: Dict[str, str] = {
    "itching": "ತುರಿಕೆ / ನವೆ (Itching)",
    "skin_rash": "ಚರ್ಮದ ದದ್ದು / ಗುಳ್ಳೆ (Skin Rash)",
    "nodal_skin_eruptions": "ಗಂಟು ಗುಳ್ಳೆಗಳು (Nodal Skin Eruptions)",
    "continuous_sneezing": "ನಿರಂತರ ಸೀನು (Continuous Sneezing)",
    "shivering": "ಚಳಿ ನಡುಕ (Shivering)",
    "chills": "ವಿಪರೀತ ಚಳಿ (Chills)",
    "joint_pain": "ಕೀಲು ನೋವು (Joint Pain)",
    "stomach_pain": "ಹೊಟ್ಟೆ ನೋವು (Stomach Pain)",
    "acidity": "ಅಸಿಡಿಟಿ / ಎದೆ ಉರಿ (Acidity)",
    "ulcers_on_tongue": "ನಾಲಿಗೆ ಹುಣ್ಣು (Tongue Ulcers)",
    "muscle_wasting": "ಮಾಂಸಖಂಡ ಕ್ಷೀಣತೆ (Muscle Wasting)",
    "vomiting": "ವಾಂತಿ (Vomiting)",
    "burning_micturition": "ಉರಿ ಮೂತ್ರ (Burning Urination)",
    "spotting_urination": "ಮೂತ್ರದಲ್ಲಿ ರಕ್ತದ ಕಲೆ (Spotting Urination)",
    "fatigue": "ಸುಸ್ತು / ಆಯಾಸ (Fatigue)",
    "weight_gain": "ತೂಕ ಹೆಚ್ಚಳ (Weight Gain)",
    "anxiety": "ಆತಂಕ / ಭಯ (Anxiety)",
    "cold_hands_and_feets": "ಕೈಕಾಲು ತಣ್ಣಗಾಗುವುದು (Cold Hands & Feet)",
    "mood_swings": "ಮನಸ್ಥಿತಿ ಬದಲಾವಣೆ (Mood Swings)",
    "weight_loss": "ತೂಕ ಇಳಿಕೆ (Weight Loss)",
    "restlessness": "ಚಡಪಡಿಕೆ / ಅಸಮಾಧಾನ (Restlessness)",
    "lethargy": "ಜಡತ್ವ / ಆಲಸ್ಯ (Lethargy)",
    "patches_in_throat": "ಗಂಟಲಿನಲ್ಲಿ ಬಿಳಿ ಕಲೆಗಳು (Patches in Throat)",
    "irregular_sugar_level": "ಸಕ್ಕರೆ ಮಟ್ಟದಲ್ಲಿ ಏರಿಳಿತ (Irregular Sugar Level)",
    "cough": "ಕೆಮ್ಮು (Cough)",
    "high_fever": "ತೀವ್ರ ಜ್ವರ (High Fever)",
    "sunken_eyes": "ಗುಳಿಬಿದ್ದ ಕಣ್ಣುಗಳು (Sunken Eyes)",
    "breathlessness": "ಉಸಿರಾಟದ ತೊಂದರೆ / ದಮ್ಮು (Breathlessness)",
    "sweating": "ಅತಿಯಾದ ಬೆವರು (Sweating)",
    "dehydration": "ನಿರ್ಜಲೀಕರಣ / ಬಾಯಾರಿಕೆ (Dehydration)",
    "indigestion": "ಅಜೀರ್ಣ / ಹೊಟ್ಟೆ ಉಬ್ಬರ (Indigestion)",
    "headache": "ತಲೆನೋವು (Headache)",
    "yellowish_skin": "ಹಳದಿ ಚರ್ಮ (Yellowish Skin)",
    "dark_urine": "ಗಾಢ ಹಳದಿ ಮೂತ್ರ (Dark Urine)",
    "nausea": "ವಾಕರಿಕೆ / ವಾಂತಿ ಬರುವ ಭಾವನೆ (Nausea)",
    "loss_of_appetite": "ಹಸಿವಿಲ್ಲದಿರುವುದು (Loss of Appetite)",
    "pain_behind_the_eyes": "ಕಣ್ಣಿನ ಹಿಂಭಾಗದ ನೋವು (Pain Behind Eyes)",
    "back_pain": "ಬೆನ್ನು ನೋವು / ಸೊಂಟ ನೋವು (Back Pain)",
    "constipation": "ಮಲಬದ್ಧತೆ (Constipation)",
    "abdominal_pain": "ಹೊಟ್ಟೆ ನೋವು (Abdominal Pain)",
    "diarrhoea": "ಭೇದಿ / ಲೂಸ್ ಮೋಷನ್ (Diarrhoea)",
    "mild_fever": "ಸೌಮ್ಯ ಜ್ವರ (Mild Fever)",
    "yellow_urine": "ಹಳದಿ ಮೂತ್ರ (Yellow Urine)",
    "yellowing_of_eyes": "ಕಣ್ಣು ಹಳದಿಯಾಗುವುದು (Yellowing of Eyes)",
    "acute_liver_failure": "ಯಕೃತ್ತಿನ ತೊಂದರೆ (Liver Failure Risk)",
    "fluid_overload": "ದೇಹದಲ್ಲಿ ನೀರು ಶೇಖರಣೆ (Fluid Overload)",
    "swelling_of_stomach": "ಹೊಟ್ಟೆ ಊತ (Swelling of Stomach)",
    "swelled_lymph_nodes": "ಗಂಟಲಿನ ಗ್ರಂಥಿಗಳ ಊತ (Swelled Lymph Nodes)",
    "malaise": "ಅಸ್ವಸ್ಥತೆ / ನಿಶ್ಯಕ್ತಿ (Malaise)",
    "blurred_and_distorted_vision": "ಕಣ್ಣು ಮಸುಕಾಗುವುದು (Blurred Vision)",
    "phlegm": "ಕಫ (Phlegm)",
    "throat_irritation": "ಗಂಟಲು ಕೆರೆತ / ಕಿರಿಕಿರಿ (Throat Irritation)",
    "redness_of_eyes": "ಕಣ್ಣು ಕೆಂಪಾಗುವುದು (Redness of Eyes)",
    "sinus_pressure": "ಸೈನಸ್ ಒತ್ತಡ / ಮೂಗು ಕಟ್ಟುವುದು (Sinus Pressure)",
    "runny_nose": "ಮೂಗು ಸೋರುವುದು (Runny Nose)",
    "congestion": "ಎದೆ / ಮೂಗು ಕಟ್ಟುವುದು (Congestion)",
    "chest_pain": "ಎದೆ ನೋವು (Chest Pain)",
    "weakness_in_limbs": "ಕೈಕಾಲುಗಳ ನಿಶ್ಯಕ್ತಿ (Weakness in Limbs)",
    "fast_heart_rate": "ವೇಗದ ಹೃದಯ ಬಡಿತ (Fast Heart Rate)",
    "pain_during_bowel_movements": "ಮಲವಿಸರ್ಜನೆಯಲ್ಲಿ ನೋವು (Pain during Stool)",
    "pain_in_anal_region": "ಗುದದ್ವಾರದ ನೋವು (Pain in Anal Region)",
    "bloody_stool": "ಮಲದಲ್ಲಿ ರಕ್ತ (Bloody Stool)",
    "irritation_in_anus": "ಗುದದ್ವಾರದಲ್ಲಿ ತುರಿಕೆ / ಉರಿ (Anal Irritation)",
    "neck_pain": "ಕುತ್ತಿಗೆ ನೋವು (Neck Pain)",
    "dizziness": "ತಲೆ ತಿರುಗುವುದು (Dizziness)",
    "cramps": "ಸ್ನಾಯು ಸೆಳೆತ (Cramps)",
    "bruising": "ರಕ್ತ ಹೆಪ್ಪುಗಟ್ಟುವಿಕೆ / ಮೂಗೇಟು (Bruising)",
    "obesity": "ಬೊಜ್ಜು / ಅತಿಯಾದ ತೂಕ (Obesity)",
    "swollen_legs": "ಕಾಲುಗಳ ಊತ (Swollen Legs)",
    "swollen_blood_vessels": "ರಕ್ತನಾಳಗಳ ಊತ (Swollen Blood Vessels)",
    "puffy_face_and_eyes": "ಮುಖ ಮತ್ತು ಕಣ್ಣುಗಳ ಊತ (Puffy Face & Eyes)",
    "enlarged_thyroid": "ಗಂಟಲು ಗ್ರಂಥಿ ಊತ (Enlarged Thyroid)",
    "brittle_nails": "ಉಗುರುಗಳು ಒಡೆಯುವುದು (Brittle Nails)",
    "swollen_extremeties": "ಕೈಕಾಲುಗಳ ಊತ (Swollen Extremities)",
    "excessive_hunger": "ಅತಿಯಾದ ಹಸಿವು (Excessive Hunger)",
    "extra_marital_contacts": "ಲೈಂಗಿಕ ಸಂಪರ್ಕ ಇತಿಹಾಸ (Extra Contacts)",
    "drying_and_tingling_lips": "ತುಟಿ ಒಣಗುವುದು / ಜುಮ್ಮೆನ್ನುವುದು (Drying Lips)",
    "slurred_speech": "ಮಾತು ತೊದಲಿಸುವುದು (Slurred Speech)",
    "knee_pain": "ಮೊಣಕಾಲು ನೋವು (Knee Pain)",
    "hip_joint_pain": "ಸೊಂಟದ ಕೀಲು ನೋವು (Hip Joint Pain)",
    "muscle_weakness": "ಸ್ನಾಯು ದೌರ್ಬಲ್ಯ (Muscle Weakness)",
    "stiff_neck": "ಕುತ್ತಿಗೆ ಬಿಗಿತ (Stiff Neck)",
    "swelling_joints": "ಕೀಲುಗಳಲ್ಲಿ ಊತ (Swelling Joints)",
    "movement_stiffness": "ಚಲನೆಗೆ ಕಷ್ಟ / ಬಿಗಿತ (Movement Stiffness)",
    "spinning_movements": "ತಲೆ ಸುತ್ತುವ ಅನುಭವ (Spinning Movements)",
    "loss_of_balance": "ಸಮತೋಲನ ತಪ್ಪುವುದು (Loss of Balance)",
    "unsteadiness": "ನಿಲ್ಲಲು ಅಸ್ಥಿರತೆ (Unsteadiness)",
    "weakness_of_one_body_side": "ದೇಹದ ಒಂದು ಭಾಗದ ನಿಶ್ಯಕ್ತಿ (One-sided Weakness)",
    "loss_of_smell": "ವಾಸನೆ ತಿಳಿಯದಿರುವುದು (Loss of Smell)",
    "bladder_discomfort": "ಮೂತ್ರಕೋಶದ ಕಿರಿಕಿರಿ / ನೋವು (Bladder Discomfort)",
    "foul_smell_of_urine": "ಮೂತ್ರದ ದುರ್ವಾಸನೆ (Foul Smell of Urine)",
    "continuous_feel_of_urine": "ಸದಾ ಮೂತ್ರ ಬಂದಂತಾಗುವುದು (Frequent Urine Urge)",
    "passage_of_gases": "ಹೊಟ್ಟೆಯಲ್ಲಿ ಗ್ಯಾಸ್ / ವಾಯು (Gas / Flatulence)",
    "internal_itching": "ಒಳಗಿನ ತುರಿಕೆ (Internal Itching)",
    "toxic_look_(typhos)": "ವಿಪರೀತ ಅಸ್ವಸ್ಥ ಮುಖಭಾವ (Toxic Appearance)",
    "depression": "ಖಿನ್ನತೆ / ಬೇಸರ (Depression)",
    "irritability": "ಕಿರಿಕಿರಿ / ಸಿಟ್ಟು (Irritability)",
    "muscle_pain": "ಮೈಕೈ ನೋವು (Muscle Pain)",
    "altered_sensorium": "ಅರೆಪ್ರಜ್ಞಾವಸ್ಥೆ (Altered Sensorium)",
    "red_spots_over_body": "ಮೈಮೇಲೆ ಕೆಂಪು ಕಲೆಗಳು (Red Spots on Body)",
    "belly_pain": "ಹೊಟ್ಟೆ ನೋವು (Belly Pain)",
    "abnormal_menstruation": "ಅನಿಯಮಿತ ಮುಟ್ಟು / ಋತುಸ್ರಾವ (Abnormal Menstruation)",
    "dischromic__patches": "ಚರ್ಮದ ಬಿಳಿ/ಕಪ್ಪು ಕಲೆಗಳು (Discolored Patches)",
    "watering_from_eyes": "ಕಣ್ಣಿನಲ್ಲಿ ನೀರು ಬರುವುದು (Watering Eyes)",
    "increased_appetite": "ಹೆಚ್ಚಿದ ಹಸಿವು (Increased Appetite)",
    "polyuria": "ಅತಿಯಾದ ಮೂತ್ರ ವಿಸರ್ಜನೆ (Frequent Urination)",
    "family_history": "ಕುಟುಂಬದ ಕಾಯಿಲೆ ಇತಿಹಾಸ (Family History)",
    "mucoid_sputum": "ದಪ್ಪ ಕಫ (Mucoid Sputum)",
    "rusty_sputum": "ಕಂದು/ಕೆಂಪು ಕಫ (Rusty Sputum)",
    "lack_of_concentration": "ಏಕಾಗ್ರತೆಯ ಕೊರತೆ (Lack of Concentration)",
    "visual_disturbances": "ದೃಷ್ಟಿ ದೋಷ / ಬೆಳಕಿನ ಕಿರಿಕಿರಿ (Visual Disturbances)",
    "receiving_blood_transfusion": "ರಕ್ತ ವರ್ಗಾವಣೆ ಇತಿಹಾಸ (Blood Transfusion)",
    "receiving_unsterile_injections": "ಅಸುರಕ್ಷಿತ ಸೂಜಿ ಬಳಕೆ (Unsterile Injections)",
    "coma": "ಪ್ರಜ್ಞಾಹೀನತೆ (Coma)",
    "stomach_bleeding": "ಹೊಟ್ಟೆಯಲ್ಲಿ ರಕ್ತಸ್ರಾವ (Stomach Bleeding)",
    "distention_of_abdomen": "ಹೊಟ್ಟೆ ಉಬ್ಬರ (Abdominal Distention)",
    "history_of_alcohol_consumption": "ಮದ್ಯಪಾನದ ಇತಿಹಾಸ (Alcohol History)",
    "blood_in_sputum": "ಕಫದಲ್ಲಿ ರಕ್ತ (Blood in Sputum)",
    "prominent_veins_on_calf": "ಕಾಲಿನಲ್ಲಿ ಉಬ್ಬಿದ ರಕ್ತನಾಳಗಳು (Prominent Calf Veins)",
    "palpitations": "ಎದೆಬಡಿತ ಹೆಚ್ಚಾಗುವುದು (Palpitations)",
    "painful_walking": "ನಡೆಯುವಾಗ ನೋವು (Painful Walking)",
    "pus_filled_pimples": "ಕೀವು ತುಂಬಿದ ಮೊಡವೆ (Pus Filled Pimples)",
    "blackheads": "ಕಪ್ಪು ಕಲೆಗಳು / ಬ್ಲ್ಯಾಕ್‌ಹೆಡ್ಸ್ (Blackheads)",
    "scurring": "ಮೊಡವೆಯ ಕಲೆಗಳು (Scars)",
    "skin_peeling": "ಚರ್ಮ ಸುಲಿಯುವುದು (Skin Peeling)",
    "silver_like_dusting": "ಬೆಳ್ಳಿಯಂತಹ ಹುರುಪೆ (Silver Dusting)",
    "small_dents_in_nails": "ಉಗುರಿನಲ್ಲಿ ಸಣ್ಣ ಗುಳಿಗಳು (Dents in Nails)",
    "inflammatory_nails": "ಉಗುರುಗಳ ಊತ (Inflammatory Nails)",
    "blister": "ನೀರಿನ ಗುಳ್ಳೆ (Blister)",
    "red_sore_around_nose": "ಮೂಗಿನ ಸುತ್ತ ಕೆಂಪು ಹುಣ್ಣು (Red Sore Around Nose)",
    "yellow_crust_ooze": "ಹಳದಿ ಕೀವು ಸ್ರವಿಸುವಿಕೆ (Yellow Crust Ooze)"
}

# Comprehensive Kannada Medical Data for all 30 diseases
DISEASE_KNOWLEDGE_KN: Dict[str, Dict[str, Any]] = {
    "Common Cold & Flu": {
        "display_name_kn": "ಸಾಮಾನ್ಯ ಶೀತ ಮತ್ತು ಜ್ವರ (Common Cold & Flu)",
        "description_kn": "ಇದು ವೈರಲ್ ಸೋಂಕಿನಿಂದ ಉಂಟಾಗುವ ಸಾಮಾನ್ಯ ಕಾಯಿಲೆಯಾಗಿದ್ದು, ಮೂಗು ಸೋರುವುದು, ಸೀನು, ಗಂಟಲು ಕೆರೆತ, ಸೌಮ್ಯ ಜ್ವರ ಮತ್ತು ಮೈಕೈ ನೋವು ಉಂಟುಮಾಡುತ್ತದೆ. ಇದು ಸಾಮಾನ್ಯವಾಗಿ 3 ರಿಂದ 5 ದಿನಗಳಲ್ಲಿ ಸರಿಯಾದ ವಿಶ್ರಾಂತಿ ಮತ್ತು ಮನೆಮದ್ದಿನಿಂದ ಗುಣವಾಗುತ್ತದೆ.",
        "specialist_kn": "ಸಾಮಾನ್ಯ ವೈದ್ಯರು (General Physician)",
        "verdict_doctor_kn": "ಜ್ವರ ಅಥವಾ ಕೆಮ್ಮು 5 ದಿನಗಳಿಗಿಂತ ಹೆಚ್ಚು ಮುಂದುವರಿದರೆ ಅಥವಾ ಉಸಿರಾಟ ಕಷ್ಟವಾದರೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಸಾಕಷ್ಟು ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ, ಬೆಚ್ಚಗಿನ ನೀರು, ತುಳಸಿ-ಶುಂಠಿ ಕಷಾಯ ಕುಡಿಯಿರಿ ಮತ್ತು ಹಬೆ (ಸ್ಟೀಮ್) ತೆಗೆದುಕೊಳ್ಳಿ.",
        "home_remedies_kn": [
            "ದಿನಕ್ಕೆ 2-3 ಬಾರಿ ಬಿಸಿ ನೀರಿನ ಹಬೆ (ಸ್ಟೀಮ್) ತೆಗೆದುಕೊಳ್ಳಿ.",
            "ಬೆಚ್ಚಗಿನ ಶುಂಠಿ, ತುಳಸಿ ಮತ್ತು ಮೆಣಸಿನ ಕಷಾಯ ಕುಡಿಯಿರಿ.",
            "ಉಗುರುಬೆಚ್ಚಗಿನ ಉಪ್ಪು ನೀರಿನಿಂದ ಗಂಟಲು ಮುಕ್ಕಳಿಸಿ (ಗಾರ್ಗಲ್ ಮಾಡಿ).",
            "ಚೆನ್ನಾಗಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು ಕನಿಷ್ಠ 8 ಗಂಟೆ ನಿದ್ರೆ ಮಾಡಿ."
        ],
        "diet_kn": "ಬಿಸಿ ಗಂಜಿ, ರಸಂ ಅನ್ನ, ಬಿಸಿ ಸೂಪ್, ಎಳನೀರು ಸೇವಿಸಿ. ತಣ್ಣನೆಯ ಪಾನೀಯಗಳು ಮತ್ತು ಐಸ್ ಕ್ರೀಮ್ ಸಂಪೂರ್ಣವಾಗಿ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಧೂಳು, ತಂಪು ಗಾಳಿ ತಪ್ಪಿಸಿ. ಕೆಮ್ಮುವಾಗ ಕರವಸ್ತ್ರ ಬಳಸಿ."
    },
    "Jaundice": {
        "display_name_kn": "ಕಾಮಾಲೆ / ಜಾಂಡೀಸ್ (Jaundice)",
        "description_kn": "ಯಕೃತ್ತಿನಲ್ಲಿ (ಲಿವರ್) ಬಿಲಿರುಬಿನ್ ಎಂಬ ಪಿಗ್ಮೆಂಟ್ ಹೆಚ್ಚಾಗುವುದರಿಂದ ಕಣ್ಣುಗಳು ಮತ್ತು ಚರ್ಮ ಹಳದಿಯಾಗುತ್ತವೆ. ಗಾಢ ಹಳದಿ ಮೂತ್ರ, ವಾಂತಿ, ಹಸಿವಿಲ್ಲದಿರುವುದು ಮತ್ತು ವಿಪರೀತ ಸುಸ್ತು ಇದರ ಪ್ರಮುಖ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಯಕೃತ್ತು ಮತ್ತು ಜಠರ ತಜ್ಞರು (Gastroenterologist / Hepatologist)",
        "verdict_doctor_kn": "ಕಾಮಾಲೆ ಲಿವರ್ ಸಮಸ್ಯೆಯಾಗಿರುವುದರಿಂದ ರಕ್ತ ಪರೀಕ್ಷೆ (LFT) ಹಾಗೂ ವೈದ್ಯರ ತಪಾಸಣೆ ಅತ್ಯಗತ್ಯ. ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಸಂಪೂರ್ಣ ದೈಹಿಕ ವಿಶ್ರಾಂತಿ ಅತ್ಯಗತ್ಯ. ಕರಿದ ಮತ್ತು ಜಿಡ್ಡಿನ ಪದಾರ್ಥಗಳನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ತ್ಯಜಿಸಿ.",
        "home_remedies_kn": [
            "ಕಬ್ಬಿನ ಹಾಲು (ಶುದ್ಧವಾದ ಸ್ಥಳದಿಂದ) ಅಥವಾ ಎಳನೀರು ಪ್ರತಿದಿನ ಕುಡಿಯಿರಿ.",
            "ನೆಲ್ಲಿಕಾಯಿ ರಸ ಅಥವಾ ಮೂಲಂಗಿ ರಸ ಸೇವನೆ ಯಕೃತ್ತಿಗೆ ಹಿತಕಾರಿ.",
            "ಸಂಪೂರ್ಣ ಬೆಡ್ ರೆಸ್ಟ್ (ದೈಹಿಕ ವಿಶ್ರಾಂತಿ) ಪಡೆಯಿರಿ.",
            "ಕರಿದ, ಎಣ್ಣೆಯುಕ್ತ ಮತ್ತು ಮಸಾಲೆಯುಕ್ತ ಆಹಾರ ಸಂಪೂರ್ಣವಾಗಿ ನಿಲ್ಲಿಸಿ."
        ],
        "diet_kn": "ಲಘು ಆಹಾರ, ಬಾರ್ಲಿ ನೀರು, ಹಣ್ಣಿನ ರಸಗಳು, ಬೇಯಿಸಿದ ತರಕಾರಿ ಸೇವಿಸಿ.",
        "precautions_kn": "ಯಾವುದೇ ಕಾರಣಕ್ಕೂ ಮದ್ಯಪಾನ ಮಾಡಬೇಡಿ. ವೈದ್ಯರ ಅನುಮತಿಯಿಲ್ಲದೆ ನೋವು ನಿವಾರಕ ಮಾತ್ರೆಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ."
    },
    "Malaria": {
        "display_name_kn": "ಮಲೇರಿಯಾ ಜ್ವರ (Malaria)",
        "description_kn": "ಹೆಣ್ಣು ಅನಾಫಿಲಿಸ್ ಸೊಳ್ಳೆಯ ಕಡಿತದಿಂದ ಹರಡುವ ಪರಾವಲಂಬಿ ಸೋಂಕು. ನಡುಕ ಹುಟ್ಟಿಸುವ ತೀವ್ರ ಚಳಿ, ಹಠಾತ್ ಜ್ವರ, ಬೆವರುವುದು, ತಲೆನೋವು ಮತ್ತು ವಾಂತಿ ಇದರ ಪ್ರಮುಖ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಸಾಮಾನ್ಯ ವೈದ್ಯರು / ಸಾಂಕ್ರಾಮಿಕ ರೋಗ ತಜ್ಞರು (General Physician)",
        "verdict_doctor_kn": "ಮಲೇರಿಯಾ ರಕ್ತ ಪರೀಕ್ಷೆ (Malaria Smear/Rapid Test) ಮಾಡಿಸಿ ನಿರ್ದಿಷ್ಟ ಮಲೇರಿಯಾ ವಿರೋಧಿ ಔಷಧಿಗಳನ್ನು ಪಡೆಯಲು ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಮನೆಯಲ್ಲಿ ಸೊಳ್ಳೆ ಪರದೆ ಬಳಸಿ, ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು ನಿರ್ಜಲೀಕರಣ ತಪ್ಪಿಸಲು ಸಾಕಷ್ಟು ನೀರು ಕುಡಿಯಿರಿ.",
        "home_remedies_kn": [
            "ದೇಹದ ಉಷ್ಣತೆ ನಿಯಂತ್ರಿಸಲು ಹಣೆಯ ಮೇಲೆ ತಣ್ಣೀರಿನ ಪಟ್ಟಿ ಹಾಕಿ.",
            "ಓಆರ್‌ಎಸ್ (ORS) ಅಥವಾ ಎಳನೀರು ಕುಡಿದು ದೇಹವನ್ನು ಹೈಡ್ರೇಟ್ ಆಗಿ ಇಟ್ಟುಕೊಳ್ಳಿ.",
            "ತುಳಸಿ ಎಲೆಗಳ ಕಷಾಯಕ್ಕೆ ಮೆಣಸು ಪುಡಿ ಸೇರಿಸಿ ಕುಡಿಯಿರಿ."
        ],
        "diet_kn": "ಬೇಯಿಸಿದ ಲಘು ಆಹಾರ, ಅನ್ನ-ಮಜ್ಜಿಗೆ, ಕಿಚಡಿ ಸೇವಿಸಿ.",
        "precautions_kn": "ಮನೆಯ ಸುತ್ತ ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ, ಸೊಳ್ಳೆ ಕಡಿತದಿಂದ ರಕ್ಷಣೆ ಪಡೆಯಿರಿ."
    },
    "Dengue": {
        "display_name_kn": "ಡೆಂಗ್ಯೂ ಜ್ವರ (Dengue Fever)",
        "description_kn": "ಈಡಿಸ್ ಸೊಳ್ಳೆಯಿಂದ ಹರಡುವ ವೈರಲ್ ಸೋಂಕು. ಹಠಾತ್ ವಿಪರೀತ ಜ್ವರ, ಕಣ್ಣಿನ ಹಿಂಭಾಗದ ನೋವು, ಕೀಲು ಮತ್ತು ಸ್ನಾಯು ನೋವು, ಹಾಗೂ ಪ್ಲೇಟ್‌ಲೆಟ್ ಸಂಖ್ಯೆ ಕುಸಿಯುವುದು ಇದರ ಪ್ರಮುಖ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಸಾಮಾನ್ಯ ವೈದ್ಯರು (General Physician)",
        "verdict_doctor_kn": "ಪ್ಲೇಟ್‌ಲೆಟ್ ಮಟ್ಟ ಮತ್ತು ರಕ್ತಸ್ರಾವದ ಅಪಾಯವನ್ನು ಪರೀಕ್ಷಿಸಲು ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ ಸಿಬಿಸಿ (CBC) ರಕ್ತ ಪರೀಕ್ಷೆ ಮಾಡಿಸಿಕೊಳ್ಳಿ.",
        "verdict_rest_kn": "ಸಂಪೂರ್ಣ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ, ನಿರಂತರವಾಗಿ ದ್ರವಾಹಾರ ಸೇವಿಸಿ.",
        "home_remedies_kn": [
            "ಪಪ್ಪಾಯಿ ಎಲೆಯ ರಸ (1-2 ಚಮಚ) ಪ್ಲೇಟ್‌ಲೆಟ್ ಹೆಚ್ಚಿಸಲು ಸಹಕಾರಿ.",
            "ಎಳನೀರು, ದಾಳಿಂಬೆ ಜ್ಯೂಸ್ ಮತ್ತು ಓಆರ್‌ಎಸ್ (ORS) ಧಾರಾಳವಾಗಿ ಕುಡಿಯಿರಿ.",
            "ವಿಪರೀತ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ."
        ],
        "diet_kn": "ಲಘು ಪಚನವಾಗುವ ಆಹಾರ, ಕಿವಿ ಹಣ್ಣು, ದಾಳಿಂಬೆ, ಬಿಸಿ ತರಕಾರಿ ಸೂಪ್ ಸೇವಿಸಿ.",
        "precautions_kn": "ಆಸ್ಪಿರಿನ್ ಅಥವಾ ಐಬುಪ್ರೊಫೇನ್ ಮಾತ್ರೆ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ (ಇದು ರಕ್ತಸ್ರಾವ ಉಂಟುಮಾಡಬಹುದು)."
    },
    "Typhoid Fever": {
        "display_name_kn": "ಟೈಫಾಯ್ಡ್ ಜ್ವರ (Typhoid Fever)",
        "description_kn": "ಕಲುಷಿತ ನೀರು ಅಥವಾ ಆಹಾರದ ಮೂಲಕ ಸಾಲ್ಮೊನೆಲ್ಲಾ ಬ್ಯಾಕ್ಟೀರಿಯಾದಿಂದ ಹರಡುವ ಜ್ವರ. ನಿರಂತರ ತೀವ್ರ ಜ್ವರ, ಹೊಟ್ಟೆ ನೋವು, ಸುಸ್ತು, ತಲೆನೋವು ಮತ್ತು ಮಲಬದ್ಧತೆ ಅಥವಾ ಭೇದಿ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಸಾಮಾನ್ಯ ವೈದ್ಯರು / ಸಾಂಕ್ರಾಮಿಕ ರೋಗ ತಜ್ಞರು (Physician)",
        "verdict_doctor_kn": "ಟೈಫಾಯ್ಡ್‌ಗೆ ಸೂಕ್ತ ಆ್ಯಂಟಿಬಯೋಟಿಕ್ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯ. ವಿಡಾಲ್ (Widal) ಪರೀಕ್ಷೆಗಾಗಿ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಸಂಪೂರ್ಣ ವಿಶ್ರಾಂತಿ ಮತ್ತು ಕಾಯಿಸಿ ಆರಿಸಿದ ನೀರನ್ನು ಮಾತ್ರ ಕುಡಿಯಿರಿ.",
        "home_remedies_kn": [
            "ಕಾಯಿಸಿ ಆರಿಸಿದ ನೀರನ್ನು ಮಾತ್ರ ಧಾರಾಳವಾಗಿ ಕುಡಿಯಿರಿ.",
            "ಲವಂಗ ಮತ್ತು ತುಳಸಿ ಕಷಾಯ ಕುಡಿಯುವುದು ಹೊಟ್ಟೆಯ ಸೋಂಕಿಗೆ ಒಳ್ಳೆಯದು.",
            "ದೇಹದ ಉಷ್ಣತೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ."
        ],
        "diet_kn": "ನುಣ್ಣಗೆ ಬೇಯಿಸಿದ ಅನ್ನ, ಹೆಸರುಬೇಳೆ ಕಿಚಡಿ, ಹಣ್ಣಿನ ಜ್ಯೂಸ್ ಸೇವಿಸಿ. ಹೊರಗಿನ ಕಚ್ಚಾ ಆಹಾರ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಹೊರಗಿನ ಅಥವಾ ಅಶುದ್ಧ ಆಹಾರ ಸೇವಿಸಬೇಡಿ."
    },
    "GERD (Acid Reflux)": {
        "display_name_kn": "ಅಸಿಡಿಟಿ / ಎದೆ ಉರಿ / ಗ್ಯಾಸ್ಟ್ರಿಕ್ (GERD)",
        "description_kn": "ಜಠರದ ಆಮ್ಲವು ಅನ್ನನಾಳಕ್ಕೆ ಹಿಮ್ಮುಖವಾಗಿ ಹರಿಯುವುದರಿಂದ ಎದೆಯಲ್ಲಿ ಉರಿ, ಹುಳಿ ತೇಗು, ಹೊಟ್ಟೆ ನೋವು ಮತ್ತು ಗಂಟಲಿನಲ್ಲಿ ಕಹಿ ರುಚಿ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಜಠರ ಮತ್ತು ಕರುಳು ರೋಗ ತಜ್ಞರು (Gastroenterologist)",
        "verdict_doctor_kn": "ಎದೆ ನೋವು ತೀವ್ರವಾಗಿದ್ದರೆ ಅಥವಾ ನಿರಂತರ ಹುಣ್ಣು ಇದ್ದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "verdict_rest_kn": "ಖಾರ, ಎಣ್ಣೆಯುಕ್ತ ಆಹಾರ ತ್ಯಜಿಸಿ, ಊಟದ ನಂತರ ತಕ್ಷಣ ಮಲಗಬೇಡಿ.",
        "home_remedies_kn": [
            "ತಣ್ಣನೆಯ ಹಾಲು ಅಥವಾ ಮಜ್ಜಿಗೆಗೆ ಜೀರಿಗೆ ಪುಡಿ ಬೆರೆಸಿ ಕುಡಿಯಿರಿ.",
            "ಊಟದ ನಂತರ ಸ್ವಲ್ಪ ಸೋಂಪು (ಬಡೆಸೋಪ್) ಅಥವಾ ಬಾಳೆಹಣ್ಣು ಸೇವಿಸಿ.",
            "ರಾತ್ರಿ ಊಟ ಮಲಗುವ 2-3 ಗಂಟೆ ಮುಂಚಿತವಾಗಿ ಮುಗಿಸಿ.",
            "ತಲೆದಿಂಬನ್ನು ಸ್ವಲ್ಪ ಎತ್ತರದಲ್ಲಿಟ್ಟುಕೊಂಡು ಮಲಗಿ."
        ],
        "diet_kn": "ಮಜ್ಜಿಗೆ, ಸೌತೆಕಾಯಿ, ಕಲ್ಲಂಗಡಿ, ಎಳನೀರು ಸೇವಿಸಿ. ಚಹಾ, ಕಾಫಿ, ಮದ್ಯಪಾನ ಮತ್ತು ತಂಬಾಕು ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಒಮ್ಮೆಲೇ ಹೊಟ್ಟೆ ತುಂಬಾ ಊಟ ಮಾಡಬೇಡಿ, ಸಣ್ಣ ಪ್ರಮಾಣದಲ್ಲಿ ಆಗಾಗ ಊಟ ಮಾಡಿ."
    },
    "Gastroenteritis (Food Poisoning / Stomach Infection)": {
        "display_name_kn": "ಫುಡ್ ಪಾಯಿಸನಿಂಗ್ / ವಾಂತಿ-ಭೇದಿ (Gastroenteritis)",
        "description_kn": "ಹಳಸಿದ ಅಥವಾ ಕಲುಷಿತ ಆಹಾರ ಸೇವನೆಯಿಂದ ಹೊಟ್ಟೆ ಮತ್ತು ಕರುಳಿನಲ್ಲಿ ಉಂಟಾಗುವ ಸೋಂಕು. ತೀವ್ರ ವಾಂತಿ, ಪದೇ ಪದೇ ನೀರಿನಂತಹ ಭೇದಿ, ಹೊಟ್ಟೆ ಸೆಳೆತ ಮತ್ತು ನಿರ್ಜಲೀಕರಣ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಸಾಮಾನ್ಯ ವೈದ್ಯರು / ಜಠರ ರೋಗ ತಜ್ಞರು (Gastroenterologist)",
        "verdict_doctor_kn": "ವಾಂತಿ ನಿಲ್ಲದಿದ್ದರೆ ಅಥವಾ ಅತಿಯಾದ ನಿರ್ಜಲೀಕರಣ (ಕಣ್ಣು ಗುಳಿಬೀಳುವುದು, ಮೂತ್ರ ನಿಲ್ಲುವುದು) ಕಂಡುಬಂದರೆ ತಕ್ಷಣ ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ.",
        "verdict_rest_kn": "ಪ್ರತಿ ಬಾರಿ ಭೇದಿಯಾದಾಗಲೂ ಓಆರ್‌ಎಸ್ (ORS) ಅಥವಾ ಎಳನೀರು ಕುಡಿದು ದೇಹದಲ್ಲಿ ನೀರಿನಾಂಶ ಕಾಪಾಡಿಕೊಳ್ಳಿ.",
        "home_remedies_kn": [
            "ಪ್ರತಿ ಗಂಟೆಗೊಮ್ಮೆ ಓಆರ್‌ಎಸ್ (ORS) ದ್ರಾವಣ ಅಥವಾ ಉಪ್ಪು-ಸಕ್ಕರೆ ನೀರು ಕುಡಿಯಿರಿ.",
            "ದಾಳಿಂಬೆ ಸಿಪ್ಪೆಯ ಕಷಾಯ ಅಥವಾ ಜೀರಿಗೆ ಕಷಾಯ ಭೇದಿ ನಿಯಂತ್ರಿಸಲು ಸಹಕಾರಿ.",
            "ಅನ್ನದ ತಿಳಿ ಗಂಜಿ ಮತ್ತು ಮೊಸರನ್ನ ಸೇವಿಸಿ."
        ],
        "diet_kn": "BRAT ಆಹಾರ (ಬಾಳೆಹಣ್ಣು, ಅನ್ನ, ಸೇಬು, ಟೋಸ್ಟ್), ಮಜ್ಜಿಗೆ. ಹಾಲು, ಚೀಸ್, ಜಿಡ್ಡಿನ ಆಹಾರ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಸ್ವಚ್ಛವಾದ ಕುದಿಸಿದ ನೀರನ್ನು ಮಾತ್ರ ಕುಡಿಯಿರಿ."
    },
    "Diabetes": {
        "display_name_kn": "ಮಧುಮೇಹ / ಸಕ್ಕರೆ ಕಾಯಿಲೆ (Diabetes Mellitus)",
        "description_kn": "ರಕ್ತದಲ್ಲಿ ಗ್ಲೂಕೋಸ್ (ಸಕ್ಕರೆ) ಪ್ರಮಾಣ ಹೆಚ್ಚಾಗುವ ಸ್ಥಿತಿ. ಪದೇ ಪದೇ ಮೂತ್ರ ವಿಸರ್ಜನೆ, ಅತಿಯಾದ ಬಾಯಾರಿಕೆ, ಅತಿಯಾದ ಹಸಿವು, ತೂಕ ಇಳಿಕೆ ಮತ್ತು ನಿಶ್ಯಕ್ತಿ ಇದರ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಮಧುಮೇಹ ಮತ್ತು ಹಾರ್ಮೋನ್ ತಜ್ಞರು (Diabetologist / Endocrinologist)",
        "verdict_doctor_kn": "ರಕ್ತದ ಸಕ್ಕರೆ ಮಟ್ಟವನ್ನು (HbA1c, FBS, PPBS) ನಿಯಮಿತವಾಗಿ ತಪಾಸಣೆ ಮಾಡಿಸಿ ವೈದ್ಯರ ಸಲಹೆಯಂತೆ ಔಷಧಿ ಪಡೆಯಿರಿ.",
        "verdict_rest_kn": "ದೈನಂದಿನ 30 ನಿಮಿಷ ನಡಿಗೆ, ಆಹಾರ ನಿಯಂತ್ರಣ ಮತ್ತು ಸಕ್ಕರೆ ಪದಾರ್ಥಗಳ ತ್ಯಜಿಸುವಿಕೆ ಅಗತ್ಯ.",
        "home_remedies_kn": [
            "ರಾತ್ರಿ 1 ಚಮಚ ಮೆಂತ್ಯ ಕಾಳುಗಳನ್ನು ನೆನೆಸಿಟ್ಟು ಮುಂಜಾನೆ ಆ ನೀರನ್ನು ಕುಡಿಯಿರಿ.",
            "ಪ್ರತಿದಿನ ಬೆಳಿಗ್ಗೆ ಹಾಗಲಕಾಯಿ ರಸ ಅಥವಾ ನೆಲ್ಲಿಕಾಯಿ ರಸ ಸೇವಿಸಿ.",
            "ದಾಲ್ಚಿನ್ನಿ (ಚಕ್ಕೆ) ಪುಡಿಯನ್ನು ಬೆಚ್ಚಗಿನ ನೀರಿನಲ್ಲಿ ಸೇವಿಸಿ."
        ],
        "diet_kn": "ಸಿರಿಧಾನ್ಯಗಳು (ರಾಗಿ, ನವಣೆ), ಹಸಿರು ತರಕಾರಿಗಳು, ಮೊಳಕೆ ಕಾಳುಗಳು. ಸಕ್ಕರೆ, ಸಿಹಿತಿಂಡಿ, ಮೈದಾ, ತಂಪು ಪಾನೀಯ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಪಾದಗಳ ಆರೈಕೆ ಮಾಡಿ, ಗಾಯಗಳಾಗದಂತೆ ಎಚ್ಚರವಹಿಸಿ."
    },
    "Hypertension (High Blood Pressure)": {
        "display_name_kn": "ಅಧಿಕ ರಕ್ತದೊತ್ತಡ / ಬಿಪಿ (Hypertension)",
        "description_kn": "ರಕ್ತನಾಳಗಳಲ್ಲಿ ರಕ್ತದ ಒತ್ತಡ ನಿರಂತರವಾಗಿ ಹೆಚ್ಚಾಗಿರುವ ಸ್ಥಿತಿ. ತಲೆನೋವು, ತಲೆತಿರುಗುವಿಕೆ, ಎದೆಬಡಿತ ಹೆಚ್ಚುವುದು ಮತ್ತು ಉಸಿರಾಟದ ತೊಂದರೆ ಉಂಟಾಗಬಹುದು.",
        "specialist_kn": "ಹೃದ್ರೋಗ ತಜ್ಞರು / ಸಾಮಾನ್ಯ ವೈದ್ಯರು (Cardiologist)",
        "verdict_doctor_kn": "ಬಿಪಿ 140/90 ಕ್ಕಿಂತ ಹೆಚ್ಚಿದ್ದರೆ ಅಥವಾ ಎದೆ ನೋವು ಇದ್ದರೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "verdict_rest_kn": "ಉಪ್ಪು ಸೇವನೆ ತಗ್ಗಿಸಿ, ಒತ್ತಡ ಕಡಿಮೆ ಮಾಡಿಕೊಳ್ಳಿ ಮತ್ತು ಪ್ರತಿದಿನ ಪ್ರಾಣಾಯಾಮ ಮಾಡಿ.",
        "home_remedies_kn": [
            "ಆಹಾರದಲ್ಲಿ ಉಪ್ಪಿನ (ಸೋಡಿಯಂ) ಪ್ರಮಾಣವನ್ನು ಗಣನೀಯವಾಗಿ ಕಡಿಮೆ ಮಾಡಿ.",
            "ಪ್ರತಿದಿನ ಮುಂಜಾನೆ ಬೆಳ್ಳುಳ್ಳಿಯ 1 ಎಸಳನ್ನು ಹಸಿಯಾಗಿ ಅಥವಾ ಬೆಚ್ಚಗಿನ ನೀರಿನೊಂದಿಗೆ ಸೇವಿಸಿ.",
            "ದಿನಕ್ಕೆ 15-20 ನಿಮಿಷ ಪ್ರಾಣಾಯಾಮ (ಅನುಲೋಮ-ವಿಲೋಮ) ಮಾಡಿ."
        ],
        "diet_kn": "ಬಾಳೆಹಣ್ಣು, ಎಳನೀರು, ಬೀಟ್‌ರೂಟ್ ಜ್ಯೂಸ್, ಸೊಪ್ಪುಗಳು. ಉಪ್ಪಿನಕಾಯಿ, ಹಪ್ಪಳ, ಬೇಕರಿ ತಿನಿಸುಗಳನ್ನು ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಧೂಮಪಾನ, ಮದ್ಯಪಾನ ಮತ್ತು ಮಾನಸಿಕ ಒತ್ತಡವನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ನಿವಾರಿಸಿ."
    },
    "Migraine": {
        "display_name_kn": "ಅರೆತಲೆನೋವು / ಮೈಗ್ರೇನ್ (Migraine)",
        "description_kn": "ತಲೆಯ ಒಂದು ಭಾಗದಲ್ಲಿ ತೀವ್ರವಾದ ಸಿಡಿಯುವಂತಹ ನೋವು, ಕಣ್ಣು ಮಸುಕಾಗುವುದು, ವಾಕರಿಕೆ, ಹಾಗೂ ಬೆಳಕು ಮತ್ತು ಶಬ್ದವನ್ನು ಸಹಿಸಲಾಗದಿರುವುದು ಇದರ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ನರರೋಗ ತಜ್ಞರು (Neurologist)",
        "verdict_doctor_kn": "ತಲೆನೋವು ವಾರಕ್ಕೆ 2 ಕ್ಕಿಂತ ಹೆಚ್ಚು ಬಾರಿ ಬಂದರೆ ಅಥವಾ ತೀವ್ರವಾಗಿದ್ದರೆ ನರರೋಗ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಕತ್ತಲೆಯಾದ, ಶಾಂತವಾದ ಕೋಣೆಯಲ್ಲಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು ಹಣೆಯ ಮೇಲೆ ಐಸ್ ಪ್ಯಾಕ್ ಇಡಿ.",
        "home_remedies_kn": [
            "ಕತ್ತಲೆಯಾದ, ನಿಶ್ಯಬ್ದ ಕೋಣೆಯಲ್ಲಿ ತಲೆಯಿಟ್ಟು ಕಣ್ಣು ಮುಚ್ಚಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
            "ಹಣೆ ಅಥವಾ ಕುತ್ತಿಗೆಯ ಹಿಂಭಾಗಕ್ಕೆ ಐಸ್ ಪ್ಯಾಕ್ (ತಣ್ಣನೆಯ ಪಟ್ಟಿ) ಇಡಿ.",
            "ಶುಂಠಿ ಚಹಾ ಕುಡಿಯುವುದು ವಾಕರಿಕೆ ಮತ್ತು ನೋವು ಕಡಿಮೆ ಮಾಡಲು ಸಹಕಾರಿ."
        ],
        "diet_kn": "ಸಾಕಷ್ಟು ನೀರು ಕುಡಿಯಿರಿ, ಊಟದ ಸಮಯ ತಪ್ಪಿಸಬೇಡಿ. ಚಾಕೊಲೇಟ್, ಚೀಸ್, ಅಜಿನೊಮೊಟೊ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಮೊಬೈಲ್/ಕಂಪ್ಯೂಟರ್ ಸ್ಕ್ರೀನ್ ಹೆಚ್ಚು ನೋಡುವುದನ್ನು ತಪ್ಪಿಸಿ, ನಿಯಮಿತ ನಿದ್ರೆ ಮಾಡಿ."
    },
    "Bronchial Asthma": {
        "display_name_kn": "ಉಬ್ಬಸ / ದಮ್ಮು / ಅಸ್ತಮಾ (Bronchial Asthma)",
        "description_kn": "ಶ್ವಾಸನಾಳಗಳ ಉರಿಯೂತದಿಂದಾಗಿ ಉಸಿರಾಟದ ತೊಂದರೆ, ಎದೆಯಲ್ಲಿ ಸಿಳ್ಳೆ ಹಾಕಿದಂತಹ ಶಬ್ದ (ವೀಸಿಂಗ್), ಎದೆ ಬಿಗಿತ ಮತ್ತು ನಿರಂತರ ಕೆಮ್ಮು ಉಂಟಾಗುವ ಸ್ಥಿತಿ.",
        "specialist_kn": "ಶ್ವಾಸಕೋಶ ತಜ್ಞರು (Pulmonologist)",
        "verdict_doctor_kn": "ಉಸಿರಾಟ ತೀವ್ರ ಕಷ್ಟವಾದರೆ ಅಥವಾ ಇನ್‌ಹೇಲರ್ ಕೆಲಸ ಮಾಡದಿದ್ದರೆ ತಕ್ಷಣ ತುರ್ತು ಚಿಕಿತ್ಸೆಗೆ ಭೇಟಿ ನೀಡಿ.",
        "verdict_rest_kn": "ಧೂಳು, ಹೊಗೆಯಿಂದ ದೂರವಿರಿ ಮತ್ತು ಇನ್‌ಹೇಲರ್ ಸದಾ ಜೊತೆಯಲ್ಲಿಟ್ಟುಕೊಳ್ಳಿ.",
        "home_remedies_kn": [
            "ಬೆಚ್ಚಗಿನ ನೀರಿಗೆ ತುಳಸಿ, ಶುಂಠಿ ಮತ್ತು ಜೇನುತುಪ್ಪ ಬೆರೆಸಿ ಕುಡಿಯಿರಿ.",
            "ಉಸಿರಾಟ ಕಷ್ಟವಾದಾಗ ನೇರವಾಗಿ ಕುಳಿತು ನಿಧಾನವಾಗಿ ಉಸಿರಾಡಿ.",
            "ಅಗತ್ಯವಿದ್ದಾಗ ವೈದ್ಯರು ಸೂಚಿಸಿದ ಇನ್‌ಹೇಲರ್ ಬಳಸಿ."
        ],
        "diet_kn": "ಬಿಸಿ ಸೂಪ್, ಲಘು ಆಹಾರ. ಫ್ರಿಡ್ಜ್ ನೀರು, ತಣ್ಣನೆಯ ಐಸ್ ಕ್ರೀಮ್, ಬಾಳೆಹಣ್ಣು ರಾತ್ರಿ ವೇಳೆ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಹೊಗೆ, ಧೂಳು, ಸೊಳ್ಳೆ ಕಾಯಿಲ್ ಹೊಗೆಯಿಂದ ದೂರವಿರಿ."
    },
    "Urinary Tract Infection (UTI)": {
        "display_name_kn": "ಮೂತ್ರ ಸೋಂಕು / ಉರಿ ಮೂತ್ರ (Urinary Tract Infection - UTI)",
        "description_kn": "ಮೂತ್ರನಾಳದಲ್ಲಿ ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಸೋಂಕು ಉಂಟಾಗುವುದು. ಮೂತ್ರ ವಿಸರ್ಜಿಸುವಾಗ ವಿಪರೀತ ಉರಿ, ಪದೇ ಪದೇ ಮೂತ್ರ ಬರುವುದು ಮತ್ತು ಕೆಳಹೊಟ್ಟೆ ನೋವು ಇದರ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಮೂತ್ರಪಿಂಡ / ಯುರಾಲಜಿ ತಜ್ಞರು (Urologist / General Physician)",
        "verdict_doctor_kn": "ಮೂತ್ರ ಪರೀಕ್ಷೆ (Urine Routine & Culture) ಮಾಡಿಸಿ ಸೂಕ್ತ ಆ್ಯಂಟಿಬಯೋಟಿಕ್ ಪಡೆಯಲು ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ದಿನಕ್ಕೆ ಕನಿಷ್ಠ 3-4 ಲೀಟರ್ ನೀರು, ಎಳನೀರು ಅಥವಾ ಬಾರ್ಲಿ ನೀರು ಕುಡಿಯಿರಿ.",
        "home_remedies_kn": [
            "ದಿನಕ್ಕೆ 3-4 ಲೀಟರ್ ನೀರು ಕುಡಿದು ಮೂತ್ರನಾಳವನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಿ.",
            "ಬಾರ್ಲಿ ನೀರು ಅಥವಾ ಎಳನೀರು ಕುಡಿಯುವುದು ಮೂತ್ರದ ಉರಿ ಶಮನ ಮಾಡುತ್ತದೆ.",
            "ಕ್ರ್ಯಾನ್‌ಬೆರಿ ಜ್ಯೂಸ್ (ಸಕ್ಕರೆಯಿಲ್ಲದ) ಕುಡಿಯುವುದು ಸೋಂಕು ತಡೆಯಲು ಸಹಕಾರಿ."
        ],
        "diet_kn": "ಎಳನೀರು, ಮಜ್ಜಿಗೆ, ಸೌತೆಕಾಯಿ. ಮಸಾಲೆಯುಕ್ತ, ಖಾರವಾದ ಆಹಾರ ಮತ್ತು ಮದ್ಯಪಾನ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಮೂತ್ರವನ್ನು ಹೆಚ್ಚು ಹೊತ್ತು ಹಿಡಿದಿಟ್ಟುಕೊಳ್ಳಬೇಡಿ, ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ."
    },
    "Allergy": {
        "display_name_kn": "ಅಲರ್ಜಿ / ಶೀತ ಅಲರ್ಜಿ (Allergic Rhinitis)",
        "description_kn": "ಧೂಳು, ಹೂವಿನ ಪರಾಗ ಅಥವಾ ತಂಪು ಗಾಳಿಗೆ ದೇಹದ ರೋಗನಿರೋಧಕ ವ್ಯವಸ್ಥೆಯ ಪ್ರತಿಕ್ರಿಯೆ. ನಿರಂತರ ಸೀನು, ಕಣ್ಣಿನಲ್ಲಿ ನೀರು, ಮೂಗು ಸೋರುವುದು ಮತ್ತು ತುರಿಕೆ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಅಲರ್ಜಿ ತಜ್ಞರು / ಸಾಮಾನ್ಯ ವೈದ್ಯರು (Allergist)",
        "verdict_doctor_kn": "ಉಸಿರಾಟ ಕಷ್ಟವಾದರೆ ಅಥವಾ ಮುಖ ಊದಿಕೊಂಡರೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "verdict_rest_kn": "ಅಲರ್ಜಿ ಉಂಟುಮಾಡುವ ಧೂಳು ಮತ್ತು ಸಾಕುಪ್ರಾಣಿಗಳ ಸಂಪರ್ಕ ತಪ್ಪಿಸಿ.",
        "home_remedies_kn": [
            "ಹಾಲಿಗೆ ಅರಿಶಿನ ಮತ್ತು ಕರಿಮೆಣಸು ಪುಡಿ ಬೆರೆಸಿ ಕುಡಿಯಿರಿ.",
            "ಉಗುರುಬೆಚ್ಚಗಿನ ಉಪ್ಪು ನೀರಿನಿಂದ ಮೂಗು ತೊಳೆಯಿರಿ (ಜಲನೇತಿ).",
            "ಬಿಸಿ ಹಬೆ ತೆಗೆದುಕೊಳ್ಳಿ."
        ],
        "diet_kn": "ಬೆಚ್ಚಗಿನ ತಾಜಾ ಆಹಾರ, ಜೇನುತುಪ್ಪ, ನೆಲ್ಲಿಕಾಯಿ. ಕೃತಕ ಬಣ್ಣಗಳಿರುವ ಆಹಾರ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಹೊರಗೆ ಹೋಗುವಾಗ ಮಾಸ್ಕ್ ಧರಿಸಿ, ಧೂಳಿನಿಂದ ರಕ್ಷಣೆ ಪಡೆಯಿರಿ."
    },
    "Fungal Infection": {
        "display_name_kn": "ಶಿಲೀಂಧ್ರ ಸೋಂಕು / ದದ್ದು (Fungal Infection / Ringworm)",
        "description_kn": "ಚರ್ಮದ ಮಡಿಕೆಗಳಲ್ಲಿ ತೇವಾಂಶದಿಂದ ಉಂಟಾಗುವ ಶಿಲೀಂಧ್ರ ಸೋಂಕು. ತೀವ್ರ ತುರಿಕೆ, ಕೆಂಪು ವೃತ್ತಾಕಾರದ ದದ್ದು ಮತ್ತು ಚರ್ಮ ಸುಲಿಯುವುದು ಇದರ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಚರ್ಮ ರೋಗ ತಜ್ಞರು (Dermatologist)",
        "verdict_doctor_kn": "ಸೋಂಕು ಹರಡದಂತೆ ತಡೆಯಲು ಚರ್ಮ ತಜ್ಞರಿಂದ ಸೂಕ್ತ ಆ್ಯಂಟಿಫಂಗಲ್ ಕ್ರೀಮ್ ಅಥವಾ ಮಾತ್ರೆ ಪಡೆಯಿರಿ.",
        "verdict_rest_kn": "ಚರ್ಮವನ್ನು ಸದಾ ಒಣಗಿಸಿ ಮತ್ತು ಸ್ವಚ್ಛವಾಗಿಟ್ಟುಕೊಳ್ಳಿ.",
        "home_remedies_kn": [
            "ಬೇವಿನ ಎಲೆಗಳ ಕಷಾಯದಿಂದ ಸೋಂಕಿತ ಜಾಗವನ್ನು ತೊಳೆಯಿರಿ.",
            "ತೆಂಗಿನ ಎಣ್ಣೆಗೆ ಸ್ವಲ್ಪ ಕರ್ಪೂರ ಬೆರೆಸಿ ಹಚ್ಚಿ (ತುರಿಕೆ ಶಮನಕ್ಕೆ).",
            "ಹತ್ತಿಯ (ಕಾಟನ್) ಸಡಿಲವಾದ ಉಡುಪುಗಳನ್ನು ಧರಿಸಿ."
        ],
        "diet_kn": "ಸಕ್ಕರೆ ಪದಾರ್ಥಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ, ರೋಗನಿರೋಧಕ ಶಕ್ತಿ ಹೆಚ್ಚಿಸುವ ಆಹಾರ ಸೇವಿಸಿ.",
        "precautions_kn": "ಇತರರ ಬಟ್ಟೆ ಅಥವಾ ಟವೆಲ್ ಹಂಚಿಕೊಳ್ಳಬೇಡಿ, ಬಟ್ಟೆಗಳನ್ನು ಬಿಸಿಲಿನಲ್ಲಿ ಚೆನ್ನಾಗಿ ಒಣಗಿಸಿ."
    },
    "Hemorrhoids (Piles)": {
        "display_name_kn": "ಮೂಲವ್ಯಾಧಿ / ಪೈಲ್ಸ್ (Piles / Hemorrhoids)",
        "description_kn": "ಗುದದ್ವಾರದ ರಕ್ತನಾಳಗಳು ಊದಿಕೊಳ್ಳುವುದು. ಮಲವಿಸರ್ಜನೆಯಲ್ಲಿ ನೋವು, ರಕ್ತಸ್ರಾವ, ತುರಿಕೆ ಮತ್ತು ಮಲಬದ್ಧತೆ ಇದರ ಪ್ರಮುಖ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಶಸ್ತ್ರಚಿಕಿತ್ಸಕರು / ಪೈಲ್ಸ್ ತಜ್ಞರು (Proctologist / General Surgeon)",
        "verdict_doctor_kn": "ಹೆಚ್ಚು ರಕ್ತಸ್ರಾವ ಅಥವಾ ತೀವ್ರ ನೋವಿದ್ದರೆ ಶೀಘ್ರದಲ್ಲೇ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ನಾರಿನಂಶವಿರುವ ಆಹಾರ ಸೇವಿಸಿ, ಮಲಬದ್ಧತೆ ನಿವಾರಿಸಿಕೊಳ್ಳಿ.",
        "home_remedies_kn": [
            "ಉಗುರುಬೆಚ್ಚಗಿನ ನೀರಿನಲ್ಲಿ 15 ನಿಮಿಷ ಕುಳಿತುಕೊಳ್ಳಿ (ಸಿಟ್ಜ್ ಬಾತ್).",
            "ರಾತ್ರಿ ಮಲಗುವ ಮುನ್ನ 1 ಚಮಚ ಇಸಾಬ್ಗೋಲ್ (ಹಸ್ಕ್) ಅಥವಾ ಬೆಚ್ಚಗಿನ ಹಾಲು ಕುಡಿಯಿರಿ.",
            "ಧಾರಾಳವಾಗಿ ನೀರು ಕುಡಿಯಿರಿ."
        ],
        "diet_kn": "ಹೆಚ್ಚು ನಾರಿನಂಶವಿರುವ (ಫೈಬರ್) ತರಕಾರಿಗಳು, ಹಣ್ಣುಗಳು, ಓಟ್ಸ್, ಬಾರ್ಲಿ. ಖಾರ, ಮಸಾಲೆಯುಕ್ತ ಆಹಾರ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಮಲವಿಸರ್ಜನೆ ವೇಳೆ ಹೆಚ್ಚು ಒತ್ತಡ ಹಾಕಬೇಡಿ."
    },
    "Arthritis": {
        "display_name_kn": "ಸಂಧಿವಾತ / ಕೀಲು ನೋವು (Arthritis)",
        "description_kn": "ಕೀಲುಗಳಲ್ಲಿ ಉರಿಯೂತ, ಬಿಗಿತ, ಊತ ಮತ್ತು ನಡೆಯಲು ಅಥವಾ ಕೈಕಾಲು ಚಲಿಸಲು ನೋವು ಉಂಟಾಗುವ ಸ್ಥಿತಿ.",
        "specialist_kn": "ಮೂಳೆ ಮತ್ತು ಕೀಲು ರೋಗ ತಜ್ಞರು (Rheumatologist / Orthopedic)",
        "verdict_doctor_kn": "ಕೀಲುಗಳಲ್ಲಿ ತೀವ್ರ ಊತ ಅಥವಾ ವಿರೂಪತೆ ಇದ್ದರೆ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಮೃದುವಾದ ವ್ಯಾಯಾಮ ಮಾಡಿ ಮತ್ತು ನೋವಿರುವ ಕೀಲುಗಳಿಗೆ ಬಿಸಿ ಶಾಖ ಕೊಡಿ.",
        "home_remedies_kn": [
            "ಸಾಸಿವೆ ಎಣ್ಣೆಗೆ ಬೆಳ್ಳುಳ್ಳಿ ಹಾಕಿ ಕಾಯಿಸಿ ನೋವಿರುವ ಜಾಗಕ್ಕೆ ಮಸಾಜ್ ಮಾಡಿ.",
            "ಅರಿಶಿನ ಮತ್ತು ಶುಂಠಿ ಬೆರೆಸಿದ ಬೆಚ್ಚಗಿನ ಹಾಲು ಕುಡಿಯಿರಿ.",
            "ದಿನವೂ ಲಘು ವಾಕಿಂಗ್ ಮತ್ತು ಯೋಗಾಭ್ಯಾಸ ಮಾಡಿ."
        ],
        "diet_kn": "ಒಮೆಗಾ-3 ಸಮೃದ್ಧ ವಾಲ್ನಟ್ಸ್, ಫ್ಲಾಕ್ಸ್ ಸೀಡ್ಸ್, ಹಸಿರು ಸೊಪ್ಪು. ಜಂಕ್ ಫುಡ್ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ತೂಕ ನಿಯಂತ್ರಣದಲ್ಲಿಟ್ಟುಕೊಳ್ಳಿ, ಹೆಚ್ಚು ಹೊತ್ತು ಒಂದೇ ಭಂಗಿಯಲ್ಲಿ ಕುಳಿತುಕೊಳ್ಳಬೇಡಿ."
    },
    "Osteoarthritis": {
        "display_name_kn": "ಅಸ್ಥಿಸಂಧಿವಾತ / ಮೊಣಕಾಲು ಸವೆತ (Osteoarthritis)",
        "description_kn": "ಕೀಲುಗಳ ರಕ್ಷಣಾತ್ಮಕ ಕಾರ್ಟಿಲೇಜ್ ಸವೆಯುವುದರಿಂದ ಮೂಳೆಗಳು ಉಜ್ಜಿಕೊಂಡು ತೀವ್ರ ನೋವು, ಶಬ್ದ (ಕ್ಲಿಕ್) ಮತ್ತು ನಡೆಯಲು ಕಷ್ಟವಾಗುತ್ತದೆ.",
        "specialist_kn": "ಮೂಳೆ ತಜ್ಞರು (Orthopedic Specialist)",
        "verdict_doctor_kn": "ನಡೆಯಲು ಅಸಾಧ್ಯವಾದರೆ ಅಥವಾ ಕೀಲು ಊದಿಕೊಂಡಿದ್ದರೆ ಮೂಳೆ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ ಎಕ್ಸ್-ರೇ ಮಾಡಿಸಿಕೊಳ್ಳಿ.",
        "verdict_rest_kn": "ಮೊಣಕಾಲಿಗೆ ಹೆಚ್ಚು ಒತ್ತಡ ನೀಡಬೇಡಿ, ವಿಶ್ರಾಂತಿ ಮತ್ತು ಫಿಸಿಯೋಥೆರಪಿ ಮಾಡಿ.",
        "home_remedies_kn": [
            "ಬೆಚ್ಚಗಿನ ನೀರಿನ ಶಾಖ ಅಥವಾ ಐಸ್ ಪ್ಯಾಕ್ ಹಚ್ಚಿ.",
            "ಕ್ಯಾಲ್ಸಿಯಂ ಮತ್ತು ವಿಟಮಿನ್ ಡಿ ಸಮೃದ್ಧ ಆಹಾರ ಸೇವಿಸಿ.",
            "ಫಿಸಿಯೋಥೆರಪಿಸ್ಟ್ ಸೂಚಿಸಿದ ವ್ಯಾಯಾಮಗಳನ್ನು ತಪ್ಪದೇ ಮಾಡಿ."
        ],
        "diet_kn": "ಹಾಲು, ಮೊಸರು, ರಾಗಿ, ಹಸಿರು ತರಕಾರಿಗಳು. ತೂಕ ಹೆಚ್ಚಿಸುವ ಆಹಾರ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ನೆಲದ ಮೇಲೆ ಚಕ್ಕಳಬಕ್ಕಳ ಹಾಕಿ ಕುಳಿತುಕೊಳ್ಳುವುದನ್ನು ಮತ್ತು ಭಾರವಾದ ವಸ್ತು ಎತ್ತುವುದನ್ನು ತಪ್ಪಿಸಿ."
    },
    "Cervical Spondylosis": {
        "display_name_kn": "ಕುತ್ತಿಗೆ ಮತ್ತು ಬೆನ್ನು ಮೂಳೆ ಸವೆತ (Cervical Spondylosis)",
        "description_kn": "ಕುತ್ತಿಗೆಯ ಬೆನ್ನುಹುರಿಯ ಮೂಳೆ ಮತ್ತು ಡಿಸ್ಕ್ ಸವೆತ. ಕುತ್ತಿಗೆ ನೋವು, ಭುಜ ಮತ್ತು ಕೈಗಳಲ್ಲಿ ಜುಮ್ಮೆನ್ನುವುದು, ತಲೆತಿರುಗುವಿಕೆ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಮೂಳೆ ತಜ್ಞರು / ನರರೋಗ ತಜ್ಞರು (Orthopedist / Neurologist)",
        "verdict_doctor_kn": "ಕೈಗಳಲ್ಲಿ ಶಕ್ತಿ ಕುಂದುವುದು ಅಥವಾ ನಿರಂತರ ತಲೆತಿರುಗುವಿಕೆ ಇದ್ದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "verdict_rest_kn": "ಕುತ್ತಿಗೆ ಕಾಲರ್ ಬಳಸಿ, ಭಂಗಿ (Posture) ಸುಧಾರಿಸಿಕೊಳ್ಳಿ ಮತ್ತು ಕುತ್ತಿಗೆಗೆ ಮೃದುವಾದ ಸ್ಟ್ರೆಚಿಂಗ್ ಮಾಡಿ.",
        "home_remedies_kn": [
            "ಕುತ್ತಿಗೆಗೆ ಬಿಸಿ ನೀರಿನ ಶಾಖ ಕೊಡಿ.",
            "ದಪ್ಪ ದಿಂಬು ಬಳಸುವುದನ್ನು ನಿಲ್ಲಿಸಿ, ನಯವಾದ ತೆಳು ದಿಂಬು ಬಳಸಿ.",
            "ಕುತ್ತಿಗೆಯ ಸರಳ ವ್ಯಾಯಾಮಗಳನ್ನು (Isometric Neck Exercises) ಮಾಡಿ."
        ],
        "diet_kn": "ಹಾಲು, ರಾಗಿ, ನಟ್ಸ್, ಕ್ಯಾಲ್ಸಿಯಂ ಸಮೃದ್ಧ ಆಹಾರ.",
        "precautions_kn": "ಮೊಬೈಲ್ ಬಳಸುವಾಗ ಕುತ್ತಿಗೆಯನ್ನು ಹೆಚ್ಚು ಹೊತ್ತು ಕೆಳಗೆ ಬಗ್ಗಿಸಬೇಡಿ."
    },
    "Chickenpox": {
        "display_name_kn": "ಅಮ್ಮ / ಚಿಕನ್‌ಪಾಕ್ಸ್ (Chickenpox)",
        "description_kn": "ವಾರಿಸೆಲ್ಲಾ ಜೋಸ್ಟರ್ ವೈರಸ್‌ನಿಂದ ಉಂಟಾಗುವ ಅತ್ಯಂತ ಸಾಂಕ್ರಾಮಿಕ ಕಾಯಿಲೆ. ಜ್ವರ, ತಲೆನೋವು, ಆಯಾಸ ಮತ್ತು ಮೈತುಂಬಾ ನೀರಿನಂತಹ ತುರಿಕೆ ಗುಳ್ಳೆಗಳು ಏಳುತ್ತವೆ.",
        "specialist_kn": "ಸಾಮಾನ್ಯ ವೈದ್ಯರು / ಮಕ್ಕಳ ವೈದ್ಯರು (Pediatrician / Physician)",
        "verdict_doctor_kn": "ಗುಳ್ಳೆಗಳು ಸೋಂಕಿಗೆ ಒಳಗಾದರೆ ಅಥವಾ ಜ್ವರ ತೀವ್ರವಾದರೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಗುಳ್ಳೆಗಳು ಒಣಗುವವರೆಗೆ ಮನೆಯಲ್ಲೇ ಪ್ರತ್ಯೇಕವಾಗಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
        "home_remedies_kn": [
            "ತುರಿಕೆ ಶಮನಕ್ಕೆ ಕ್ಯಾಲಮೈನ್ ಲೋಷನ್ ಹಚ್ಚಿ.",
            "ಬೇವಿನ ಎಲೆಗಳನ್ನು ಹಾಸಿಗೆಯ ಮೇಲೆ ಹರಡಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
            "ಗುಳ್ಳೆಗಳನ್ನು ಕೈಯಿಂದ ಕೆರೆಯಬೇಡಿ ಅಥವಾ ಒಡೆಯಬೇಡಿ."
        ],
        "diet_kn": "ತಂಪು ಆಹಾರ, ಎಳನೀರು, ಹಣ್ಣುಗಳು, ಮಜ್ಜಿಗೆ. ಖಾರ, ಎಣ್ಣೆಯುಕ್ತ ಆಹಾರ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಗುಳ್ಳೆ ಕೆರೆದರೆ ಕಲೆಗಳು ಉಳಿಯುತ್ತವೆ, ಆದ್ದರಿಂದ ಉಗುರುಗಳನ್ನು ಸಣ್ಣದಾಗಿ ಕತ್ತರಿಸಿ."
    },
    "Acne": {
        "display_name_kn": "ಮೊಡವೆ (Acne / Pimples)",
        "description_kn": "ಚರ್ಮದ ಎಣ್ಣೆ ಗ್ರಂಥಿಗಳು ಮತ್ತು ರಂಧ್ರಗಳು ಮುಚ್ಚಿಹೋಗುವುದರಿಂದ ಮುಖ, ಬೆನ್ನು ಮತ್ತು ಭುಜದ ಮೇಲೆ ಕೀವು ಗುಳ್ಳೆಗಳು ಮತ್ತು ಕಪ್ಪು ಕಲೆಗಳು ಉಂಟಾಗುತ್ತವೆ.",
        "specialist_kn": "ಚರ್ಮ ರೋಗ ತಜ್ಞರು (Dermatologist)",
        "verdict_doctor_kn": "ಮೊಡವೆಗಳು ತೀವ್ರ ಗಂಟುಗಳಾಗಿದ್ದರೆ ಚರ್ಮ ತಜ್ಞರಿಂದ ಚಿಕಿತ್ಸೆ ಪಡೆಯಿರಿ.",
        "verdict_rest_kn": "ಮುಖವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ ಮತ್ತು ಮೊಡವೆಗಳನ್ನು ಕೈಯಿಂದ ಚಿವುಟಬೇಡಿ.",
        "home_remedies_kn": [
            "ದಿನಕ್ಕೆ 2-3 ಬಾರಿ ಮೃದುವಾದ ಫೇಸ್ ವಾಶ್‌ನಿಂದ ಮುಖ ತೊಳೆಯಿರಿ.",
            "ಬೇವಿನ ಪುಡಿ ಅಥವಾ ಮುಲ್ತಾನಿ ಮಿಟ್ಟಿ ಲೇಪನ ಹಚ್ಚಿ.",
            "ಮೊಡವೆಗಳನ್ನು ಕೈಯಿಂದ ಮುಟ್ಟಬೇಡಿ ಅಥವಾ ಚಿವುಟಬೇಡಿ."
        ],
        "diet_kn": "ധാರಾಳವಾಗಿ ನೀರು ಕುಡಿಯಿರಿ, ಹಣ್ಣು-ತರಕಾರಿ ಸೇವಿಸಿ. ಚಾಕೊಲೇಟ್, ಎಣ್ಣೆಯುಕ್ತ ಆಹಾರ ಕಡಿಮೆ ಮಾಡಿ.",
        "precautions_kn": "ಹೆಚ್ಚು ಕೆಮಿಕಲ್ ಇರುವ ಕಾಸ್ಮೆಟಿಕ್ಸ್ ಬಳಸಬೇಡಿ."
    },
    "Psoriasis": {
        "display_name_kn": "ಸೋರಿಯಾಸಿಸ್ ಚರ್ಮ ರೋಗ (Psoriasis)",
        "description_kn": "ಸ್ವಯಂ ನಿರೋಧಕ (ಆಟೋಇಮ್ಯೂನ್) ಚರ್ಮ ರೋಗ. ಚರ್ಮದ ಕೋಶಗಳು ವೇಗವಾಗಿ ಬೆಳೆದು ಬೆಳ್ಳಿಯಂತಹ ಹುರುಪೆ, ಕೆಂಪು ದದ್ದು ಮತ್ತು ತುರಿಕೆ ಉಂಟುಮಾಡುತ್ತವೆ.",
        "specialist_kn": "ಚರ್ಮ ರೋಗ ತಜ್ಞರು (Dermatologist)",
        "verdict_doctor_kn": "ದೀರ್ಘಕಾಲೀನ ನಿಯಂತ್ರಣಕ್ಕಾಗಿ ಚರ್ಮ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ ಸೂಕ್ತ ಮುಲಾಮು ಪಡೆಯಿರಿ.",
        "verdict_rest_kn": "ಚರ್ಮವನ್ನು ಮಾಯಿಶ್ಚರೈಸ್ ಆಗಿಡಿ ಮತ್ತು ಮಾನಸಿಕ ಒತ್ತಡ ಕಡಿಮೆ ಮಾಡಿಕೊಳ್ಳಿ.",
        "home_remedies_kn": [
            "ಶುದ್ಧ ತೆಂಗಿನ ಎಣ್ಣೆ ಅಥವಾ ಅಲೋವೆರಾ ಜೆಲ್ ಹಚ್ಚಿ ಚರ್ಮ ತೇವವಾಗಿಡಿ.",
            "ಬೆಳಗಿನ ಎಳೆಯ ಬಿಸಿಲಿನಲ್ಲಿ 10-15 ನಿಮಿಷ ಕುಳಿತುಕೊಳ್ಳಿ.",
            "ಮಾನಸಿಕ ಒತ್ತಡ ಕಡಿಮೆ ಮಾಡಲು ಧ್ಯಾನ ಮಾಡಿ."
        ],
        "diet_kn": "ಉರಿಯೂತ ನಿವಾರಕ ಹಣ್ಣುಗಳು, ತರಕಾರಿಗಳು, ಅರಿಶಿನ. ಮದ್ಯಪಾನ ಮತ್ತು ತಂಬಾಕು ಸಂಪೂರ್ಣ ನಿಲ್ಲಿಸಿ.",
        "precautions_kn": "ಚರ್ಮ ಕೆರೆಯಬೇಡಿ ಮತ್ತು ಕಠಿಣ ಸಾಬೂನು ಬಳಸಬೇಡಿ."
    },
    "Impetigo": {
        "display_name_kn": "ಚರ್ಮದ ಕೀವು ಗುಳ್ಳೆಗಳು (Impetigo)",
        "description_kn": "ಬ್ಯಾಕ್ಟೀರಿಯಾದಿಂದ ಉಂಟಾಗುವ ಸಾಂಕ್ರಾಮಿಕ ಚರ್ಮದ ಸೋಂಕು. ಮೂಗು ಮತ್ತು ಬಾಯಿಯ ಸುತ್ತ ಕೆಂಪು ಹುಣ್ಣುಗಳು ಹಾಗೂ ಹಳದಿ ಬಣ್ಣದ ಜೇನುತುಪ್ಪದಂತಹ ಕೀವು ಹೊರಬರುತ್ತದೆ.",
        "specialist_kn": "ಚರ್ಮ ರೋಗ ತಜ್ಞರು / ಮಕ್ಕಳ ವೈದ್ಯರು (Dermatologist)",
        "verdict_doctor_kn": "ಇದು ಸಾಂಕ್ರಾಮಿಕವಾಗಿರುವುದರಿಂದ ಆ್ಯಂಟಿಬಯೋಟಿಕ್ ಕ್ರೀಮ್ ಮತ್ತು ಮಾತ್ರೆಗಳಿಗಾಗಿ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಹುಣ್ಣುಗಳನ್ನು ಮುಟ್ಟಬೇಡಿ ಮತ್ತು ಬಟ್ಟೆಗಳನ್ನು ಪ್ರತ್ಯೇಕವಾಗಿ ಬಿಸಿ ನೀರಿನಲ್ಲಿ ತೊಳೆಯಿರಿ.",
        "home_remedies_kn": [
            "ಬೇವಿನ ನೀರಿನಿಂದ ಹುಣ್ಣುಗಳ ಜಾಗವನ್ನು ನಿಧಾನವಾಗಿ ಸ್ವಚ್ಛಗೊಳಿಸಿ.",
            "ವೈದ್ಯರು ನೀಡಿದ ಆ್ಯಂಟಿಬಯೋಟಿಕ್ ಕ್ರೀಮ್ ಲೇಪಿಸಿ.",
            "ಮಕ್ಕಳ ಉಗುರುಗಳನ್ನು ಸಣ್ಣದಾಗಿ ಕತ್ತರಿಸಿ."
        ],
        "diet_kn": "ಪೌಷ್ಟಿಕ ಆಹಾರ ಮತ್ತು ಹಣ್ಣಿನ ರಸಗಳು.",
        "precautions_kn": "ಟವೆಲ್ ಮತ್ತು ಬಟ್ಟೆಗಳನ್ನು ಇತರರೊಂದಿಗೆ ಹಂಚಿಕೊಳ್ಳಬೇಡಿ."
    },
    "Peptic Ulcer Disease": {
        "display_name_kn": "ಹೊಟ್ಟೆಯ ಹುಣ್ಣು / ಪೆಪ್ಟಿಕ್ ಅಲ್ಸರ್ (Peptic Ulcer Disease)",
        "description_kn": "ಹೊಟ್ಟೆ ಅಥವಾ ಡ್ಯುಯೊಡಿನಮ್‌ನ ಒಳಪದರದಲ್ಲಿ ಉಂಟಾಗುವ ಹುಣ್ಣು. ಹೊಟ್ಟೆಯ ಮೇಲ್ಭಾಗದಲ್ಲಿ ಸುಡುವಂತಹ ನೋವು, ಅಜೀರ್ಣ, ವಾಕರಿಕೆ ಮತ್ತು ಹಸಿವಿನ ವ್ಯತ್ಯಾಸ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಜಠರ ಮತ್ತು ಕರುಳು ರೋಗ ತಜ್ಞರು (Gastroenterologist)",
        "verdict_doctor_kn": "ತೀವ್ರ ಹೊಟ್ಟೆ ನೋವು ಅಥವಾ ಮಲದಲ್ಲಿ ರಕ್ತ ಕಂಡುಬಂದರೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಸಮಯಕ್ಕೆ ಸರಿಯಾಗಿ ಲಘು ಊಟ ಮಾಡಿ ಮತ್ತು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ಇರಬೇಡಿ.",
        "home_remedies_kn": [
            "ಎಲೆಕೋಸು (ಕ್ಯಾಬೇಜ್) ರಸ ಕುಡಿಯುವುದು ಹುಣ್ಣು ಶಮನಕ್ಕೆ ಸಹಕಾರಿ.",
            "ತಣ್ಣನೆಯ ಹಾಲು ಅಥವಾ ಎಳನೀರು ಕುಡಿಯಿರಿ.",
            "ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ದೀರ್ಘಕಾಲ ಇರಬೇಡಿ."
        ],
        "diet_kn": "ಓಟ್ಸ್, ಬಾಳೆಹಣ್ಣು, ಮಜ್ಜಿಗೆ. ಅತಿಯಾದ ಖಾರ, ಉಪ್ಪಿನಕಾಯಿ, ಆಲ್ಕೋಹಾಲ್ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ವೈದ್ಯರ ಸಲಹೆಯಿಲ್ಲದೆ ನೋವು ನಿವಾರಕ ಮಾತ್ರೆಗಳನ್ನು (Painkillers) ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ."
    },
    "Varicose Veins": {
        "display_name_kn": "ಉಬ್ಬಿದ ರಕ್ತನಾಳಗಳು (Varicose Veins)",
        "description_kn": "ಕಾಲುಗಳಲ್ಲಿ ರಕ್ತನಾಳಗಳ ಕವಾಟಗಳು ದುರ್ಬಲಗೊಂಡು ರಕ್ತ ನಿಲ್ಲುವುದರಿಂದ ನರಗಳು ಉಬ್ಬುವುದು, ಕಾಲು ನೋವು, ಊತ ಮತ್ತು ಭಾರವಾದ ಭಾವನೆ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ರಕ್ತನಾಳ ಶಸ್ತ್ರಚಿಕಿತ್ಸಕರು (Vascular Surgeon)",
        "verdict_doctor_kn": "ಕಾಲುಗಳಲ್ಲಿ ಹುಣ್ಣು ಅಥವಾ ವಿಪರೀತ ಊತವಿದ್ದರೆ ನಾಳ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಮಲಗುವಾಗ ಕಾಲುಗಳ ಕೆಳಗೆ ದಿಂಬಿಟ್ಟು ಎತ್ತರಿಸಿ ಮತ್ತು ಸಂಕೋಚನ ಸಾಕ್ಸ್ (Compression Stockings) ಬಳಸಿ.",
        "home_remedies_kn": [
            "ಮಲಗುವಾಗ ಕಾಲಿನ ಕೆಳಗೆ ದಿಂಬು ಇಟ್ಟು ಕಾಲನ್ನು ಎತ್ತರದಲ್ಲಿಡಿ.",
            "ದೀರ್ಘಕಾಲ ಒಂದೇ ಕಡೆ ನಿಲ್ಲುವುದನ್ನು ಅಥವಾ ಕುಳಿತುಕೊಳ್ಳುವುದನ್ನು ತಪ್ಪಿಸಿ.",
            "ದಿನವೂ ವಾಕಿಂಗ್ ಮತ್ತು ಕಾಲುಗಳ ವ್ಯಾಯಾಮ ಮಾಡಿ."
        ],
        "diet_kn": "ನಾರಿನಂಶವಿರುವ ಆಹಾರ, ನೀರು. ಮಲಬದ್ಧತೆ ಉಂಟುಮಾಡುವ ಆಹಾರ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಬಿಗಿಯಾದ ಉಡುಪುಗಳನ್ನು ಧರಿಸಬೇಡಿ."
    },
    "Hypoglycemia (Low Blood Sugar)": {
        "display_name_kn": "ಸಕ್ಕರೆ ಕೊರತೆ / ಲೋ ಶುಗರ್ (Hypoglycemia)",
        "description_kn": "ರಕ್ತದಲ್ಲಿ ಗ್ಲೂಕೋಸ್ ಮಟ್ಟ ತೀವ್ರವಾಗಿ ಕುಸಿಯುವುದು. ನಡುಕ, ವಿಪರೀತ ಬೆವರು, ತಲೆತಿರುಗುವಿಕೆ, ಆತಂಕ, ಗೊಂದಲ ಮತ್ತು ಹಸಿವು ಇದರ ತುರ್ತು ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಮಧುಮೇಹ ತಜ್ಞರು (Endocrinologist)",
        "verdict_doctor_kn": "ಇದು ವೈದ್ಯಕೀಯ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಾಗಿದ್ದು, ತಕ್ಷಣ ಸಕ್ಕರೆ ಸೇವಿಸಿ ನಂತರ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "verdict_rest_kn": "ತಕ್ಷಣ 3 ಚಮಚ ಸಕ್ಕರೆ, ಜೇನುತುಪ್ಪ ಅಥವಾ ಗ್ಲೂಕೋಸ್ ನೀರು ಕುಡಿಯಿರಿ.",
        "home_remedies_kn": [
            "ತಕ್ಷಣ 15 ಗ್ರಾಂ ವೇಗವಾಗಿ ಕರಗುವ ಸಕ್ಕರೆ (ಗ್ಲೂಕೋಸ್, ಜೇನುತುಪ್ಪ, ಹಣ್ಣಿನ ಜ್ಯೂಸ್) ಸೇವಿಸಿ.",
            "15 ನಿಮಿಷಗಳ ನಂತರ ರಕ್ತದ ಸಕ್ಕರೆ ಮಟ್ಟವನ್ನು ಪರೀಕ್ಷಿಸಿ.",
            "ಪ್ರಜ್ಞೆ ತಪ್ಪುವಂತಿದ್ದರೆ ತಕ್ಷಣ ಆಸ್ಪತ್ರೆಗೆ ಕರೆದೊಯ್ಯಿರಿ."
        ],
        "diet_kn": "ಸಮಯಕ್ಕೆ ಸರಿಯಾಗಿ ಊಟ-ಉಪಾಹಾರ ಸೇವಿಸಿ.",
        "precautions_kn": "ಮಧುಮೇಹ ರೋಗಿಗಳು ಊಟ ಬಿಡಬೇಡಿ ಮತ್ತು ಸದಾ ಜೊತೆಯಲ್ಲಿ ಸಕ್ಕರೆ ಚಾಕೊಲೇಟ್ ಇಟ್ಟುಕೊಳ್ಳಿ."
    },
    "Hypothyroidism": {
        "display_name_kn": "ಹೈಪೋಥೈರಾಯ್ಡ್ / ಥೈರಾಯ್ಡ್ ಕೊರತೆ (Hypothyroidism)",
        "description_kn": "ಥೈರಾಯ್ಡ್ ಗ್ರಂಥಿಯು ಸಾಕಷ್ಟು ಹಾರ್ಮೋನ್ ಉತ್ಪಾದಿಸದಿರುವ ಸ್ಥಿತಿ. ಅತಿಯಾದ ತೂಕ ಹೆಚ್ಚಳ, ಆಲಸ್ಯ, ಚಳಿ ಸಹಿಸಲಾಗದಿರುವುದು, ಮಲಬದ್ಧತೆ ಮತ್ತು ಮುಖ ಊದುವುದು ಇದರ ಲಕ್ಷಣಗಳು.",
        "specialist_kn": "ಹಾರ್ಮೋನ್ ತಜ್ಞರು (Endocrinologist)",
        "verdict_doctor_kn": "ರಕ್ತದಲ್ಲಿ ಥೈರಾಯ್ಡ್ ಪರೀಕ್ಷೆ (TSH, T3, T4) ಮಾಡಿಸಿ ಸರಿಯಾದ ಪ್ರಮಾಣದ ಮಾತ್ರೆ ಪಡೆಯಲು ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಪ್ರತಿದಿನ ಮುಂಜಾನೆ ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ವೈದ್ಯರು ನೀಡಿದ ಥೈರಾಯ್ಡ್ ಮಾತ್ರೆ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "home_remedies_kn": [
            "ಮುಂಜಾನೆ ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ವೈದ್ಯರು ನೀಡಿದ ಮಾತ್ರೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಸೇವಿಸಿ.",
            "ದಿನವೂ 30-40 ನಿಮಿಷ ನಡಿಗೆ ಅಥವಾ ಯೋಗ ಮಾಡಿ.",
            "ಒತ್ತಡ ನಿರ್ವಹಣೆಗೆ ಪ್ರಾಣಾಯಾಮ ಮಾಡಿ."
        ],
        "diet_kn": "ಅಯೋಡಿನ್‌ಯುಕ್ತ ಉಪ್ಪು, ಬ್ರೆಜಿಲ್ ನಟ್ಸ್, ಮೊಟ್ಟೆ, ಹಸಿರು ತರಕಾರಿಗಳು. ಹಸಿ ಎಲೆಕೋಸು, ಹೂಕೋಸು ಸೇವನೆ ಮಿತಿಗೊಳಿಸಿ.",
        "precautions_kn": "ಮಾತ್ರೆ ಸೇವನೆಯ ನಂತರ ಕನಿಷ್ಠ 1 ಗಂಟೆ ಕಾಫಿ/ಚಹಾ ಕುಡಿಯಬೇಡಿ."
    },
    "Hyperthyroidism": {
        "display_name_kn": "ಹೈಪರ್‌ಥೈರಾಯ್ಡ್ / ಅತಿಯಾದ ಥೈರಾಯ್ಡ್ (Hyperthyroidism)",
        "description_kn": "ಥೈರಾಯ್ಡ್ ಗ್ರಂಥಿಯು ಅತಿಯಾಗಿ ಹಾರ್ಮೋನ್ ಸ್ರವಿಸುವ ಸ್ಥಿತಿ. ವೇಗವಾದ ತೂಕ ಇಳಿಕೆ, ತ್ವರಿತ ಹೃದಯ ಬಡಿತ, ಅತಿಯಾದ ಬೆವರು, ನಡುಕ ಮತ್ತು ಕಿರಿಕಿರಿ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಹಾರ್ಮೋನ್ ತಜ್ಞರು (Endocrinologist)",
        "verdict_doctor_kn": "ಹೃದಯದ ಮೇಲಿನ ಒತ್ತಡ ತಪ್ಪಿಸಲು ತಕ್ಷಣ ಹಾರ್ಮೋನ್ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ ಚಿಕಿತ್ಸೆ ಪಡೆಯಿರಿ.",
        "verdict_rest_kn": "ಶಾಂತವಾಗಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು ಕೆಫೀನ್ ಪದಾರ್ಥಗಳಿಂದ ದೂರವಿರಿ.",
        "home_remedies_kn": [
            "ಶಾಂತ ವಾತಾವರಣದಲ್ಲಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
            "ಮಾನಸಿಕ ಒತ್ತಡ ನಿವಾರಣೆಗೆ ಧ್ಯಾನ ಮಾಡಿ.",
            "ಸಾಕಷ್ಟು ನೀರು ಕುಡಿಯಿರಿ."
        ],
        "diet_kn": "ಕ್ಯಾಲ್ಸಿಯಂ ಮತ್ತು ಮೆಗ್ನೀಸಿಯಮ್ ಸಮೃದ್ಧ ಆಹಾರ. ಕಾಫಿ, ಚಹಾ, ಎನರ್ಜಿ ಡ್ರಿಂಕ್ಸ್ ಸಂಪೂರ್ಣ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ಅತಿಯಾದ ಬಿಸಿಲು ಮತ್ತು ಕಠಿಣ ದೈಹಿಕ ವ್ಯಾಯಾಮ ತಪ್ಪಿಸಿ."
    },
    "Vertigo (BPPV)": {
        "display_name_kn": "ತಲೆ ತಿರುಗುವಿಕೆ / ವರ್ಟಿಗೋ (Vertigo - BPPV)",
        "description_kn": "ಒಳಕಿವಿಯ ಸಮತೋಲನ ವ್ಯವಸ್ಥೆಯ ತೊಂದರೆಯಿಂದಾಗಿ ತಲೆ ಅಥವಾ ಸುತ್ತಲಿನ ಜಗತ್ತು ಗಿರಗಿರನೆ ಸುತ್ತುವಂತೆ ಭಾಸವಾಗುವುದು, ಸಮತೋಲನ ತಪ್ಪುವುದು ಮತ್ತು ವಾಕರಿಕೆ ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಕಿವಿ-ಮೂಗು-ಗಂಟಲು (ENT) ತಜ್ಞರು / ನರರೋಗ ತಜ್ಞರು (ENT / Neurologist)",
        "verdict_doctor_kn": "ತಲೆತಿರುಗುವಿಕೆ ನಿರಂತರವಾಗಿದ್ದರೆ ಅಥವಾ ವಾಂತಿ ತೀವ್ರವಾಗಿದ್ದರೆ ಇಎನ್‌ಟಿ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಹಠಾತ್ ತಲೆ ತಿರುಗಿಸಬೇಡಿ, ನಿಧಾನವಾಗಿ ಎದ್ದು ಕುಳಿತುಕೊಳ್ಳಿ ಮತ್ತು ಸಮತೋಲನ ಕಾಪಾಡಿ.",
        "home_remedies_kn": [
            "ತಲೆತಿರುಗಿದಾಗ ತಕ್ಷಣ ಶಾಂತವಾಗಿ ಒಂದೇ ಕಡೆ ಕುಳಿತುಕೊಳ್ಳಿ ಅಥವಾ ಮಲಗಿ.",
            "ಇಎನ್‌ಟಿ ತಜ್ಞರು ಸೂಚಿಸಿದ ಎಪ್ಲೆ ವ್ಯಾಯಾಮ (Epley Maneuver) ಮಾಡಿ.",
            "ಸಾಕಷ್ಟು ನೀರು ಕುಡಿಯಿರಿ ಮತ್ತು ನಿರ್ಜಲೀಕರಣ ತಪ್ಪಿಸಿ."
        ],
        "diet_kn": "ಶುಂಠಿ ಚಹಾ, ಲಘು ಆಹಾರ. ಅತಿಯಾದ ಉಪ್ಪು ಮತ್ತು ಕೆಫೀನ್ ತ್ಯಜಿಸಿ.",
        "precautions_kn": "ತಲೆತಿರುಗುವಾಗ ವಾಹನ ಚಾಲನೆ ಮಾಡಬೇಡಿ ಅಥವಾ ಎತ್ತರದ ಸ್ಥಳಗಳಿಗೆ ಹೋಗಬೇಡಿ."
    },
    "Hepatitis A": {
        "display_name_kn": "ಹೆಪಟೈಟಿಸ್ ಎ / ಯಕೃತ್ತಿನ ಸೋಂಕು (Hepatitis A)",
        "description_kn": "ಕಲುಷಿತ ಆಹಾರ ಅಥವಾ ನೀರಿನಿಂದ ಹರಡುವ ಯಕೃತ್ತಿನ ವೈರಲ್ ಸೋಂಕು. ತೀವ್ರ ಕಾಮಾಲೆ, ಹಳದಿ ಕಣ್ಣುಗಳು, ಗಾಢ ಮೂತ್ರ, ವಾಂತಿ, ಜ್ವರ ಮತ್ತು ಹೊಟ್ಟೆ ನೋವು ಉಂಟಾಗುತ್ತದೆ.",
        "specialist_kn": "ಯಕೃತ್ತು ಮತ್ತು ಜಠರ ತಜ್ಞರು (Hepatologist / Gastroenterologist)",
        "verdict_doctor_kn": "ಯಕೃತ್ತಿನ ಆರೋಗ್ಯ ತಪಾಸಣೆಗೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ ಲಿವರ್ ಫಂಕ್ಷನ್ ಟೆಸ್ಟ್ ಮಾಡಿಸಿಕೊಳ್ಳಿ.",
        "verdict_rest_kn": "ಸಂಪೂರ್ಣ ಬೆಡ್ ರೆಸ್ಟ್ ಮತ್ತು ಶುದ್ಧ ಕಾಯಿಸಿದ ನೀರು ಮಾತ್ರ ಕುಡಿಯಿರಿ.",
        "home_remedies_kn": [
            "ಸಂಪೂರ್ಣ ದೈಹಿಕ ವಿಶ್ರಾಂತಿ (ಬೆಡ್ ರೆಸ್ಟ್) ಪಡೆಯಿರಿ.",
            "ಕಬ್ಬಿನ ಹಾಲು, ಎಳನೀರು ಮತ್ತು ಗ್ಲೂಕೋಸ್ ನೀರು ಸೇವಿಸಿ.",
            "ಎಣ್ಣೆ ಮತ್ತು ಜಿಡ್ಡಿನ ಪದಾರ್ಥಗಳನ್ನು ಸಂಪೂರ್ಣ ನಿಲ್ಲಿಸಿ."
        ],
        "diet_kn": "ಕಾಯಿಸಿ ಆರಿಸಿದ ನೀರು, ಗಂಜಿ, ಹಣ್ಣಿನ ಜ್ಯೂಸ್, ಬೇಯಿಸಿದ ತರಕಾರಿ.",
        "precautions_kn": "ಮದ್ಯಪಾನ ಸಂಪೂರ್ಣವಾಗಿ ನಿಷೇಧ. ಅಶುದ್ಧ ಆಹಾರ ಸೇವಿಸಬೇಡಿ."
    },
    "Drug Reaction": {
        "display_name_kn": "ಔಷಧದ ಅಡ್ಡಪರಿಣಾಮ / ಅಲರ್ಜಿ (Drug Reaction)",
        "description_kn": "ಯಾವುದಾದರೂ ಮಾತ್ರೆ ಅಥವಾ ಚುಚ್ಚುಮದ್ದಿಗೆ ದೇಹದ ಅನಿರೀಕ್ಷಿತ ಪ್ರತಿಕ್ರಿಯೆ. ಚರ್ಮದ ಮೇಲೆ ತೀವ್ರ ತುರಿಕೆ, ಕೆಂಪು ದದ್ದು, ಗುಳ್ಳೆಗಳು ಮತ್ತು ಮೂತ್ರದಲ್ಲಿ ಉರಿ ಉಂಟಾಗಬಹುದು.",
        "specialist_kn": "ಚರ್ಮ ರೋಗ ತಜ್ಞರು / ಸಾಮಾನ್ಯ ವೈದ್ಯರು (Dermatologist / Physician)",
        "verdict_doctor_kn": "ಅನುಮಾನಾಸ್ಪದ ಔಷಧಿಯನ್ನು ತಕ್ಷಣ ನಿಲ್ಲಿಸಿ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "verdict_rest_kn": "ಔಷಧಿಯ ರಶೀದಿಯನ್ನು ವೈದ್ಯರಿಗೆ ತೋರಿಸಿ ಮತ್ತು ಹೆಚ್ಚಿನ ನೀರು ಕುಡಿಯಿರಿ.",
        "home_remedies_kn": [
            "ಹೊಸದಾಗಿ ಪ್ರಾರಂಭಿಸಿದ ಅನುಮಾನಾಸ್ಪದ ಔಷಧಿಯನ್ನು ತಕ್ಷಣವೇ ನಿಲ್ಲಿಸಿ.",
            "ತುರಿಕೆ ನಿವಾರಣೆಗೆ ತಣ್ಣೀರಿನ ಬಟ್ಟೆಯಿಂದ ಮೃದುವಾಗಿ ಒರೆಸಿ.",
            "ವೈದ್ಯರಿಗೆ ನೀವು ತೆಗೆದುಕೊಂಡ ಎಲ್ಲಾ ಔಷಧಿಗಳ ವಿವರ ನೀಡಿ."
        ],
        "diet_kn": "ಧಾರಾಳವಾಗಿ ನೀರು, ಎಳನೀರು ಕುಡಿಯಿರಿ.",
        "precautions_kn": "ಉಸಿರಾಟ ಕಷ್ಟವಾದರೆ ಅಥವಾ ತುಟಿ ಊದಿಕೊಂಡರೆ ತಕ್ಷಣ ಆಸ್ಪತ್ರೆ ತುರ್ತು ವಿಭಾಗಕ್ಕೆ ಹೋಗಿ."
    }
}


def detect_kannada_input(text: str) -> bool:
    """
    Detects if the input text is in native Kannada script (Unicode \u0C80-\u0CFF)
    or contains common transliterated Kannada symptom terms.
    """
    if not text:
        return False

    # 1. Native Kannada Unicode block check
    if any('\u0c80' <= ch <= '\u0cff' for ch in text):
        return True

    # 2. Transliterated Kannada symptom keywords
    cleaned_lower = text.lower()
    kannada_keywords = [
        "jwara", "jvara", "kemmu", "talenovu", "tale novu", "tale novvu",
        "hotte novu", "hotte novvu", "bhedi", "bedhi", "kaamale", "kamale",
        "dammu", "ubbaasa", "kannu novu", "mai novu", "maikainovu", "mai kai novu",
        "uri moothra", "uri mootra", "ede novu", "ede uri", "gantalu novu",
        "gantlu novu", "vaanti", "vakarike", "susthu", "aayasa", "malabaddhate",
        "nanage", "ide", "baruttide", "aagide", "illave"
    ]
    for kw in kannada_keywords:
        if kw in cleaned_lower:
            return True

    return False


def get_kannada_symptom_display(symptom_key: str) -> str:
    """
    Returns clean Kannada representation of a symptom with English subtitle.
    """
    clean_k = symptom_key.replace("has_", "").strip()
    return SYMPTOM_NAMES_KN.get(clean_k, f"{clean_k.replace('_', ' ').title()}")


def get_kannada_report_data(
    top_disease: str,
    matched_keys: List[str],
    urgency_level: str,
    days: int = 3,
    severity: int = 5,
    vuln_score: int = 20
) -> Dict[str, Any]:
    """
    Compiles a crystal-clear, concise Kannada Quick Medical Summary Report dataset.
    """
    standard_name = DISEASE_NAME_MAP.get(top_disease, top_disease)
    kn_data = DISEASE_KNOWLEDGE_KN.get(standard_name, {})

    display_name_kn = kn_data.get("display_name_kn", f"{standard_name}")
    description_kn = kn_data.get("description_kn", "ರೋಗಲಕ್ಷಣಗಳ ಆಧಾರದ ಮೇಲೆ ಎಐ ಮಾದರಿಯಿಂದ ಈ ಕಾಯಿಲೆ ಗುರುತಿಸಲಾಗಿದೆ.")
    specialist_kn = kn_data.get("specialist_kn", "ಸಾಮಾನ್ಯ ವೈದ್ಯರು (General Physician)")
    home_remedies_kn = kn_data.get("home_remedies_kn", [
        "ಸಾಕಷ್ಟು ಕುದಿಸಿ ಆರಿಸಿದ ನೀರು ಕುಡಿಯಿರಿ.",
        "ಉತ್ತಮ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು ಲಘು ಆಹಾರ ಸೇವಿಸಿ.",
        "ರೋಗಲಕ್ಷಣಗಳು ಹೆಚ್ಚಾದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ])
    diet_kn = kn_data.get("diet_kn", "ಬೆಚ್ಚಗಿನ ನೀರು, ಲಘು ಆಹಾರ ಸೇವಿಸಿ. ಜಂಕ್ ಫುಡ್ ತ್ಯಜಿಸಿ.")
    precautions_kn = kn_data.get("precautions_kn", "ವೈದ್ಯರ ಸಲಹೆಯಿಲ್ಲದೆ ಸ್ವಯಂ ಔಷಧಿ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.")

    # Determine doctor vs rest verdict in Kannada
    requires_doctor = (days >= 5) or ("EMERGENCY" in urgency_level.upper()) or ("SEE DOCTOR" in urgency_level.upper()) or ("CONSULT DOCTOR" in urgency_level.upper())

    if requires_doctor:
        if days >= 5:
            verdict_headline_kn = "🚨 ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡುವುದು ಸೂಕ್ತ (ವೈದ್ಯರ ಭೇಟಿ ಅಗತ್ಯ)"
            verdict_reason_kn = f"ರೋಗಲಕ್ಷಣಗಳು <b>{days} ದಿನಗಳಿಂದ</b> ಮುಂದುವರಿದಿದ್ದು (5 ದಿನಗಳ ಮಿತಿ ಮೀರಿದೆ), ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ ಪರೀಕ್ಷಿಸಿಕೊಳ್ಳುವುದು ಅತ್ಯಗತ್ಯವಾಗಿದೆ."
        else:
            verdict_headline_kn = "🚨 ವೈದ್ಯರ ಸಲಹೆ ಪಡೆಯುವುದು ಅಗತ್ಯ (ವೈದ್ಯರ ಭೇಟಿ)"
            verdict_reason_kn = f"ರೋಗಲಕ್ಷಣಗಳ ತೀವ್ರತೆ (ಮಟ್ಟ {severity}/10) ಮತ್ತು {display_name_kn} ಕಾಯಿಲೆಯ ಸೂಚನೆಗಳಿರುವುದರಿಂದ ವೈದ್ಯರ ತಪಾಸಣೆ ಅಗತ್ಯ."
        verdict_badge_kn = "ವೈದ್ಯರ ಭೇಟಿ ಅಗತ್ಯ"
        immediate_action_kn = "ಹತ್ತಿರದ ಕ್ಲಿನಿಕ್ ಅಥವಾ ಆಸ್ಪತ್ರೆಗೆ ತೆರಳಿ ತಪಾಸಣೆ ಮಾಡಿಸಿಕೊಳ್ಳಿ."
    else:
        verdict_headline_kn = "🏡 ಮನೆಯಲ್ಲೇ ವಿಶ್ರಾಂತಿ ಮತ್ತು ಆರೈಕೆ ಸಾಕು (ವಿಶ್ರಾಂತಿ ತೆಗೆದುಕೊಳ್ಳಿ)"
        verdict_reason_kn = f"ರೋಗಲಕ್ಷಣಗಳು ಸೌಮ್ಯವಾಗಿದ್ದು ({days} ದಿನಗಳು), ಯಾವುದೇ ತುರ್ತು ಅಪಾಯವಿಲ್ಲ. ಮನೆಯಲ್ಲೇ ಸೂಕ್ತ ಆರೈಕೆ ಮತ್ತು ವಿಶ್ರಾಂತಿಯಿಂದ ಗುಣಮುಖರಾಗಬಹುದು."
        verdict_badge_kn = "ಮನೆ ವಿಶ್ರಾಂತಿ ಸಾಕು"
        immediate_action_kn = "ಸಾಕಷ್ಟು ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ, ಬೆಚ್ಚಗಿನ ದ್ರವಾಹಾರ ಸೇವಿಸಿ ಮತ್ತು 48 ಗಂಟೆಗಳ ಕಾಲ ಗಮನಿಸಿ."

    # Urgency Level in Kannada
    urgency_map_kn = {
        "EMERGENCY": "🚨 ತಕ್ಷಣ ತುರ್ತು ಆಸ್ಪತ್ರೆಗೆ ದಾಖಲಾಗಿ",
        "See Doctor Immediately": "🚨 ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ",
        "SEE DOCTOR IMMEDIATELY": "🚨 ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ",
        "See Doctor Soon": "⚠️ ಶೀಘ್ರದಲ್ಲೇ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ",
        "CONSULT DOCTOR SOON": "⚠️ ಶೀಘ್ರದಲ್ಲೇ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ",
        "Monitor 2-3 Days": "📋 2-3 ದಿನ ಗಮನಿಸಿ & ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ",
        "MONITOR 2-3 DAYS": "📋 2-3 ದಿನ ಗಮನಿಸಿ & ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ",
        "Self-Care": "🏡 ಮನೆ ಮದ್ದು & ವಿಶ್ರಾಂತಿ",
        "SELF-CARE & HOME REMEDIES": "🏡 ಮನೆ ಮದ್ದು & ವಿಶ್ರಾಂತಿ"
    }
    urgency_kn = urgency_map_kn.get(urgency_level, urgency_level)

    # Symptom list in Kannada
    translated_symptoms = [get_kannada_symptom_display(s) for s in matched_keys]

    # Ayurvedic Formulations in Kannada
    ayurvedic_kn = kn_data.get("ayurvedic_kn", [
        "ಶುಂಠಿ, ತುಳಸಿ ಮತ್ತು ಕರಿಮೆಣಸು ಬೆರೆಸಿದ ಬೆಚ್ಚಗಿನ ಕಷಾಯ ಸೇವಿಸಿ.",
        "ರಾತ್ರಿ ಮಲಗುವ ಮುನ್ನ ಬೆಚ್ಚಗಿನ ಹಾಲಿಗೆ ಚಿಟಿಕೆ ಅರಿಶಿನ ಬೆರೆಸಿ ಕುಡಿಯಿರಿ.",
        "ನಿಮ್ಮ ದೇಹ ಪ್ರಕೃತಿಗೆ ಅನುಗುಣವಾಗಿ ಆಯುರ್ವೇದ ವೈದ್ಯರ ಸಲಹೆ ಪಡೆಯಿರಿ."
    ])

    # Diet Do's & Don'ts in Kannada
    diet_dos_kn = kn_data.get("diet_dos_kn", [
        "ಕಾಯಿಸಿ ಆರಿಸಿದ ಬೆಚ್ಚಗಿನ ನೀರು ಮತ್ತು ಗಿಡಮೂಲಿಕೆ ಚಹಾ",
        "ಲಘುವಾಗಿ ಬೇಯಿಸಿದ ಗಂಜಿ, ರಸಂ ಅನ್ನ, ಹೆಸರುಬೇಳೆ ಕಿಚಡಿ",
        "ತಾಜಾ ಹಣ್ಣುಗಳು ಮತ್ತು ಎಳನೀರು"
    ])
    diet_donts_kn = kn_data.get("diet_donts_kn", [
        "ಅತಿಯಾದ ಎಣ್ಣೆ, ಮಸಾಲೆ ಮತ್ತು ಕರಿದ ಜಂಕ್ ಫುಡ್",
        "ತಣ್ಣನೆಯ ಐಸ್ ಕ್ರೀಮ್ ಮತ್ತು ಕಾರ್ಬೊನೇಟೆಡ್ ತಂಪು ಪಾನೀಯಗಳು",
        "ಹೊರಗಿನ ಅಶುದ್ಧ ಅಥವಾ ಹಳಸಿದ ಆಹಾರ ಪದಾರ್ಥಗಳು"
    ])

    # Detailed Clinical Precautions in Kannada
    precautions_list_kn = kn_data.get("precautions_list_kn", [
        "ರೋಗಲಕ್ಷಣಗಳು 5 ದಿನ ಮೀರಿದರೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
        "ವೈದ್ಯರ ಸಲಹೆಯಿಲ್ಲದೆ ಯಾವುದೇ ಆ್ಯಂಟಿಬಯೋಟಿಕ್ ಅಥವಾ ನೋವು ನಿವಾರಕ ಮಾತ್ರೆಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.",
        "ದೇಹದ ಉಷ್ಣತೆ ಮತ್ತು ರಕ್ತದೊತ್ತಡವನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ."
    ])

    # Clean text for Kannada TTS speech synthesis
    tts_text_kn = f"ವೈದ್ಯಕೀಯ ವರದಿ: ಗುರುತಿಸಲಾದ ಕಾಯಿಲೆ {display_name_kn}. ತುರ್ತು ಮಟ್ಟ {urgency_kn}. {'ನೀವು ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಬೇಕು' if requires_doctor else 'ಮನೆಯಲ್ಲೇ ಉತ್ತಮ ವಿಶ್ರಾಂತಿ ಮತ್ತು ಬಿಸಿ ನೀರು ಕುಡಿಯುವುದು ಸೂಕ್ತ'}."

    return {
        "display_name_kn": display_name_kn,
        "description_kn": description_kn,
        "specialist_kn": specialist_kn,
        "urgency_kn": urgency_kn,
        "requires_doctor": requires_doctor,
        "verdict_headline_kn": verdict_headline_kn,
        "verdict_reason_kn": verdict_reason_kn,
        "verdict_badge_kn": verdict_badge_kn,
        "immediate_action_kn": immediate_action_kn,
        "translated_symptoms": translated_symptoms,
        "home_remedies_kn": home_remedies_kn,
        "ayurvedic_kn": ayurvedic_kn,
        "diet_kn": diet_kn,
        "diet_dos_kn": diet_dos_kn,
        "diet_donts_kn": diet_donts_kn,
        "precautions_kn": precautions_kn,
        "precautions_list_kn": precautions_list_kn,
        "tts_text_kn": tts_text_kn
    }


def get_comorbidity_tailored_precautions_kn(existing_conditions: List[str], disease_name: str) -> List[str]:
    """
    Returns specific, actionable safety guidelines customized for the patient's pre-existing conditions in Kannada.
    """
    if not existing_conditions or "None" in existing_conditions:
        return []

    tailored = []
    active = [c for c in existing_conditions if c and c != "None"]

    if "Diabetes" in active:
        tailored.append("🩺 **ಮಧುಮೇಹ ಸುರಕ್ಷತಾ ಎಚ್ಚರಿಕೆ:** ಜ್ವರ ಮತ್ತು ಸೋಂಕು ಇನ್ಸುಲಿನ್ ಪ್ರತಿರೋಧವನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ. ಪ್ರತಿ 4-6 ಗಂಟೆಗೊಮ್ಮೆ ರಕ್ತದ ಸಕ್ಕರೆ ಮಟ್ಟ ತಪಾಸಿಸಿ. ಜೇನುತುಪ್ಪ, ಬೆಲ್ಲ ಅಥವಾ ಸಕ್ಕರೆ ಸಿರಪ್ ಹೊಂದಿರುವ ಮನೆಮದ್ದುಗಳನ್ನು ತಪ್ಪಿಸಿ.")
    if "Hypertension" in active:
        tailored.append("💓 **ಅಧಿಕ ರಕ್ತದೊತ್ತಡ ಎಚ್ಚರಿಕೆ:** ಅತಿಯಾದ ಉಪ್ಪು ಅಥವಾ ಸೋಡಿಯಂ ಸೇವನೆ ತಪ್ಪಿಸಿ. ಬಿಪಿ ಹೆಚ್ಚಿಸುವ ಗಿಡಮೂಲಿಕೆಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.")
    if "Asthma" in active:
        tailored.append("🫁 **ಉಬ್ಬಸ / ಅಸ್ತಮಾ ಎಚ್ಚರಿಕೆ:** ಇನ್‌ಹೇಲರ್ (ಸಾಲ್ಬುಟಮಾಲ್) ಸದಾ ಜೊತೆಯಲ್ಲಿಡಿ. ತಂಪು ಗಾಳಿ, ಧೂಳು, ತೀವ್ರ ಪರಿಮಳದ ಹೊಗೆಯಿಂದ ದೂರವಿರಿ.")
    if "Heart Disease" in active:
        tailored.append("❤️ **ಹೃದ್ರೋಗ ಎಚ್ಚರಿಕೆ:** ನಾಡಿಬಡಿತ ಗಮನಿಸಿ. ಒಮ್ಮೆಗೆ ಅತಿಯಾದ ದ್ರವ ಸೇವಿಸಬೇಡಿ. ಎದೆಭಾರ ಕಂಡುಬಂದರೆ ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.")
    if "Kidney Disease" in active:
        tailored.append("🧪 **ಮೂತ್ರಪಿಂಡ / ಕಿಡ್ನಿ ಎಚ್ಚರಿಕೆ:** ನೋವು ನಿವಾರಕ ಮಾತ್ರೆಗಳನ್ನು (Painkillers) ಸ್ವಯಂ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ. ವೈದ್ಯರು ಸೂಚಿಸಿದ ನೀರಿನ ಮಿತಿ ಪಾಲಿಸಿ.")

# =========================================================================
# PATIENT INFORMATION BILINGUAL TRANSLATIONS & NORMALIZATION
# =========================================================================
GENDER_MAP_KN = {
    "Male": "ಪುರುಷ (Male)",
    "Female": "ಮಹಿಳೆ (Female)",
    "Other": "ಇತರ (Other)",
    "Prefer not to say": "ಹೇಳಲು ಇಷ್ಟವಿಲ್ಲ (Prefer not to say)"
}

REVERSE_GENDER_MAP_KN = {
    "ಪುರುಷ (Male)": "Male",
    "ಮಹಿಳೆ (Female)": "Female",
    "ಇತರ (Other)": "Other",
    "ಹೇಳಲು ಇಷ್ಟವಿಲ್ಲ (Prefer not to say)": "Prefer not to say",
    "Male": "Male",
    "Female": "Female",
    "Other": "Other",
    "Prefer not to say": "Prefer not to say"
}

CONDITIONS_MAP_KN = {
    "None": "ಯಾವುದೂ ಇಲ್ಲ (None)",
    "Diabetes": "ಮಧುಮೇಹ / ಶುಗರ್ (Diabetes)",
    "Hypertension": "ರಕ್ತದೊತ್ತಡ / ಬಿಪಿ (Hypertension)",
    "Asthma": "ಉಬ್ಬಸ / ಅಸ್ತಮಾ (Asthma)",
    "Heart Disease": "ಹೃದ್ರೋಗ (Heart Disease)",
    "Kidney Disease": "ಮೂತ್ರಪಿಂಡ ಕಾಯಿಲೆ (Kidney Disease)",
    "Thyroid Disorder": "ಥೈರಾಯ್ಡ್ ಸಮಸ್ಯೆ (Thyroid Disorder)"
}

REVERSE_CONDITIONS_MAP_KN = {
    "ಯಾವುದೂ ಇಲ್ಲ (None)": "None",
    "ಮಧುಮೇಹ / ಶುಗರ್ (Diabetes)": "Diabetes",
    "ರಕ್ತದೊತ್ತಡ / ಬಿಪಿ (Hypertension)": "Hypertension",
    "ಉಬ್ಬಸ / ಅಸ್ತಮಾ (Asthma)": "Asthma",
    "ಹೃದ್ರೋಗ (Heart Disease)": "Heart Disease",
    "ಮೂತ್ರಪಿಂಡ ಕಾಯಿಲೆ (Kidney Disease)": "Kidney Disease",
    "ಥೈರಾಯ್ಡ್ ಸಮಸ್ಯೆ (Thyroid Disorder)": "Thyroid Disorder",
    "None": "None",
    "Diabetes": "Diabetes",
    "Hypertension": "Hypertension",
    "Asthma": "Asthma",
    "Heart Disease": "Heart Disease",
    "Kidney Disease": "Kidney Disease",
    "Thyroid Disorder": "Thyroid Disorder"
}


def normalize_patient_gender(gender_val: str) -> str:
    """Normalizes any English or Kannada gender string to standard internal English token."""
    return REVERSE_GENDER_MAP_KN.get(gender_val, "Male")


def normalize_patient_conditions(cond_list: List[str]) -> List[str]:
    """Normalizes any English or Kannada conditions list to standard internal English tokens."""
    if not cond_list:
        return ["None"]
    normalized = [REVERSE_CONDITIONS_MAP_KN.get(c, c) for c in cond_list]
    # If 'None' is selected alongside other conditions, prioritize actual conditions
    if len(normalized) > 1 and "None" in normalized:
        normalized = [c for c in normalized if c != "None"]
    return normalized if normalized else ["None"]


def get_kannada_gender_display(gender_val: str) -> str:
    """Returns clean display name for gender in Kannada."""
    norm = normalize_patient_gender(gender_val)
    return GENDER_MAP_KN.get(norm, norm)


def get_kannada_conditions_display(cond_list: List[str]) -> str:
    """Returns formatted display string for patient pre-existing conditions in Kannada."""
    norm_list = normalize_patient_conditions(cond_list)
    active = [c for c in norm_list if c != "None"]
    if not active:
        return "ಯಾವುದೂ ಇಲ್ಲ (ಆರೋಗ್ಯಕರ ಸ್ಥಿತಿ)"
    kn_active = [CONDITIONS_MAP_KN.get(c, c) for c in active]
    return ", ".join(kn_active)

