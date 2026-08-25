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

# Comprehensive NLP Synonym mapping for all 130+ symptoms
SYMPTOM_SYNONYMS: Dict[str, List[str]] = {
    "itching": ["itching", "itch", "itchy", "scratching", "pruritus", "skin itching", "itchiness", "itchy skin", "itching all over"],
    "skin_rash": ["skin rash", "rash", "rashes", "red marks", "breakout", "skin irritation", "erythema", "skin eruptions", "red rash", "skin rashes", "body rash"],
    "nodal_skin_eruptions": ["nodal skin eruptions", "skin eruptions", "bumps on skin", "nodules", "lumps on skin", "skin bumps"],
    "continuous_sneezing": ["continuous sneezing", "sneezing", "sneeze", "constant sneezing", "sneezing fits"],
    "shivering": ["shivering", "shiver", "trembling", "body shaking", "rigors", "shivers", "violent shivering"],
    "chills": ["chills", "feeling cold", "cold chills", "chilly", "goosebumps", "cold feeling", "shivering with cold"],
    "joint_pain": ["joint pain", "pain in joints", "knee pain", "knees hurt", "arthralgia", "elbow pain", "finger joint pain", "wrist pain"],
    "stomach_pain": ["stomach pain", "stomach ache", "tummy ache", "belly ache", "pain in stomach", "gastric pain", "stomach cramps", "gut pain", "burning stomach"],
    "acidity": ["acidity", "acid reflux", "heartburn", "burning chest", "sour burps", "acid regurgitation", "hyperacidity", "burning in food pipe", "acid problem"],
    "ulcers_on_tongue": ["ulcers on tongue", "tongue ulcers", "mouth ulcers", "canker sores", "sores in mouth", "tongue sore"],
    "muscle_wasting": ["muscle wasting", "muscle loss", "loss of muscle mass", "shrinking muscles", "muscle atrophy"],
    "vomiting": ["vomiting", "vomit", "throwing up", "puking", "emesis", "heaving", "food throwing", "vomitted", "barfing"],
    "burning_micturition": ["burning micturition", "burning urination", "pain when peeing", "burning urine", "painful urination", "dysuria", "burning pee", "burning sensation when urinating", "pain in urination"],
    "spotting_ urination": ["spotting urination", "blood in urine drops", "spotting urine", "scanty dark urine drops"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhaustion", "no energy", "feeling drained", "burnout", "extreme weakness", "lack of energy", "weariness", "feeling weak", "weakness", "weak", "body weakness", "exhausted"],
    "weight_gain": ["weight gain", "gaining weight", "putting on weight", "unexplained weight gain", "getting fat"],
    "anxiety": ["anxiety", "anxious", "nervous", "nervousness", "panic", "feeling uneasy", "worrying too much", "fearfulness"],
    "cold_hands_and_feets": ["cold hands and feets", "cold hands", "cold feet", "freezing extremities", "chilled palms", "cold feet and hands"],
    "mood_swings": ["mood swings", "moody", "sudden mood changes", "emotional instability", "irritability swings"],
    "weight_loss": ["weight loss", "losing weight", "rapid weight loss", "unintentional weight loss", "slimming rapidly"],
    "restlessness": ["restlessness", "restless", "unable to sit still", "agitation", "fidgeting"],
    "lethargy": ["lethargy", "lethargic", "sluggishness", "feeling sluggish", "laziness", "lack of motivation", "drowsy feeling"],
    "patches_in_throat": ["patches in throat", "white patches in throat", "throat spots", "throat coating", "tonsil patches"],
    "irregular_sugar_level": ["irregular sugar level", "fluctuating blood sugar", "high sugar", "unstable glucose", "sugar spikes"],
    "cough": ["cough", "coughing", "dry cough", "wet cough", "phlegmy cough", "hacking cough", "persistent cough", "caugh", "throat coughing"],
    "high_fever": ["high fever", "fever", "high temperature", "temperature", "hot body", "pyrexia", "feverish", "burning with fever", "spiking fever", "having fever", "running temperature", "running fever"],
    "sunken_eyes": ["sunken eyes", "deep hollow eyes", "hollowed eyes", "dark sunken eyes", "eyes sunken in"],
    "breathlessness": ["breathlessness", "shortness of breath", "cant breathe", "difficulty breathing", "dyspnea", "gasping for air", "winded", "labored breathing", "trouble breathing"],
    "sweating": ["sweating", "night sweats", "perspiring", "excessive sweat", "profuse sweating", "sweaty body", "sweat"],
    "dehydration": ["dehydration", "dehydrated", "very thirsty", "extreme thirst", "dry mouth and throat", "lack of water in body", "water loss"],
    "indigestion": ["indigestion", "bloating", "upset stomach", "dyspepsia", "food not digesting", "heavy stomach", "gas problem"],
    "headache": ["headache", "head pain", "head hurts", "throbbing head", "head heavy", "migraine ache", "forehead ache", "cephalalgia", "severe headache", "pain in head"],
    "yellowish_skin": ["yellowish skin", "yellow skin", "jaundice skin", "icterus", "skin looking yellow", "pale yellow skin"],
    "dark_urine": ["dark urine", "brown urine", "tea colored urine", "deep yellow urine", "dark colored pee"],
    "nausea": ["nausea", "feeling sick", "queasy", "nauseous", "feeling like vomiting", "sick to stomach", "urge to vomit"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "not hungry", "reduced eating", "anorexia", "poor appetite", "dont feel like eating"],
    "pain_behind_the_eyes": ["pain behind the eyes", "eye socket pain", "retro orbital pain", "eye pain on moving", "behind eyes aching"],
    "back_pain": ["back pain", "back ache", "lower back pain", "upper back pain", "spine pain", "lumbago", "back hurts"],
    "constipation": ["constipation", "constipated", "hard stool", "difficulty passing stool", "irregular bowel"],
    "abdominal_pain": ["abdominal pain", "belly pain", "stomach cramps", "lower abdominal pain", "cramping belly", "gut cramps", "tummy pain"],
    "diarrhoea": ["diarrhoea", "diarrhea", "loose motion", "loose motions", "watery stool", "loose stools", "frequent motions", "running stomach", "watery motions", "dysentery", "loose stool", "stomach loose"],
    "mild_fever": ["mild fever", "low fever", "slight fever", "low grade fever", "warm body"],
    "yellow_urine": ["yellow urine", "bright yellow urine", "dark yellow pee"],
    "yellowing_of_eyes": ["yellowing of eyes", "yellow eyes", "scleral icterus", "whites of eyes turned yellow"],
    "acute_liver_failure": ["acute liver failure", "liver failure", "hepatic dysfunction"],
    "fluid_overload": ["fluid overload", "water retention", "edema", "body swelling with fluid"],
    "swelling_of_stomach": ["swelling of stomach", "stomach distension", "ascites", "swollen abdomen", "belly swelling"],
    "swelled_lymph_nodes": ["swelled lymph nodes", "swollen glands", "swollen lymph nodes", "neck swelling", "gland swelling", "lymphadenopathy"],
    "malaise": ["malaise", "general discomfort", "feeling unwell", "body unwell", "overall sickness"],
    "blurred_and_distorted_vision": ["blurred and distorted vision", "blurred vision", "blurry vision", "distorted vision", "foggy vision", "hazy sight"],
    "phlegm": ["phlegm", "mucus", "sputum", "coughing mucus", "thick phlegm", "catarrh"],
    "throat_irritation": ["throat irritation", "sore throat", "throat pain", "scratchy throat", "raw throat", "pain swallowing", "itchy throat", "throat infection", "pain in throat"],
    "redness_of_eyes": ["redness of eyes", "red eyes", "bloodshot eyes", "pink eye", "eye inflammation"],
    "sinus_pressure": ["sinus pressure", "sinus pain", "facial pressure", "forehead pressure", "nasal sinus blockage"],
    "runny_nose": ["runny nose", "running nose", "dripping nose", "nasal discharge", "watery nose", "rhinorrhea", "cold in nose", "sniffles"],
    "congestion": ["congestion", "blocked nose", "stuffy nose", "nasal blockage", "clogged nose", "stuffy nasal passages", "nose block", "chest congestion"],
    "chest_pain": ["chest pain", "pain in chest", "heart pain", "chest tightness", "chest pressure", "angina", "sternum pain"],
    "weakness_in_limbs": ["weakness in limbs", "weak arms", "weak legs", "loss of limb strength", "limbs feel heavy"],
    "fast_heart_rate": ["fast heart rate", "tachycardia", "racing heart", "rapid pulse", "quick heartbeat"],
    "pain_during_bowel_movements": ["pain during bowel movements", "painful defecation", "pain passing stool", "anal pain on stool"],
    "pain_in_anal_region": ["pain in anal region", "anal pain", "rectal pain", "pain in bottom", "pain around anus"],
    "bloody_stool": ["bloody stool", "blood in stool", "rectal bleeding", "red blood in feces", "hematochezia"],
    "irritation_in_anus": ["irritation in anus", "anal itching", "pruritus ani", "burning in anus"],
    "neck_pain": ["neck pain", "neck ache", "stiff neck pain", "cervical pain", "sore neck"],
    "dizziness": ["dizziness", "dizzy", "giddy", "head spinning", "lightheaded", "lightheadedness", "fainting sensation"],
    "cramps": ["cramps", "muscle cramps", "leg cramps", "calf cramps", "spasms"],
    "bruising": ["bruising", "easy bruising", "blue marks on skin", "ecchymosis", "skin bruises"],
    "obesity": ["obesity", "overweight", "excessive body fat", "morbid obesity"],
    "swollen_legs": ["swollen legs", "leg swelling", "swollen feet", "pedal edema", "puffy legs"],
    "swollen_blood_vessels": ["swollen blood vessels", "enlarged veins", "visible blue veins", "bulging veins"],
    "puffy_face_and_eyes": ["puffy face and eyes", "puffy face", "facial swelling", "swollen eyelids", "puffy eyes"],
    "enlarged_thyroid": ["enlarged thyroid", "goiter", "swollen neck front", "thyroid swelling"],
    "brittle_nails": ["brittle nails", "breaking nails", "fragile nails", "cracking nails"],
    "swollen_extremeties": ["swollen extremeties", "swollen hands and feet", "extremity swelling"],
    "excessive_hunger": ["excessive hunger", "always hungry", "polyphagia", "increased appetite", "constant hunger"],
    "drying_and_tingling_lips": ["drying and tingling lips", "tingling lips", "dry numb lips", "lip numbness"],
    "slurred_speech": ["slurred speech", "difficulty speaking", "garbled speech", "incoherent speech"],
    "knee_pain": ["knee pain", "pain in knees", "knee joint ache", "knees hurting"],
    "hip_joint_pain": ["hip joint pain", "pain in hip", "hip ache"],
    "muscle_weakness": ["muscle weakness", "weak muscles", "loss of muscle strength", "myasthenia"],
    "stiff_neck": ["stiff neck", "neck stiffness", "inability to bend neck", "rigid neck"],
    "swelling_joints": ["swelling joints", "swollen joints", "joint swelling", "puffy joints"],
    "movement_stiffness": ["movement stiffness", "stiff body", "stiffness in morning", "joint rigidity"],
    "spinning_movements": ["spinning movements", "room spinning", "vertigo sensation", "spinning head"],
    "loss_of_balance": ["loss of balance", "unsteadiness", "off balance", "imbalance walking", "stumbling", "poor coordination"],
    "unsteadiness": ["unsteadiness", "wobbly walking", "shaky footing", "instability"],
    "weakness_of_one_body_side": ["weakness of one body side", "hemiparesis", "one side weak", "paralysis of one side"],
    "loss_of_smell": ["loss of smell", "anosmia", "cannot smell", "lost sense of smell"],
    "bladder_discomfort": ["bladder discomfort", "pelvic discomfort", "full bladder pressure", "bladder ache", "bladder pain"],
    "foul_smell_of urine": ["foul smell of urine", "smelly urine", "malodorous urine", "stinky pee"],
    "continuous_feel_of_urine": ["continuous feel of urine", "frequent urge to pee", "urgency urination", "feeling like peeing continuously"],
    "passage_of_gases": ["passage of gases", "flatulence", "excessive gas", "burping and farting", "passing wind"],
    "internal_itching": ["internal itching", "deep itching", "internal prickling sensation"],
    "toxic_look_(typhos)": ["toxic look (typhos)", "typhoid look", "toxic appearance", "severely sick appearance", "dull lethargic appearance"],
    "depression": ["depression", "depressed mood", "feeling very low", "sadness", "hopelessness"],
    "irritability": ["irritability", "irritable", "cranky", "getting angry easily", "short tempered"],
    "muscle_pain": ["muscle pain", "body ache", "body pain", "bodypain", "myalgia", "sore muscles", "whole body hurts", "body aches", "bodyache"],
    "altered_sensorium": ["altered sensorium", "confusion", "delirium", "disorientation", "altered consciousness"],
    "red_spots_over_body": ["red spots over body", "petechiae", "red dots on skin", "measles rash", "purpura", "red skin spots"],
    "belly_pain": ["belly pain", "lower stomach ache", "abdomen hurts", "gut pain"],
    "abnormal_menstruation": ["abnormal menstruation", "irregular periods", "heavy menstrual bleeding", "missed periods"],
    "dischromic _patches": ["dischromic  patches", "discolored patches", "skin discoloration", "tinea versicolor", "white skin patches"],
    "watering_from_eyes": ["watering from eyes", "watery eyes", "excessive tears", "lacrimation", "eyes watering"],
    "increased_appetite": ["increased appetite", "huge appetite", "eating frequently"],
    "polyuria": ["polyuria", "frequent urination", "peeing a lot", "excessive urination", "urination at night"],
    "family_history": ["family history", "hereditary", "genetics", "runs in family"],
    "mucoid_sputum": ["mucoid sputum", "thick white phlegm", "sticky sputum", "mucus cough"],
    "rusty_sputum": ["rusty sputum", "brownish phlegm", "blood tinged sputum"],
    "lack_of_concentration": ["lack of concentration", "brain fog", "cannot focus", "difficulty concentrating"],
    "visual_disturbances": ["visual disturbances", "aura", "flashing lights", "blind spots in vision"],
    "coma": ["coma", "unconscious", "unresponsive", "passed out deeply"],
    "stomach_bleeding": ["stomach bleeding", "vomiting blood", "black tarry stool", "hematemesis", "melena"],
    "distention_of_abdomen": ["distention of abdomen", "swollen belly", "bloated abdomen", "abdominal swelling"],
    "blood_in_sputum": ["blood in sputum", "coughing blood", "hemoptysis", "red blood in phlegm"],
    "prominent_veins_on_calf": ["prominent veins on calf", "spider veins", "twisted veins on legs", "engorged calf veins"],
    "palpitations": ["palpitations", "heart racing", "fluttering heart", "skipped heartbeats", "pounding heart"],
    "painful_walking": ["painful walking", "limping", "hurts to walk", "difficulty walking due to pain"],
    "pus_filled_pimples": ["pus filled pimples", "acne pustules", "pus pimples", "zits with pus", "boils on face"],
    "blackheads": ["blackheads", "clogged pores", "comedones", "open comedones"],
    "scurring": ["scurring", "acne scars", "scarring", "pockmarks"],
    "skin_peeling": ["skin peeling", "flaking skin", "desquamation", "peeling epidermis"],
    "silver_like_dusting": ["silver like dusting", "silvery scales", "psoriasis scales", "white flaky crusts"],
    "small_dents_in_nails": ["small dents in nails", "nail pitting", "pitted fingernails"],
    "inflammatory_nails": ["inflammatory nails", "nail inflammation", "swollen nail beds", "paronychia"],
    "blister": ["blister", "blisters", "skin vesicles", "fluid filled blisters", "bullae"],
    "red_sore_around_nose": ["red sore around nose", "sores near nostrils", "crusty nose sores"],
    "yellow_crust_ooze": ["yellow crust ooze", "honey colored crust", "oozing golden crust", "impetigo crust"],
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
    against the 132 valid model features.
    """
    if not text:
        return {}, [0] * len(valid_features)

    # Normalize text
    cleaned = text.lower()
    cleaned = re.sub(r"[,/;&+]", " , ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Pre-clean punctuation
    alphanumeric_clean = re.sub(r"[^a-z0-9\s_]", " ", cleaned)

    matched: Dict[str, str] = {}

    # 1. Match multi-word synonyms first (ordered by phrase length descending)
    for symptom, phrases in SYMPTOM_SYNONYMS.items():
        col = f"has_{symptom}"
        if col not in valid_features:
            continue
        for phrase in sorted(phrases, key=len, reverse=True):
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, alphanumeric_clean):
                matched[symptom] = col
                break

    # 2. Match exact feature tokens from valid_features
    for col in valid_features:
        sym = col.replace("has_", "")
        token_phrase = sym.replace("_", " ")
        if token_phrase in alphanumeric_clean and sym not in matched:
            matched[sym] = col

    # 3. Handle special colloquial combos
    # E.g. "cold" alone usually implies congestion / runny nose / chills
    if "cold" in alphanumeric_clean and not any(k in matched for k in ["runny_nose", "congestion", "chills"]):
        if "has_congestion" in valid_features:
            matched["congestion"] = "has_congestion"
        if "has_chills" in valid_features:
            matched["chills"] = "has_chills"

    # E.g. "body pain" or "body ache" or "bodypain"
    if any(bp in alphanumeric_clean for bp in ["bodypain", "body pain", "body ache", "bodyaches", "myalgia", "body hurts"]):
        if "has_muscle_pain" in valid_features:
            matched["muscle_pain"] = "has_muscle_pain"

    # E.g. "loose motion" or "watery motion" or "food poisoning"
    if any(lm in alphanumeric_clean for lm in ["loose motion", "loose motions", "watery motion", "food poisoning"]):
        if "has_diarrhoea" in valid_features:
            matched["diarrhoea"] = "has_diarrhoea"
        if "has_vomiting" in valid_features and "vomit" in alphanumeric_clean:
            matched["vomiting"] = "has_vomiting"

    # E.g. "bladder pain" or "bladder discomfort"
    if any(bp in alphanumeric_clean for bp in ["bladder pain", "bladder ache", "bladder discomfort", "pelvic discomfort"]):
        if "has_bladder_discomfort" in valid_features:
            matched["bladder_discomfort"] = "has_bladder_discomfort"

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

    if days >= 7 and severity >= 7:
        return (
            "See Doctor Immediately",
            "Severe symptoms persisting for over a week require professional in-person medical evaluation and diagnostic testing.",
            [f"Persistent duration ({days} days)", f"High severity rating ({severity}/10)"],
            vuln_score,
            vuln_tier
        )

    if days >= 4 or severity >= 6:
        return (
            "See Doctor Soon",
            "Consult a physician within 24-48 hours for definitive diagnosis and targeted prescription.",
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
