import csv
import json
import joblib
from clinical_knowledge import DISEASE_SYMPTOM_PROFILES, DISEASE_KNOWLEDGE

le = joblib.load('disease_label_encoder.joblib')
classes = list(le.classes_)

sample_prompts = {
    'Common Cold & Flu': 'I have continuous sneezing, runny nose, cough, chills and congestion',
    'Bronchial Asthma': 'I have severe breathlessness, wheezing, persistent cough and fatigue',
    'Dengue': 'I have sudden high fever, chills, severe headache and deep bone/back pain',
    'Malaria': 'I have high fever with violent shivering, chills, sweating and muscle pain',
    'Typhoid Fever': 'I have step-ladder high fever, severe abdominal belly pain, headache and fatigue',
    'Chickenpox': 'I have red spots over body, blister eruptions, severe itching and high fever',
    'Gastroenteritis (Food Poisoning / Stomach Infection)': 'I have severe vomiting, dehydration, loose motion and sunken eyes',
    'GERD (Acid Reflux)': 'I have severe acidity, chest burn, heartburn and sour stomach pain',
    'Peptic Ulcer Disease': 'I have burning abdominal stomach pain, severe indigestion, gas and loss of appetite',
    'Hepatitis A': 'I have yellowish skin, dark urine, joint pain, vomiting and loss of appetite',
    'Jaundice': 'I have yellowing of eyes, yellowish skin, dark urine, fatigue and itching',
    'Hemorrhoids (Piles)': 'I have severe constipation, pain during bowel movements, irritation and bloody stool',
    'Diabetes': 'I have excessive hunger, increased appetite, polyuria frequent urination and blurred vision',
    'Hypoglycemia (Low Blood Sugar)': 'I have sudden sweating, anxiety, trembling, blurred vision and tingling lips',
    'Hypothyroidism': 'I have chronic fatigue, abnormal weight gain, cold hands and feet and depression',
    'Hyperthyroidism': 'I have heart palpitations, rapid weight loss, excessive sweating and irritability',
    'Hypertension (High Blood Pressure)': 'I have dizziness, morning headache, lack of concentration and loss of balance',
    'Varicose Veins': 'I have prominent twisted veins on calf, heavy leg fatigue, cramps and bruising',
    'Migraine': 'I have throbbing headache on one side, visual disturbances aura and nausea',
    'Vertigo (BPPV)': 'I have spinning sensation of room, loss of balance, unsteadiness and nausea',
    'Urinary Tract Infection (UTI)': 'I have burning urination, foul smell of urine, bladder pain and continuous feel of urine',
    'Osteoarthritis': 'I have severe knee joint pain, hip joint pain, joint stiffness and painful walking',
    'Cervical Spondylosis': 'I have chronic neck pain, radiating back pain, dizziness and weakness in limbs',
    'Arthritis': 'I have swelling in joints, movement stiffness, joint pain and painful walking',
    'Acne': 'I have pus filled pimples on face, blackheads and inflamed red skin breakouts',
    'Fungal Infection': 'I have intense skin itching, discolored patches on skin and nodal eruptions',
    'Psoriasis': 'I have skin peeling, silver like dusting scales, skin rash and inflammatory nails',
    'Impetigo': 'I have yellow crust ooze on face, red sore around nose, blisters and skin rash',
    'Allergy': 'I have continuous sneezing, watering from eyes, shivering and chills',
    'Drug Reaction': 'I have sudden generalized skin rash, burning micturition and intense body itching'
}

categories = {
    'Common Cold & Flu': 'Respiratory & Viral',
    'Bronchial Asthma': 'Respiratory & Pulmonology',
    'Dengue': 'Tropical & Infectious',
    'Malaria': 'Tropical & Parasitic',
    'Typhoid Fever': 'Bacterial & Infectious',
    'Chickenpox': 'Viral & Dermatological',
    'Gastroenteritis (Food Poisoning / Stomach Infection)': 'Gastrointestinal',
    'GERD (Acid Reflux)': 'Gastrointestinal',
    'Peptic Ulcer Disease': 'Gastrointestinal',
    'Hepatitis A': 'Hepatic & Liver',
    'Jaundice': 'Hepatic & Liver',
    'Hemorrhoids (Piles)': 'Colorectal & Gastrointestinal',
    'Diabetes': 'Endocrine & Metabolic',
    'Hypoglycemia (Low Blood Sugar)': 'Endocrine & Metabolic',
    'Hypothyroidism': 'Endocrine & Thyroid',
    'Hyperthyroidism': 'Endocrine & Thyroid',
    'Hypertension (High Blood Pressure)': 'Cardiovascular',
    'Varicose Veins': 'Vascular',
    'Migraine': 'Neurological',
    'Vertigo (BPPV)': 'Neurological & ENT',
    'Urinary Tract Infection (UTI)': 'Urological',
    'Osteoarthritis': 'Orthopedic & Joint',
    'Cervical Spondylosis': 'Spine & Musculoskeletal',
    'Arthritis': 'Rheumatology',
    'Acne': 'Dermatological',
    'Fungal Infection': 'Dermatological (Mycotic)',
    'Psoriasis': 'Dermatological (Autoimmune)',
    'Impetigo': 'Dermatological (Bacterial)',
    'Allergy': 'Immunology',
    'Drug Reaction': 'Pharmacology & Allergy'
}

data = []
for i, d in enumerate(classes, 1):
    profile = [s.replace('_', ' ').title() for s in DISEASE_SYMPTOM_PROFILES.get(d, [])]
    k_info = DISEASE_KNOWLEDGE.get(d, {})
    remedy_list = k_info.get('home_remedies', [])
    first_remedy = remedy_list[0] if remedy_list else 'Hydration, balanced light nutrition, and adequate bed rest'
    
    data.append({
        'Index': i,
        'Disease': d,
        'Category': categories.get(d, 'General Medicine'),
        'Specialist': k_info.get('specialist', 'General Physician'),
        'Key_Symptoms': ', '.join(profile),
        'Ready_Test_Prompt': sample_prompts.get(d, f"I have {', '.join(profile[:4])}"),
        'Primary_Home_Remedy': first_remedy
    })

# 1. Export CSV
with open('30_Diseases_And_Symptoms_Test_Guide.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
    writer.writeheader()
    writer.writerows(data)

# 2. Export JSON
with open('30_Diseases_And_Symptoms_Test_Guide.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print('Export completed: 30_Diseases_And_Symptoms_Test_Guide.csv & 30_Diseases_And_Symptoms_Test_Guide.json')
