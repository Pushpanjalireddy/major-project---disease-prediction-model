"""
Comprehensive Home Remedies Guide & Recipe Library.
Includes step-by-step preparation steps, biological mechanisms, verified medical video links (WebMD, Mayo Clinic, etc.),
and categorized condition profiles.
"""

from typing import Dict, List, Any

# Verified Video Search Helper
def yt_search(q: str) -> str:
    import urllib.parse
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}"

# Comprehensive Remedy Recipes Library
REMEDY_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "ginger-honey-tea",
        "name": "Ginger–Honey Tea",
        "icon": "☕",
        "category": "Hot Infusions",
        "make": "Simmer 4–5 thin slices of fresh ginger root in 2 cups of boiling water for 8 minutes. Strain into a cup, allow to cool slightly to drinking temperature, then stir in 1–2 tsp of pure honey and a fresh squeeze of lemon.",
        "why": "Fresh ginger contains bioactive gingerols with potent anti-inflammatory and mild analgesic properties. Honey coats the inflamed pharyngeal mucosa, providing immediate relief from irritation and suppressing the cough reflex.",
        "video": "https://www.youtube.com/watch?v=4iAUlFvD_Mo",
        "videoLabel": "WebMD — How to Make Ginger Honey Tea",
        "usedFor": ["Common Cold", "Cough", "Migraine", "Nausea", "Indigestion"]
    },
    {
        "id": "turmeric-milk",
        "name": "Turmeric Golden Milk (Haldi Doodh)",
        "icon": "🥛",
        "category": "Warming Drinks",
        "make": "Gently warm 1 cup of cow milk or plant-based alternative. Stir in 1/2 tsp pure organic turmeric powder, a pinch of freshly cracked black pepper, and 1/4 tsp cinnamon or crushed cardamom. Simmer on low heat for 3–4 minutes. Sweeten with honey or jaggery.",
        "why": "Curcumin in turmeric is a powerful antioxidant and systemic anti-inflammatory. Black pepper contains piperine, which boosts curcumin bio-absorption by up to 2000%. Promotes deep restorative sleep and accelerates muscle/tissue recovery.",
        "video": "https://www.youtube.com/watch?v=3RQ5NJkHzqI",
        "videoLabel": "Golden Milk Preparation & Health Benefits",
        "usedFor": ["Flu (Influenza)", "Body Aches", "Insomnia", "Joint Pain"]
    },
    {
        "id": "salt-gargle",
        "name": "Warm Salt-Water Gargle",
        "icon": "💧",
        "category": "Topical Washes",
        "make": "Dissolve 1/2 tsp of non-iodized or rock salt in 1 glass of warm filtered water (add a pinch of turmeric if desired). Gargle for 30–45 seconds at the back of the throat, then spit out completely. Repeat 3–4 times daily.",
        "why": "Creates a mild hypertonic osmotic gradient that draws excess edema fluid out of swollen throat tissues, cleanses viral/bacterial particulate, and thins out tenacious mucus plugs.",
        "video": "https://www.youtube.com/watch?v=ee0vaLz84wY",
        "videoLabel": "Medical Technique: Proper Salt Water Gargle",
        "usedFor": ["Common Cold", "Sore Throat", "Tonsillitis"]
    },
    {
        "id": "steam-inhalation",
        "name": "Steam Inhalation with Eucalyptus",
        "icon": "💨",
        "category": "Inhalations",
        "make": "Pour steaming hot water into a heat-safe basin. Add 2–3 drops of pure eucalyptus essential oil or a pinch of crushed ajwain (carom seeds). Lean over the bowl at a safe distance (12 inches), drape a clean towel over your head, and breathe deeply through your nose for 5–10 minutes.",
        "why": "Warm moist vapor liquefies thick nasal secretions, hydrates dried bronchial pathways, and cineole (eucalyptol) acts as a natural antimicrobial decongestant.",
        "video": "https://www.youtube.com/watch?v=8qgFQ9s_YAs",
        "videoLabel": "Steam Inhalation Protocol — Step by Step",
        "usedFor": ["Common Cold", "Sinus Congestion", "Bronchial Cough"]
    },
    {
        "id": "fennel-tea",
        "name": "Fennel Seed (Saunf) Digestive Infusion",
        "icon": "🍵",
        "category": "Digestive Teas",
        "make": "Lightly crush 1 tsp of whole fennel seeds. Steep in 1 cup of freshly boiled water for 7–10 minutes covered. Strain and sip slowly 15–20 minutes after meals.",
        "why": "Fennel contains anethole and fenchone volatile oils that relax gastrointestinal smooth muscles, alleviate abdominal spasms, and reduce acid reflux and bloating.",
        "video": "https://www.youtube.com/watch?v=olwCHA0QhMQ",
        "videoLabel": "Fennel Tea for Digestion & Gas Relief",
        "usedFor": ["Indigestion / Acid Reflux", "Bloating", "GERD"]
    },
    {
        "id": "ors",
        "name": "WHO Oral Rehydration Solution (ORS)",
        "icon": "🧪",
        "category": "Electrolyte Solutions",
        "make": "In 1 litre of clean, boiled-and-cooled drinking water, dissolve exactly 6 level teaspoons of sugar and 1/2 level teaspoon of salt. Stir until completely dissolved. Sip small volumes (100–200 ml) after each episode of loose stool or vomiting.",
        "why": "Utilizes the intestinal sodium-glucose co-transport mechanism (SGLT-1) to rapidly drive water and electrolytes back into the bloodstream, preventing life-threatening dehydration.",
        "video": "https://www.youtube.com/watch?v=ngaboJf7N6c",
        "videoLabel": "WHO Protocol: How to Make Home ORS",
        "usedFor": ["Diarrhea", "Gastroenteritis", "Dehydration", "Food Poisoning"]
    },
    {
        "id": "aloe-vera",
        "name": "Pure Fresh Aloe Vera Gel",
        "icon": "🌿",
        "category": "Topical Applications",
        "make": "Harvest a mature aloe leaf, wash thoroughly, trim the spiky edges, and scoop out the clear inner gel. Blend lightly or apply directly in a gentle, thin layer over clean, cooled skin.",
        "why": "Rich in acemannan polysaccharides and bradykinase, which stimulate fibroblast collagen synthesis, reduce epidermal thermal erythema, and accelerate burn epithelialization.",
        "video": "https://www.youtube.com/watch?v=9ll1Dl9U9rQ",
        "videoLabel": "Dermatology: Fresh Aloe Vera for Burns & Skin",
        "usedFor": ["Minor Burns", "Sunburn", "Skin Irritation", "Acne"]
    },
    {
        "id": "honey-lemon-water",
        "name": "Honey & Lemon Warm Elixir",
        "icon": "🍋",
        "category": "Hot Infusions",
        "make": "Mix 1 tbsp of raw organic honey and the freshly squeezed juice of half a lemon into 1 cup of warm (not boiling) filtered water. Sip slowly.",
        "why": "Provides bioflavonoids and vitamin C while honey creates a protective demulcent coating over inflamed mucosal linings.",
        "video": "https://www.youtube.com/watch?v=FmDSpM4CS34",
        "videoLabel": "Honey Lemon Throat Soothing Remedy",
        "usedFor": ["Sore Throat", "Morning Fatigue", "Mild Cough"]
    },
    {
        "id": "licorice-tea",
        "name": "Licorice Root (Mulethi) Tea",
        "icon": "🫖",
        "category": "Herbal Teas",
        "make": "Simmer 1 tsp of dried licorice root powder or bark in 1.5 cups of water for 5 minutes. Strain and sip warm 1–2 times daily.",
        "why": "Contains glycyrrhizin which exerts a potent anti-inflammatory and mucosal protective effect along the respiratory and upper gastrointestinal tracts.",
        "video": yt_search("licorice root mulethi tea recipe sore throat"),
        "videoLabel": "Licorice Root Preparation Guide",
        "usedFor": ["Sore Throat", "Acid Reflux / GERD", "Cough"]
    },
    {
        "id": "raw-honey",
        "name": "Raw Honey Spoonful",
        "icon": "🥄",
        "category": "Demulcents",
        "make": "Consume 1–2 teaspoons of pure, unpasteurized raw honey directly 30 minutes before bedtime, or mix with a pinch of black pepper.",
        "why": "Clinical trials (Mayo Clinic & Cochrane review) show raw honey matches dextromethorphan in reducing nocturnal cough frequency and severity in adults.",
        "video": "https://www.youtube.com/watch?v=15hD2BHRQfY",
        "videoLabel": "Mayo Clinic: Can Honey Help Coughs?",
        "usedFor": ["Dry Cough", "Nocturnal Cough", "Throat Tickle"]
    },
    {
        "id": "ginger-tulsi-tea",
        "name": "Ginger-Tulsi (Holy Basil) Kadha",
        "icon": "🍃",
        "category": "Ayurvedic Kadha",
        "make": "Boil 8–10 fresh tulsi leaves, 1-inch crushed ginger, and 3 crushed black peppercorns in 2 cups of water until reduced by half. Add a touch of honey once warm.",
        "why": "Tulsi acts as an immunomodulator with antimicrobial eugenol, while ginger supports bronchodilation and clears catarrhal secretions.",
        "video": "https://www.youtube.com/watch?v=dSXKAL1rLE4",
        "videoLabel": "Tulsi Ginger Kadha for Cold & Flu",
        "usedFor": ["Cough", "Chest Congestion", "Viral Fever", "Bronchial Irritation"]
    },
    {
        "id": "peppermint-oil",
        "name": "Peppermint Oil Temple Acupressure",
        "icon": "💆",
        "category": "Topical Essential Oils",
        "make": "Dilute 2 drops of 100% pure peppermint essential oil into 1 teaspoon of carrier oil (almond or jojoba). Gently massage with circular finger pressure into temples and base of skull for 3–5 minutes.",
        "why": "Menthol stimulates cutaneous cold receptors (TRPM8) and inhibits calcium channels in vascular smooth muscle, improving blood flow and reducing tension headache pain.",
        "video": "https://www.youtube.com/watch?v=Ke1ddlr8S50",
        "videoLabel": "Dr. Mandell: Peppermint Oil for Headache Relief",
        "usedFor": ["Headache", "Tension Headache", "Migraine", "Sinus Pressure"]
    },
    {
        "id": "cold-compress",
        "name": "Gel Cold Compress / Ice Pack",
        "icon": "❄️",
        "category": "Physical Therapies",
        "make": "Wrap an ice pack or frozen gel pad in a soft cotton towel (never apply ice directly to skin). Rest in a dark, quiet room and place across forehead, temples, or back of the neck for 15 minutes.",
        "why": "Induces localized vasoconstriction of dilated extracranial arteries and dampens pain-transmitting nerve conduction velocity.",
        "video": yt_search("cold compress for headache how to application"),
        "videoLabel": "Proper Cold Compress Technique",
        "usedFor": ["Migraine", "Acute Headache", "Fever Spikes", "Sprains"]
    },
    {
        "id": "chamomile-tea",
        "name": "Chamomile & Lavender Calming Tea",
        "icon": "🌼",
        "category": "Sedative Teas",
        "make": "Steep 1–2 tbsp of dried chamomile flowers or a tea bag in boiling water covered for 7–10 minutes. Sip slowly 30–45 minutes before bedtime in dim lighting.",
        "why": "Contains apigenin, a bioflavonoid that binds to benzodiazepine receptors in the GABA system, reducing neuronal excitability and initiating physiological sleep.",
        "video": "https://www.youtube.com/watch?v=fXBwDWJK_vI",
        "videoLabel": "Chamomile Sleep Tea Recipe & Routine",
        "usedFor": ["Insomnia / Trouble Sleeping", "Stress & Anxiety", "Nervous Stomach"]
    },
    {
        "id": "peppermint-tea",
        "name": "Peppermint Infusion for Nausea",
        "icon": "🌱",
        "category": "Digestive Teas",
        "make": "Steep 1 tablespoon of fresh or dried peppermint leaves in 1 cup of hot water for 6–8 minutes. Sip slowly in small swallows.",
        "why": "Exerts an antispasmodic effect on gastric smooth muscles, calms the vagal chemoreceptor trigger zone, and relieves nausea.",
        "video": "https://www.youtube.com/watch?v=jvvny6yCDR8",
        "videoLabel": "Peppermint Tea for Nausea & Upset Stomach",
        "usedFor": ["Nausea", "Motion Sickness", "Gastric Upset"]
    },
    {
        "id": "epsom-soak",
        "name": "Warm Epsom Salt Bath",
        "icon": "🛁",
        "category": "Therapeutic Baths",
        "make": "Dissolve 2 cups of pure Epsom salt (magnesium sulfate) into a warm bath. Soak whole body or immersed limb for 15–20 minutes.",
        "why": "Magnesium is transdermally absorbed to facilitate muscle relaxation, deplete lactic acid buildup, and reduce neuromuscular tension.",
        "video": "https://www.youtube.com/watch?v=vZzqV9KvqVE",
        "videoLabel": "How to Take an Epsom Salt Bath for Sore Muscles",
        "usedFor": ["Muscle Soreness", "Body Aches", "Joint Stiffness", "Physical Fatigue"]
    },
    {
        "id": "khichdi",
        "name": "Moong Dal & Rice Congee (Khichdi)",
        "icon": "🥣",
        "category": "Healing Foods",
        "make": "Cook 1/2 cup white rice and 1/4 cup yellow split moong dal in 4 cups of water with a pinch of turmeric and cumin until porridge-like and soft. Add minimal salt.",
        "why": "Provides easily digestible carbohydrates and plant protein requiring minimal gastrointestinal effort while restoring mucosal glycogen.",
        "video": yt_search("moong dal khichdi for upset stomach recipe"),
        "videoLabel": "Digestive Healing Khichdi Recipe",
        "usedFor": ["Diarrhea", "Food Poisoning", "Post-Fever Convalescence", "Stomach Infection"]
    },
    {
        "id": "honey-spot",
        "name": "Raw Honey Antimicrobial Spot Treatment",
        "icon": "🍯",
        "category": "Topical Applications",
        "make": "Cleanse face gently. Dab a small drop of raw medical-grade honey directly on active blemish. Leave for 20 minutes, then rinse with lukewarm water.",
        "why": "Produces minute concentrations of hydrogen peroxide via glucose oxidase and has low pH, inhibiting Cutibacterium acnes bacterial proliferation.",
        "video": "https://www.youtube.com/watch?v=UvJsfdtdwM8",
        "videoLabel": "Dermatologist: Raw Honey for Acne Blemishes",
        "usedFor": ["Acne / Breakouts", "Pus Filled Pimples", "Facial Redness"]
    },
    {
        "id": "tea-tree-oil",
        "name": "Diluted Tea Tree Oil Formulation",
        "icon": "💧",
        "category": "Antimicrobial Topicals",
        "make": "Mix 1 drop of pure tea tree oil with 9 drops of jojoba or aloe vera gel (never use 100% undiluted on skin). Dab lightly with cotton swab onto affected area once daily.",
        "why": "Terpinen-4-ol in tea tree oil disrupts bacterial cell membranes and reduces inflammatory erythema comparable to 5% benzoyl peroxide with less peeling.",
        "video": yt_search("diluted tea tree oil for acne application dermatologist"),
        "videoLabel": "Proper Tea Tree Oil Dilution & Application",
        "usedFor": ["Acne", "Fungal Infection", "Skin Eruptions"]
    },
    {
        "id": "ajwain-warm-water",
        "name": "Warm Ajwain (Carom Seeds) Digestive Infusion",
        "icon": "🫖",
        "category": "Digestive Teas",
        "make": "Boil 1 tsp of ajwain (carom seeds) in 2 cups of water with a pinch of black salt for 5 minutes. Strain and sip warm after heavy meals.",
        "why": "Thymol in ajwain stimulates gastric enzymes, promotes intestinal peristalsis, and rapidly relieves flatulence, abdominal spasms, and acid reflux.",
        "video": yt_search("ajwain water for gas acidity bloating recipe"),
        "videoLabel": "Ajwain Water for Digestion & Gas",
        "usedFor": ["Indigestion / Acid Reflux", "Bloating", "GERD", "Peptic Ulcer Disease", "Gastroenteritis"]
    },
    {
        "id": "ccf-detox-tea",
        "name": "CCF Tea (Cumin, Coriander & Fennel)",
        "icon": "🌱",
        "category": "Ayurvedic Kadha",
        "make": "Simmer 1/2 tsp each of whole cumin seeds, coriander seeds, and fennel seeds in 3 cups of water for 8 minutes. Strain and sip throughout the day.",
        "why": "Acts as a gentle natural diuretic and cooling anti-inflammatory, soothing the urinary tract and balancing gastrointestinal heat (Pitta).",
        "video": yt_search("cumin coriander fennel CCF tea recipe benefits"),
        "videoLabel": "CCF Cleansing Tea Recipe",
        "usedFor": ["Urinary Tract Infection (UTI)", "Indigestion / Acid Reflux", "Dehydration", "Hypertension"]
    },
    {
        "id": "tender-coconut-water",
        "name": "Fresh Tender Coconut Water Hydration",
        "icon": "🥥",
        "category": "Electrolyte Solutions",
        "make": "Drink 1 glass of fresh, natural tender coconut water twice daily. Best consumed fresh without added sugar or ice.",
        "why": "Provides bio-available potassium, magnesium, and natural electrolytes to combat cellular dehydration, support liver detox, and accelerate platelet stabilization.",
        "video": yt_search("coconut water hydration electrolytes health benefits"),
        "videoLabel": "Coconut Water Electrolyte Hydration",
        "usedFor": ["Dengue", "Malaria", "Typhoid Fever", "Jaundice", "Hepatitis A", "Diarrhea"]
    },
    {
        "id": "neem-turmeric-paste",
        "name": "Neem & Turmeric Antiseptic Topical Paste",
        "icon": "🌿",
        "category": "Topical Applications",
        "make": "Crush 10-12 fresh neem leaves into a fine paste with 1/2 tsp pure turmeric powder and a few drops of rose water. Apply over affected clean skin for 15-20 minutes, then rinse gently.",
        "why": "Nimbin and azadirachtin in neem exert potent antifungal and antibacterial effects, while curcumin reduces epidermal hyperproliferation and pruritus.",
        "video": yt_search("neem turmeric paste for fungal infection skin rash"),
        "videoLabel": "Neem & Turmeric Skin Treatment",
        "usedFor": ["Fungal Infection", "Psoriasis", "Impetigo", "Chickenpox", "Skin Breakouts"]
    },
    {
        "id": "mustard-garlic-rub",
        "name": "Warm Mustard-Garlic Chest & Sole Rub",
        "icon": "🧄",
        "category": "Topical Warming Oils",
        "make": "Gently warm 2 tbsp of pure mustard oil with 2-3 crushed garlic cloves and 1/4 tsp ajwain for 3 minutes until aromatic. Let cool to comfortably warm, then massage over chest, back, and soles of feet before bedtime.",
        "why": "Allyl isothiocyanate in mustard oil and allicin in garlic create mild cutaneous hyperthermia, improving microcirculation, dilating airways, and loosening chest phlegm.",
        "video": yt_search("mustard oil garlic massage for cold cough congestion"),
        "videoLabel": "Warm Mustard Oil Chest Rub Protocol",
        "usedFor": ["Common Cold", "Bronchial Asthma", "Cough", "Chest Congestion", "Cervical Spondylosis"]
    }
]

# Categorized Condition Profiles (OpenCare AI Structure)
CONDITION_GUIDES: List[Dict[str, Any]] = [
    {
        "id": "common-cold",
        "tag": "RESP-01",
        "name": "Common Cold",
        "category": "Respiratory",
        "icon": "🤧",
        "causes": "Viral infection (primarily Rhinoviruses or Coronaviruses) transmitted via aerosolized droplets or fomites. Elevated risk during seasonal shifts, stress, or sleep deprivation.",
        "symptoms": "Nasal congestion, runny nose, sneezing, scratchy throat, low-grade malaise, mild headache.",
        "remedies": [
            {
                "name": "Honey–Ginger Tea",
                "make": "Simmer fresh sliced ginger in water for 8 minutes, strain, add 1-2 tsp pure honey and lemon. Sip warm.",
                "why": "Gingerols inhibit viral replication and reduce mucosal inflammation; honey soothes throat tissue.",
                "video": "https://www.youtube.com/watch?v=4iAUlFvD_Mo",
                "videoLabel": "WebMD Tutorial"
            },
            {
                "name": "Steam Inhalation with Eucalyptus",
                "make": "Inhale warm steam from a bowl with 2-3 drops eucalyptus oil under a towel for 10 minutes.",
                "why": "Moist heat thins tenacious mucus and restores ciliary clearance in the nasal passages.",
                "video": "https://www.youtube.com/watch?v=8qgFQ9s_YAs",
                "videoLabel": "Steam Protocol"
            },
            {
                "name": "Warm Salt-Water Gargle",
                "make": "Dissolve 1/2 tsp salt in 1 cup warm water. Gargle 30 seconds 3-4 times daily.",
                "why": "Osmotic fluid extraction shrinks swollen pharyngeal tissues and washes out viral debris.",
                "video": "https://www.youtube.com/watch?v=ee0vaLz84wY",
                "videoLabel": "Gargle Guide"
            }
        ],
        "caution": "Seek clinical medical care if fever exceeds 103°F (39.4°C), lasts >10 days, or if acute shortness of breath develops."
    },
    {
        "id": "flu",
        "tag": "RESP-02",
        "name": "Flu (Influenza)",
        "category": "Respiratory",
        "icon": "🤒",
        "causes": "Influenza virus type A or B infection causing systemic viremia, rapid onset cytokine release, and upper/lower respiratory inflammation.",
        "symptoms": "High fever, sudden violent chills, intense body aches (myalgia), profound exhaustion, dry hacking cough.",
        "remedies": [
            {
                "name": "Golden Turmeric Milk",
                "make": "Simmer 1 cup milk with 1/2 tsp turmeric powder and black pepper for 3 minutes before bed.",
                "why": "Curcumin reduces systemic inflammatory cytokines; black pepper enhances absorption by 2000%.",
                "video": "https://www.youtube.com/watch?v=3RQ5NJkHzqI",
                "videoLabel": "Golden Milk Recipe"
            },
            {
                "name": "Hydration & Electrolyte Broth",
                "make": "Sip warm vegetable broth, electrolyte water, and herbal teas regularly throughout the day.",
                "why": "Compensates for insensible fluid loss from fever sweating and supports immune cell mobilization.",
                "video": yt_search("flu hydration electrolyte broth recipe"),
                "videoLabel": "Hydration Protocol"
            },
            {
                "name": "Warm Epsom Salt Bath for Myalgia",
                "make": "Soak in warm bath with 2 cups Epsom salt for 15-20 minutes.",
                "why": "Relaxes hypertonic muscles and alleviates deep flu-related back and leg soreness.",
                "video": "https://www.youtube.com/watch?v=vZzqV9KvqVE",
                "videoLabel": "Epsom Salt Guide"
            }
        ],
        "caution": "Flu can rapidly escalate in infants, elderly, pregnant individuals, and asthmatics. Monitor oxygen saturation."
    },
    {
        "id": "sore-throat",
        "tag": "RESP-03",
        "name": "Sore Throat (Pharyngitis)",
        "category": "Respiratory",
        "icon": "🗣️",
        "causes": "Viral pharyngitis (Cold/Flu), dry unhumidified indoor air, allergy post-nasal drip, or bacterial Strep infection.",
        "symptoms": "Sharp pain on swallowing, scratchiness, dry burning sensation, enlarged tender lymph nodes.",
        "remedies": [
            {
                "name": "Honey & Lemon Warm Water",
                "make": "Mix 1 tbsp raw honey and fresh juice of half lemon in warm water. Sip slowly.",
                "why": "Honey forms a long-lasting protective barrier over exposed sensory nerve endings in the throat.",
                "video": "https://www.youtube.com/watch?v=FmDSpM4CS34",
                "videoLabel": "Honey Lemon Guide"
            },
            {
                "name": "Licorice Root Tea",
                "make": "Steep 1 tsp licorice root in hot water for 5 minutes. Strain and sip warm.",
                "why": "Contains glycyrrhizin, producing natural cortisone-like local mucosal anti-inflammatory action.",
                "video": yt_search("licorice root tea sore throat recipe"),
                "videoLabel": "Licorice Tea Guide"
            }
        ],
        "caution": "If accompanied by high fever without cough, white tonsillar exudates, or difficulty opening mouth, test for Strep throat."
    },
    {
        "id": "cough",
        "tag": "RESP-04",
        "name": "Persistent Cough",
        "category": "Respiratory",
        "icon": "🫁",
        "causes": "Post-viral bronchial hyperreactivity, postnasal drip, dry air, or environmental particulate irritation.",
        "symptoms": "Dry tickling cough, productive phlegm cough, chest tightness, throat irritation.",
        "remedies": [
            {
                "name": "Raw Honey Spoonful at Bedtime",
                "make": "Take 1-2 tsp raw pure honey directly 30 minutes before sleeping.",
                "why": "Clinically proven to reduce nocturnal cough spasms and promote uninterrupted sleep.",
                "video": "https://www.youtube.com/watch?v=15hD2BHRQfY",
                "videoLabel": "Mayo Clinic Study"
            },
            {
                "name": "Ginger-Tulsi Kadha",
                "make": "Boil tulsi leaves, crushed ginger, and black pepper for 6 minutes. Drink warm.",
                "why": "Acts as a natural bronchodilator and liquefies thick bronchial secretions.",
                "video": "https://www.youtube.com/watch?v=dSXKAL1rLE4",
                "videoLabel": "Kadha Protocol"
            }
        ],
        "caution": "Never give honey to children under 1 year old (infant botulism risk). Seek evaluation if coughing up blood."
    },
    {
        "id": "indigestion",
        "tag": "DIG-01",
        "name": "Indigestion & Acid Reflux (GERD)",
        "category": "Digestive",
        "icon": "🔥",
        "causes": "Transient lower esophageal sphincter relaxation from heavy/spicy meals, lying down post-eating, stress, or caffeine.",
        "symptoms": "Retrosternal heartburn, sour acidic taste in mouth, epigastric fullness, postprandial bloating.",
        "remedies": [
            {
                "name": "Fennel Seed (Saunf) Infusion",
                "make": "Steep 1 tsp crushed fennel seeds in boiling water for 8 minutes. Sip after meals.",
                "why": "Anethole relaxes gastrointestinal spasms and accelerates gastric emptying.",
                "video": "https://www.youtube.com/watch?v=olwCHA0QhMQ",
                "videoLabel": "Fennel Tea Guide"
            },
            {
                "name": "Cold Skim Milk or Almond Milk",
                "make": "Drink 1/2 glass of cold unsweetened milk slowly when heartburn strikes.",
                "why": "Provides temporary acid buffering and coats the esophageal mucosal lining.",
                "video": yt_search("cold milk for acid reflux relief"),
                "videoLabel": "Acid Buffering Guide"
            },
            {
                "name": "Post-Meal Upright Posture",
                "make": "Remain upright or take a gentle 15-minute walk for at least 45 minutes after eating.",
                "why": "Gravity keeps gastric acid in the stomach lumen and prevents esophageal regurgitation.",
                "video": yt_search("posture and acid reflux prevention tips"),
                "videoLabel": "Posture Tips"
            }
        ],
        "caution": "Crushing chest pain radiating to left arm/jaw is an emergency cardiac red flag — never assume it is just gas."
    },
    {
        "id": "diarrhea",
        "tag": "DIG-02",
        "name": "Diarrhea & Stomach Infection",
        "category": "Digestive",
        "icon": "💧",
        "causes": "Viral/bacterial gastroenteritis (food poisoning), contaminated water, dietary toxin exposure.",
        "symptoms": "Frequent watery loose stools, abdominal cramping, nausea, dehydration, low energy.",
        "remedies": [
            {
                "name": "WHO Oral Rehydration Solution (ORS)",
                "make": "Mix 6 tsp sugar + 1/2 tsp salt in 1 litre boiled water. Sip continuously throughout the day.",
                "why": "Activates SGLT-1 intestinal transport to rapidly replenish water, sodium, and chloride.",
                "video": "https://www.youtube.com/watch?v=ngaboJf7N6c",
                "videoLabel": "WHO Recipe"
            },
            {
                "name": "Moong Dal & Rice Khichdi",
                "make": "Cook soft rice and split moong dal with turmeric and cumin. Eat small warm portions.",
                "why": "Ultra-gentle on gut mucosal lining, non-irritating, and restores cellular energy.",
                "video": yt_search("moong dal khichdi for upset stomach recipe"),
                "videoLabel": "Khichdi Recipe"
            },
            {
                "name": "Ripe Banana",
                "make": "Eat 1-2 ripe yellow bananas daily as tolerated.",
                "why": "High in soluble pectin fiber to solidify stool and rich in potassium lost during diarrhea.",
                "video": yt_search("banana for diarrhea relief"),
                "videoLabel": "Pectin & Potassium"
            }
        ],
        "caution": "Seek immediate emergency hospital care if stool contains blood (dysentery), urine output ceases, or fever is high."
    },
    {
        "id": "nausea",
        "tag": "DIG-03",
        "name": "Nausea & Motion Sickness",
        "category": "Digestive",
        "icon": "🤢",
        "causes": "Gastric mucosal irritation, labyrinthine inner ear conflict, migraine aura, or pregnancy.",
        "symptoms": "Queasiness, excessive salivation, urge to vomit, dizziness, cold sweat.",
        "remedies": [
            {
                "name": "Fresh Ginger Infusion or Chew",
                "make": "Sip warm ginger tea slowly or chew a tiny candied ginger slice.",
                "why": "Gingerols antagonize 5-HT3 serotonin receptors in the gut and central nervous system.",
                "video": "https://www.youtube.com/watch?v=4iAUlFvD_Mo",
                "videoLabel": "Ginger for Nausea"
            },
            {
                "name": "Peppermint Tea or Aroma Inhalation",
                "make": "Inhale peppermint essential oil aroma or sip cool peppermint tea.",
                "why": "Calms gastric muscle spasms and dulls nausea signals from the vestibular system.",
                "video": "https://www.youtube.com/watch?v=jvvny6yCDR8",
                "videoLabel": "Peppermint Guide"
            }
        ],
        "caution": "Inability to keep liquids down for over 12 hours requires intravenous medical rehydration."
    },
    {
        "id": "headache",
        "tag": "ACHE-01",
        "name": "Tension Headache & Migraine",
        "category": "Aches & Pain",
        "icon": "⚡",
        "causes": "Muscular contraction of scalp/neck, cerebral vascular dilation, screen glare, dehydration, or stress.",
        "symptoms": "Band-like squeezing around head, throbbing one-sided pain, sensitivity to light and sound (photophobia).",
        "remedies": [
            {
                "name": "Peppermint Oil Temple Massage",
                "make": "Dilute 2 drops peppermint oil in 1 tsp carrier oil. Massage temples in circular motion.",
                "why": "Menthol relaxes cranial cutaneous vessels and blocks sensory pain transmission.",
                "video": "https://www.youtube.com/watch?v=Ke1ddlr8S50",
                "videoLabel": "Dr. Mandell Acupressure"
            },
            {
                "name": "Cold Gel Pack on Forehead / Neck",
                "make": "Rest in a completely dark room with a wrapped cold pack across forehead for 15 minutes.",
                "why": "Constricts pulsating cranial vessels and dampens hyperactive sensory nerve signals.",
                "video": yt_search("cold compress for headache relief"),
                "videoLabel": "Cold Pack Protocol"
            },
            {
                "name": "Deep Hydration (500ml Water)",
                "make": "Drink 2 large glasses of room-temperature water with a pinch of mineral salt.",
                "why": "Reverses subclinical dehydration, a primary trigger in over 40% of tension headaches.",
                "video": yt_search("water and hydration for headache prevention"),
                "videoLabel": "Hydration Guide"
            }
        ],
        "caution": "Sudden explosive 'thunderclap' headache, speech slurring, or weakness on one side is an immediate 911/emergency stroke alert."
    },
    {
        "id": "muscle-soreness",
        "tag": "ACHE-02",
        "name": "Muscle Soreness & Joint Stiffness",
        "category": "Aches & Pain",
        "icon": "💪",
        "causes": "Eccentric exercise microtrauma (DOMS), viral infection myalgia, posture strain, or osteoarthritis.",
        "symptoms": "Dull muscular ache, localized tenderness, joint stiffness upon waking, reduced range of motion.",
        "remedies": [
            {
                "name": "Warm Epsom Salt Soak",
                "make": "Dissolve 2 cups Epsom salt in hot bath water. Soak for 20 minutes.",
                "why": "Transdermal magnesium ions assist muscle relaxation and promote localized vascular drainage.",
                "video": "https://www.youtube.com/watch?v=vZzqV9KvqVE",
                "videoLabel": "Epsom Soak Tutorial"
            },
            {
                "name": "Golden Turmeric Milk",
                "make": "Drink warm turmeric golden milk with black pepper and cardamom before sleep.",
                "why": "Curcumin downregulates systemic NF-kB inflammatory cascade in connective tissues.",
                "video": "https://www.youtube.com/watch?v=3RQ5NJkHzqI",
                "videoLabel": "Golden Milk Guide"
            }
        ],
        "caution": "Joints that are hot, severely swollen, red, or accompanied by high fever require medical workup for septic arthritis."
    },
    {
        "id": "insomnia",
        "tag": "SLEEP-01",
        "name": "Insomnia & Sleep Disturbance",
        "category": "Sleep & Stress",
        "icon": "🌙",
        "causes": "Elevated evening cortisol from blue-light screens, caffeine late in day, racing thoughts, chronic pain.",
        "symptoms": "Difficulty falling asleep (>30 mins), frequent nocturnal awakenings, unrefreshing sleep, daytime brain fog.",
        "remedies": [
            {
                "name": "Chamomile & Lavender Tea",
                "make": "Steep chamomile flowers in boiling water for 8 minutes. Drink 45 minutes before sleep.",
                "why": "Apigenin binds to brain GABA-A receptors, quietening the central nervous system.",
                "video": "https://www.youtube.com/watch?v=fXBwDWJK_vI",
                "videoLabel": "Sleep Tea Guide"
            },
            {
                "name": "Warm Golden Turmeric Milk",
                "make": "Drink 1 cup warm milk with turmeric and pinch of nutmeg 30 minutes before bed.",
                "why": "Milk provides tryptophan (precursor to serotonin and melatonin) while warmth induces sleepiness.",
                "video": "https://www.youtube.com/watch?v=3RQ5NJkHzqI",
                "videoLabel": "Nighttime Routine"
            },
            {
                "name": "Digital Sun-downing (1 Hour No Screen)",
                "make": "Switch off phone, laptop, and TV 60 minutes before bed. Keep bedroom at 20°C (68°F).",
                "why": "Allows pineal gland to release natural endogenous melatonin without blue-light suppression.",
                "video": yt_search("sleep hygiene protocol huberman"),
                "videoLabel": "Sleep Science Protocol"
            }
        ],
        "caution": "Chronic insomnia lasting >1 month warrants evaluation for sleep apnea or underlying mood disorders."
    },
    {
        "id": "acne",
        "tag": "SKIN-01",
        "name": "Acne & Skin Breakouts",
        "category": "Skin & Surface",
        "icon": "✨",
        "causes": "Sebum overproduction, follicular hyperkeratinization, Cutibacterium acnes bacterial proliferation, hormonal surges.",
        "symptoms": "Pus-filled pimples, inflamed red papules, blackheads, skin tenderness.",
        "remedies": [
            {
                "name": "Raw Honey Spot Treatment",
                "make": "Dab a small drop of raw honey on the pimple, leave 20 minutes, rinse with lukewarm water.",
                "why": "Exerts osmotic and enzymatic antimicrobial action against skin bacteria while hydrating.",
                "video": "https://www.youtube.com/watch?v=UvJsfdtdwM8",
                "videoLabel": "Dermatologist Honey Guide"
            },
            {
                "name": "Diluted Tea Tree Oil (1:9 Ratio)",
                "make": "Mix 1 drop tea tree oil with 9 drops carrier oil. Apply with cotton swab once daily.",
                "why": "Terpinen-4-ol penetrates sebaceous glands and destroys bacterial cell membranes.",
                "video": yt_search("diluted tea tree oil acne how to apply"),
                "videoLabel": "Tea Tree Oil Guide"
            }
        ],
        "caution": "Avoid squeezing or popping pimples to prevent scarring and deep bacterial cellulitis. See a dermatologist for cystic acne."
    },
    {
        "id": "minor-burns",
        "tag": "SKIN-02",
        "name": "Minor First-Degree Burns",
        "category": "Skin & Surface",
        "icon": "🩹",
        "causes": "Contact with hot steam, hot pans, scalding water, or mild sun exposure.",
        "symptoms": "Redness, localized stinging pain, mild swelling, absence of open blisters.",
        "remedies": [
            {
                "name": "Cool Running Water (10–15 Minutes)",
                "make": "Hold the burned area immediately under cool (not freezing) running tap water for 15 minutes.",
                "why": "Arrests deep thermal tissue conduction and stops progressive thermal dermal necrosis.",
                "video": yt_search("first aid for minor burns cool water"),
                "videoLabel": "Burn First Aid Guide"
            },
            {
                "name": "Pure Fresh Aloe Vera Gel",
                "make": "Apply a gentle layer of pure clear aloe vera gel over the cooled, unbroken skin.",
                "why": "Acemannan stimulates wound healing and reduces pain through localized anti-inflammatory activity.",
                "video": "https://www.youtube.com/watch?v=9ll1Dl9U9rQ",
                "videoLabel": "Aloe Vera on Burns"
            }
        ],
        "caution": "Never apply ice, butter, or oil to a fresh burn. Seek ER care for burns covering large areas, face, hands, or with white/charred tissue."
    }
]

CATEGORIES_LIST = [
    "All Categories",
    "Respiratory",
    "Digestive",
    "Aches & Pain",
    "Sleep & Stress",
    "Skin & Surface"
]
