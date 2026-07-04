import streamlit as st
import numpy as np
import joblib
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained PCOS detection model
pcos_model = joblib.load('model\exported_model\data_model.pkl')

# Load the trained image model
image_model = load_model('model\exported_model\image_model.keras')  

# Main Title
st.title("Women's Health Assistant")
st.write("A one-stop solution for PCOS prediction, period tracking, diet planning, and image-based PCOS detection.")

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.write("Use the options below to navigate:")
navigation = st.sidebar.selectbox(
    "Choose a section",
    ["🏠 Home", "📚 PCOS Education", "🩺 PCOS Detection", "📅 Period Tracker", "🥗 Diet Plan", "🖼️ Image-based PCOS Detection"]
)

# Home Section
if navigation == "🏠 Home":
    st.header("Welcome to Women's Health Assistant")
    st.write(""" 
        This application provides several features:
        - 📚 **PCOS Education**: Comprehensive educational module about PCOS including types, causes, symptoms, myths, and prevention.
        - 🖼️ **Image-based PCOS Detection**: Uses an image of an ultrasound to detect PCOS.
        - 🩺 **PCOS Detection**: Predicts if a person is at risk of Polycystic Ovary Syndrome (PCOS) based on medical parameters.
        - 🥗 **Diet Plan**: Suggests a personalized diet plan based on weight, activity level, and dietary preferences.
        - 📅 **Period Tracker**: Tracks menstrual cycles and predicts the next period date.
    """)
    st.image("home.jpg", caption="Women's Health Assistant", use_container_width=True)
    st.write("Select a section from the sidebar to get started.")

# PCOS Education Module Section
elif navigation == "📚 PCOS Education":
    st.header("📚 PCOS Education Module")
    st.write("Comprehensive information about Polycystic Ovary Syndrome (PCOS)")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "What is PCOS?", 
        "Types of PCOS", 
        "Causes", 
        "Symptoms", 
        "Myths vs Facts", 
        "Prevention Tips"
    ])
    
    # Tab 1: What is PCOS?
    with tab1:
        st.subheader("What is PCOS?")
        st.markdown("""
        **Polycystic Ovary Syndrome (PCOS)** is a common hormonal disorder affecting women of reproductive age. 
        It is one of the most prevalent endocrine disorders, affecting approximately 5-10% of women worldwide.
        
        ### Key Characteristics:
        
        **PCOS is characterized by three main features:**
        
        1. **Irregular Menstrual Cycles**: Women with PCOS often experience irregular, infrequent, or prolonged periods
        2. **Excess Androgen Levels**: Higher than normal levels of male hormones (androgens) in the body
        3. **Polycystic Ovaries**: Ovaries may develop numerous small collections of fluid (follicles) and fail to regularly release eggs
        
        ### Understanding the Condition:
        
        - **Hormonal Imbalance**: PCOS involves an imbalance of reproductive hormones, particularly insulin and androgens
        - **Metabolic Impact**: It's not just a reproductive issue; PCOS affects metabolism and can lead to insulin resistance
        - **Chronic Condition**: PCOS is a lifelong condition that requires ongoing management
        - **Variable Presentation**: Symptoms vary widely among individuals, making diagnosis challenging
        
        ### Why It Matters:
        
        PCOS can affect various aspects of health including:
        - Reproductive health and fertility
        - Metabolic health (diabetes risk)
        - Cardiovascular health
        - Mental health and quality of life
        - Physical appearance and self-esteem
        
        ### Diagnosis:
        
        PCOS is typically diagnosed when at least two of the following three criteria are met:
        - Irregular or absent menstrual periods
        - Clinical or biochemical signs of excess androgens
        - Polycystic ovaries on ultrasound
        
        **Note**: Early diagnosis and management are crucial for preventing long-term complications.
        """)
    
    # Tab 2: Types of PCOS
    with tab2:
        st.subheader("Types of PCOS")
        st.markdown("""
        PCOS can be categorized into different types based on the underlying causes and characteristics. 
        Understanding the type helps in determining the most effective treatment approach.
        
        ### 1. Insulin-Resistant PCOS (Most Common - 70% of cases)
        
        **Characteristics:**
        - High insulin levels
        - Difficulty losing weight
        - Sugar cravings
        - Fatigue after meals
        - Dark patches on skin (acanthosis nigricans)
        
        **Root Cause**: Cells become resistant to insulin, leading to increased insulin production
        
        **Management Focus**: 
        - Low-glycemic diet
        - Regular exercise
        - Insulin-sensitizing medications (metformin)
        - Weight management
        
        ---
        
        ### 2. Post-Pill PCOS
        
        **Characteristics:**
        - Develops after stopping birth control pills
        - Temporary condition
        - Androgen levels may spike initially
        
        **Root Cause**: Hormonal readjustment after discontinuing oral contraceptives
        
        **Management Focus**:
        - Time and patience (usually resolves in 3-6 months)
        - Supporting natural hormone balance
        - Healthy lifestyle practices
        
        ---
        
        ### 3. Adrenal PCOS (10% of cases)
        
        **Characteristics:**
        - Elevated DHEA-S (adrenal androgen)
        - Normal ovarian androgens
        - High stress levels
        - Anxiety and sleep issues
        
        **Root Cause**: Overactive adrenal glands producing excess androgens, often stress-related
        
        **Management Focus**:
        - Stress management techniques
        - Adequate sleep
        - Adrenal support supplements
        - Meditation and relaxation
        
        ---
        
        ### 4. Inflammatory PCOS
        
        **Characteristics:**
        - Chronic inflammation
        - Headaches
        - Joint pain
        - Skin issues (eczema, psoriasis)
        - Fatigue
        - Digestive issues
        
        **Root Cause**: Chronic inflammation preventing ovulation and causing hormonal imbalance
        
        **Management Focus**:
        - Anti-inflammatory diet
        - Identifying and eliminating inflammatory triggers
        - Gut health improvement
        - Omega-3 fatty acids
        
        ---
        
        ### 5. Hidden PCOS
        
        **Characteristics:**
        - All symptoms present but tests appear normal
        - May have nutrient deficiencies
        - Thyroid issues
        - Iodine deficiency
        - Artificial sweetener consumption
        
        **Root Cause**: Underlying factors like nutrient deficiencies or thyroid dysfunction
        
        **Management Focus**:
        - Addressing nutrient deficiencies
        - Thyroid function optimization
        - Eliminating artificial sweeteners
        - Comprehensive health assessment
        
        ---
        
        **Important Note**: Many women may have a combination of these types. 
        A healthcare provider can help determine the specific type(s) through comprehensive testing.
        """)
    
    # Tab 3: Causes
    with tab3:
        st.subheader("Causes of PCOS")
        st.markdown("""
        The exact cause of PCOS is not fully understood, but research suggests it results from a combination of genetic, 
        environmental, and lifestyle factors.
        
        ### Primary Contributing Factors:
        
        #### 1. **Genetic Factors**
        - **Family History**: PCOS tends to run in families
        - **Genetic Predisposition**: Multiple genes may be involved
        - **Inherited Traits**: Risk increases if mother or sister has PCOS
        - **Ethnic Background**: Higher prevalence in certain populations
        
        #### 2. **Hormonal Imbalances**
        
        **Insulin Resistance:**
        - Body cells don't respond normally to insulin
        - Pancreas produces more insulin to compensate
        - Excess insulin increases androgen production
        - Leads to weight gain and difficulty losing weight
        
        **Excess Androgens:**
        - Ovaries produce higher-than-normal levels of male hormones
        - Causes physical symptoms like excess hair growth and acne
        - Interferes with normal ovulation
        
        **Luteinizing Hormone (LH) Imbalance:**
        - Abnormally high levels of LH
        - Disrupts normal ovarian function
        - Affects egg development and release
        
        #### 3. **Lifestyle Factors**
        
        **Diet:**
        - High glycemic index foods
        - Processed foods and sugars
        - Inflammatory foods
        - Poor nutrition quality
        
        **Physical Activity:**
        - Sedentary lifestyle
        - Lack of regular exercise
        - Contributes to insulin resistance
        
        **Stress:**
        - Chronic stress affects hormone production
        - Increases cortisol levels
        - Can trigger or worsen PCOS symptoms
        
        #### 4. **Environmental Factors**
        
        **Endocrine Disruptors:**
        - Chemicals in plastics (BPA)
        - Pesticides
        - Certain cosmetics
        - Can interfere with hormone function
        
        **Toxins:**
        - Environmental pollutants
        - Heavy metals
        - May contribute to hormonal imbalances
        
        #### 5. **Inflammation**
        
        - Chronic low-grade inflammation
        - Linked to insulin resistance
        - May be triggered by diet, stress, or environmental factors
        - Contributes to PCOS development and progression
        
        #### 6. **Gut Health**
        
        - Imbalanced gut microbiome
        - Leaky gut syndrome
        - Digestive issues
        - Can affect hormone metabolism and inflammation
        
        ### Risk Factors:
        
        - **Age**: Most commonly diagnosed in women in their 20s and 30s
        - **Weight**: Overweight or obesity increases risk
        - **Family History**: Genetic predisposition
        - **Ethnicity**: Higher rates in certain populations
        - **Lifestyle**: Sedentary lifestyle, poor diet
        
        ### Understanding the Complexity:
        
        PCOS is likely caused by an interaction of multiple factors rather than a single cause. 
        The condition manifests differently in each individual, suggesting that different combinations 
        of factors may be at play.
        
        **Key Takeaway**: While you can't change genetic factors, lifestyle modifications can significantly 
        impact PCOS symptoms and progression.
        """)
    
    # Tab 4: Symptoms
    with tab4:
        st.subheader("Symptoms of PCOS")
        st.markdown("""
        PCOS symptoms vary widely among individuals. Some women may experience severe symptoms, 
        while others have mild or no noticeable symptoms. Symptoms often develop gradually and may worsen over time.
        
        ### Menstrual Symptoms:
        
        **Irregular Periods:**
        - Infrequent, irregular, or prolonged menstrual cycles
        - Periods may occur every 35+ days or fewer than 8 times per year
        - Some women may have no periods at all (amenorrhea)
        - Heavy bleeding when periods do occur
        
        **Ovulation Issues:**
        - Irregular or absent ovulation
        - Difficulty predicting fertile days
        - Challenges with conception
        
        ---
        
        ### Physical Symptoms:
        
        **Excess Hair Growth (Hirsutism):**
        - Dark, coarse hair on face, chest, back, abdomen, or thighs
        - Male-pattern hair growth
        - Affects 70% of women with PCOS
        
        **Hair Loss (Androgenic Alopecia):**
        - Thinning hair on the scalp
        - Male-pattern baldness
        - Hair becomes finer and weaker
        
        **Acne:**
        - Persistent acne, especially on face, chest, and upper back
        - Often resistant to typical treatments
        - May continue beyond teenage years
        
        **Skin Darkening (Acanthosis Nigricans):**
        - Dark, velvety patches of skin
        - Common in body folds (neck, groin, under breasts)
        - Sign of insulin resistance
        
        **Skin Tags:**
        - Small flaps of excess skin
        - Often in armpits or neck area
        
        ---
        
        ### Weight-Related Symptoms:
        
        **Weight Gain:**
        - Difficulty losing weight
        - Weight gain, especially around the abdomen
        - Increased BMI
        - Affects 40-80% of women with PCOS
        
        **Difficulty Losing Weight:**
        - Slower metabolism
        - Insulin resistance makes weight loss challenging
        - May require specialized approaches
        
        ---
        
        ### Metabolic Symptoms:
        
        **Insulin Resistance:**
        - High blood sugar levels
        - Increased risk of Type 2 diabetes
        - Sugar cravings
        - Fatigue after meals
        
        **High Blood Pressure:**
        - Elevated blood pressure
        - Increased cardiovascular risk
        
        **High Cholesterol:**
        - Abnormal lipid levels
        - Increased risk of heart disease
        
        ---
        
        ### Reproductive Symptoms:
        
        **Infertility:**
        - Difficulty getting pregnant
        - Due to irregular or absent ovulation
        - Most common cause of female infertility
        
        **Miscarriage Risk:**
        - Higher risk of early pregnancy loss
        - Related to hormonal imbalances
        
        ---
        
        ### Emotional and Mental Health Symptoms:
        
        **Mood Changes:**
        - Depression
        - Anxiety
        - Mood swings
        - Irritability
        
        **Self-Esteem Issues:**
        - Body image concerns
        - Impact of physical symptoms
        - Social anxiety
        
        **Sleep Problems:**
        - Sleep apnea
        - Insomnia
        - Poor sleep quality
        
        ---
        
        ### Other Symptoms:
        
        **Fatigue:**
        - Low energy levels
        - Difficulty concentrating
        - Brain fog
        
        **Pelvic Pain:**
        - Some women experience pelvic pain
        - May be related to ovarian cysts
        
        **Headaches:**
        - Hormonal headaches
        - Migraines
        
        ---
        
        ### When to See a Doctor:
        
        Consult a healthcare provider if you experience:
        - Irregular or absent periods
        - Signs of excess androgens (hair growth, acne, hair loss)
        - Difficulty getting pregnant
        - Symptoms of diabetes (increased thirst, frequent urination)
        - Unexplained weight gain
        
        **Important**: Early diagnosis and treatment can help manage symptoms and prevent long-term complications.
        """)
    
    # Tab 5: Myths vs Facts
    with tab5:
        st.subheader("Myths vs Facts about PCOS")
        st.markdown("""
        There are many misconceptions about PCOS. Let's separate fact from fiction to better understand this condition.
        
        ### Myth vs Fact Comparison:
        
        ---
        
        #### ❌ **MYTH 1**: PCOS only affects overweight women
        **✅ FACT**: PCOS affects women of all body types. While weight gain is common, 
        thin women can also have PCOS. Approximately 20-30% of women with PCOS are at a normal weight.
        
        ---
        
        #### ❌ **MYTH 2**: You can't get pregnant if you have PCOS
        **✅ FACT**: While PCOS is a leading cause of infertility, many women with PCOS can and do get pregnant. 
        With proper treatment, lifestyle changes, and sometimes fertility medications, pregnancy is possible. 
        Some women may need assisted reproductive technologies.
        
        ---
        
        #### ❌ **MYTH 3**: PCOS is just a reproductive issue
        **✅ FACT**: PCOS is a complex metabolic and endocrine disorder affecting multiple body systems. 
        It increases the risk of Type 2 diabetes, heart disease, sleep apnea, and mental health issues. 
        It's not just about periods and fertility.
        
        ---
        
        #### ❌ **MYTH 4**: All women with PCOS have ovarian cysts
        **✅ FACT**: Despite the name, not all women with PCOS have visible cysts on their ovaries. 
        The "polycystic" refers to many small follicles (immature eggs), not necessarily cysts. 
        Some women with PCOS have normal-looking ovaries on ultrasound.
        
        ---
        
        #### ❌ **MYTH 5**: PCOS is rare
        **✅ FACT**: PCOS is very common, affecting 5-10% of women of reproductive age worldwide. 
        It's one of the most common hormonal disorders in women. Many cases go undiagnosed.
        
        ---
        
        #### ❌ **MYTH 6**: Birth control pills cure PCOS
        **✅ FACT**: Birth control pills manage symptoms but don't cure PCOS. They regulate periods, 
        reduce androgen levels, and help with acne and hair growth, but symptoms typically return 
        when you stop taking them. PCOS is a lifelong condition that requires ongoing management.
        
        ---
        
        #### ❌ **MYTH 7**: You caused PCOS by your lifestyle choices
        **✅ FACT**: While lifestyle factors can influence PCOS symptoms, the condition has strong genetic 
        and hormonal components. You didn't cause PCOS. However, healthy lifestyle choices can significantly 
        improve symptoms and quality of life.
        
        ---
        
        #### ❌ **MYTH 8**: Losing weight will completely cure PCOS
        **✅ FACT**: Weight loss can significantly improve PCOS symptoms, especially in overweight women, 
        but it doesn't "cure" the condition. Even with weight loss, PCOS requires ongoing management. 
        However, losing just 5-10% of body weight can improve symptoms dramatically.
        
        ---
        
        #### ❌ **MYTH 9**: PCOS only affects older women
        **✅ FACT**: PCOS can develop at any age after puberty, but symptoms often become noticeable in 
        the late teens and early twenties. Early diagnosis and treatment are important for long-term health.
        
        ---
        
        #### ❌ **MYTH 10**: All PCOS symptoms are the same for everyone
        **✅ FACT**: PCOS is highly variable. Symptoms differ greatly among individuals. Some women have 
        severe symptoms, while others have mild or no noticeable symptoms. This variability makes diagnosis challenging.
        
        ---
        
        #### ❌ **MYTH 11**: You can't exercise if you have PCOS
        **✅ FACT**: Exercise is actually one of the best treatments for PCOS! Regular physical activity 
        improves insulin sensitivity, helps with weight management, reduces stress, and can regulate periods. 
        Both cardio and strength training are beneficial.
        
        ---
        
        #### ❌ **MYTH 12**: PCOS means you have too many hormones
        **✅ FACT**: It's not about having "too many" hormones overall, but rather an imbalance. 
        Women with PCOS typically have higher androgens (male hormones) and insulin, but may have 
        normal or low levels of other hormones like progesterone.
        
        ---
        
        #### ❌ **MYTH 13**: Natural remedies can completely cure PCOS
        **✅ FACT**: While natural approaches (diet, exercise, supplements) can significantly help manage 
        PCOS symptoms, there's no known cure. A combination of lifestyle changes and medical treatment 
        (when needed) provides the best outcomes.
        
        ---
        
        #### ❌ **MYTH 14**: PCOS symptoms will go away after menopause
        **✅ FACT**: While some symptoms may improve after menopause (like irregular periods), 
        the metabolic aspects of PCOS (insulin resistance, diabetes risk) persist. 
        Women with PCOS need continued health monitoring throughout life.
        
        ---
        
        #### ❌ **MYTH 15**: You need to avoid all carbs if you have PCOS
        **✅ FACT**: You don't need to eliminate carbs completely. Focus on complex, low-glycemic carbs 
        (whole grains, vegetables, legumes) and avoid refined sugars and processed foods. 
        A balanced diet is key, not complete carb elimination.
        
        ---
        
        ### Key Takeaways:
        
        - PCOS is a complex, individualized condition
        - It's not your fault - genetic and hormonal factors play major roles
        - While there's no cure, symptoms can be effectively managed
        - Early diagnosis and treatment are crucial
        - Lifestyle changes can make a significant difference
        - Support and education are important for managing PCOS
        
        **Remember**: Always consult with healthcare professionals for accurate information and personalized treatment plans.
        """)
    
    # Tab 6: Prevention Tips
    with tab6:
        st.subheader("Prevention Tips for PCOS")
        st.markdown("""
        While you can't completely prevent PCOS (especially if you have a genetic predisposition), 
        you can take steps to reduce your risk, manage symptoms, and prevent complications. 
        Early intervention and healthy lifestyle choices are key.
        
        ### 1. Maintain a Healthy Weight
        
        **Why it matters:**
        - Excess weight increases insulin resistance
        - Weight loss of just 5-10% can improve symptoms significantly
        - Helps regulate menstrual cycles
        
        **Tips:**
        - Aim for gradual, sustainable weight loss (1-2 lbs per week)
        - Focus on body composition, not just the number on the scale
        - Set realistic goals
        - Consider working with a nutritionist or dietitian
        
        ---
        
        ### 2. Follow a PCOS-Friendly Diet
        
        **Key Principles:**
        
        **Choose Low-Glycemic Foods:**
        - Whole grains (oats, quinoa, brown rice)
        - Legumes (beans, lentils, chickpeas)
        - Non-starchy vegetables
        - Berries and low-sugar fruits
        
        **Include Anti-Inflammatory Foods:**
        - Fatty fish (salmon, mackerel)
        - Nuts and seeds
        - Olive oil
        - Leafy greens
        - Turmeric, ginger
        
        **Balance Your Macros:**
        - Include protein with every meal
        - Choose healthy fats
        - Limit refined carbs and sugars
        - Eat regular, balanced meals
        
        **Avoid or Limit:**
        - Refined sugars and sweets
        - Processed foods
        - High-glycemic foods (white bread, white rice, potatoes)
        - Sugary beverages
        - Trans fats
        
        ---
        
        ### 3. Exercise Regularly
        
        **Benefits:**
        - Improves insulin sensitivity
        - Helps with weight management
        - Reduces stress
        - Can help regulate periods
        
        **Recommended Activities:**
        
        **Cardiovascular Exercise:**
        - 150 minutes per week of moderate-intensity exercise
        - Walking, cycling, swimming, dancing
        - Start slow and build gradually
        
        **Strength Training:**
        - 2-3 times per week
        - Builds muscle, improves metabolism
        - Helps with insulin sensitivity
        
        **Yoga and Flexibility:**
        - Reduces stress
        - Improves mental health
        - Complements other exercises
        
        **Tips:**
        - Find activities you enjoy
        - Start with 10-15 minutes and build up
        - Consistency is more important than intensity
        - Mix different types of exercise
        
        ---
        
        ### 4. Manage Stress
        
        **Why it matters:**
        - Chronic stress affects hormone production
        - Increases cortisol, which can worsen PCOS
        - Affects insulin sensitivity
        
        **Stress Management Techniques:**
        - Meditation and mindfulness
        - Deep breathing exercises
        - Yoga or tai chi
        - Regular sleep schedule
        - Time management
        - Hobbies and activities you enjoy
        - Therapy or counseling if needed
        - Social support
        
        ---
        
        ### 5. Get Quality Sleep
        
        **Importance:**
        - Poor sleep affects hormones
        - Increases insulin resistance
        - Affects stress levels and mood
        
        **Sleep Tips:**
        - Aim for 7-9 hours per night
        - Maintain a regular sleep schedule
        - Create a relaxing bedtime routine
        - Keep bedroom cool, dark, and quiet
        - Limit screen time before bed
        - Avoid caffeine late in the day
        - Address sleep apnea if present
        
        ---
        
        ### 6. Regular Health Monitoring
        
        **Regular Check-ups:**
        - Annual physical exams
        - Regular blood work (glucose, insulin, lipids, hormones)
        - Blood pressure monitoring
        - Weight tracking
        
        **Screenings:**
        - Diabetes screening (especially if overweight)
        - Cholesterol levels
        - Thyroid function
        - Vitamin D levels
        
        ---
        
        ### 7. Early Detection and Treatment
        
        **Know the Signs:**
        - Irregular periods
        - Excess hair growth
        - Acne
        - Weight gain
        - Hair loss
        
        **Take Action:**
        - Don't ignore symptoms
        - See a healthcare provider early
        - Get proper diagnosis
        - Start treatment promptly
        
        ---
        
        ### 8. Support Gut Health
        
        **Why it matters:**
        - Gut health affects hormone metabolism
        - Influences inflammation
        - Affects insulin sensitivity
        
        **Tips:**
        - Eat probiotic-rich foods (yogurt, kefir, fermented foods)
        - Include prebiotic foods (garlic, onions, bananas, oats)
        - Stay hydrated
        - Limit processed foods
        - Consider probiotic supplements if needed
        
        ---
        
        ### 9. Limit Exposure to Endocrine Disruptors
        
        **Sources to Avoid:**
        - BPA in plastics (use BPA-free containers)
        - Pesticides (choose organic when possible)
        - Certain cosmetics and personal care products
        - Environmental pollutants
        
        **Tips:**
        - Use glass or stainless steel containers
        - Choose natural, organic products
        - Filter drinking water
        - Avoid heating food in plastic containers
        
        ---
        
        ### 10. Build a Support System
        
        **Importance:**
        - Emotional support is crucial
        - Reduces stress
        - Helps with motivation
        - Provides accountability
        
        **Ways to Build Support:**
        - Join PCOS support groups (online or in-person)
        - Talk to family and friends
        - Work with healthcare providers
        - Consider therapy or counseling
        - Connect with others who have PCOS
        
        ---
        
        ### 11. Stay Informed
        
        **Education:**
        - Learn about PCOS
        - Understand your symptoms
        - Know your treatment options
        - Stay updated on research
        - Ask questions to your healthcare provider
        
        ---
        
        ### 12. Be Patient and Consistent
        
        **Remember:**
        - Changes take time
        - Consistency is key
        - Small steps lead to big improvements
        - Setbacks are normal
        - Celebrate small victories
        - Focus on progress, not perfection
        
        ---
        
        ### Prevention Timeline:
        
        **Immediate Actions:**
        - Start with one healthy habit
        - Schedule a doctor's appointment
        - Begin tracking your symptoms
        
        **Short-term (1-3 months):**
        - Establish exercise routine
        - Improve diet
        - Develop stress management practices
        
        **Long-term (3+ months):**
        - Maintain healthy lifestyle
        - Regular monitoring
        - Ongoing management
        
        ---
        
        ### Important Reminders:
        
        ⚠️ **These tips can help prevent complications and manage symptoms, but PCOS is a complex condition.**
        
        ✅ **Work with healthcare professionals for personalized advice.**
        
        ✅ **What works for one person may not work for another - find what works for you.**
        
        ✅ **Be kind to yourself - managing PCOS is a journey, not a destination.**
        
        **Remember**: Prevention and management of PCOS is about creating sustainable, healthy habits 
        that support your overall well-being. Small, consistent changes can make a significant difference 
        in your symptoms and quality of life.
        """)
    
    # Add a disclaimer at the bottom
    st.markdown("---")
    st.info("""
    **Medical Disclaimer**: The information provided in this education module is for educational purposes only 
    and should not be considered as medical advice. Always consult with qualified healthcare professionals 
    for diagnosis, treatment, and personalized medical advice regarding PCOS.
    """)

# PCOS Detection Section
elif navigation == "🩺 PCOS Detection":
    st.header("PCOS Disease Prediction")
    st.write("Fill in the details below to predict if a person has PCOS.")

    # User Inputs for PCOS Prediction
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=64)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=156)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=26.3)
    blood_group = st.selectbox("Blood Group", options=["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"], index=0)
    pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=30, max_value=200, value=70)
    rr = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=50, value=18)
    hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=11.2)
    cycle = st.number_input("Cycle Regularity (days)", min_value=0, max_value=40, value=2)
    cycle_length = st.number_input("Cycle Length (days)", min_value=0, max_value=40, value=6)
    marriage_status = st.selectbox("Marriage Status", options=["Married", "Unmarried"], index=0)
    pregnant = st.selectbox("Pregnant", options=["Yes", "No"], index=1)
    no_of_abortions = st.number_input("Number of Abortions", min_value=0, max_value=10, value=0)
    hip = st.number_input("Hip Circumference (cm)", min_value=20, max_value=200, value=39)
    waist = st.number_input("Waist Circumference (cm)", min_value=20, max_value=200, value=34)
    waist_hip_ratio = st.number_input("Waist-Hip Ratio", min_value=0.5, max_value=2.0, value=0.87)
    weight_gain = st.selectbox("Weight Gain", options=["Yes", "No"], index=1)
    hair_growth = st.selectbox("Excess Hair Growth", options=["Yes", "No"], index=1)
    skin_darkening = st.selectbox("Skin Darkening", options=["Yes", "No"], index=1)
    hair_loss = st.selectbox("Hair Loss", options=["Yes", "No"], index=1)
    pimples = st.selectbox("Pimples", options=["Yes", "No"], index=1)
    fast_food = st.selectbox("Frequent Fast Food Intake", options=["Yes", "No"], index=1)
    reg_exercise = st.selectbox("Regular Exercise", options=["Yes", "No"], index=1)
    bp_systolic = st.number_input("BP Systolic (mmHg)", min_value=50, max_value=200, value=110)
    bp_diastolic = st.number_input("BP Diastolic (mmHg)", min_value=30, max_value=150, value=80)

    # Prepare input data
    input_data = [
        age, weight, height, bmi, 0 if blood_group == "A+" else 1, pulse_rate, rr, hb,
        cycle, cycle_length, 0 if marriage_status == "Married" else 1,
        1 if pregnant == "Yes" else 0, no_of_abortions, hip, waist, waist_hip_ratio,
        1 if weight_gain == "Yes" else 0, 1 if hair_growth == "Yes" else 0,
        1 if skin_darkening == "Yes" else 0, 1 if hair_loss == "Yes" else 0,
        1 if pimples == "Yes" else 0, 1 if fast_food == "Yes" else 0,
        1 if reg_exercise == "Yes" else 0, bp_systolic, bp_diastolic
    ]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Prediction Button
    if st.button("Predict"):
        prediction = pcos_model.predict(input_data_as_numpy_array)
        if prediction[0] == 0:
            st.success("The Person does not have PCOS Disease.")
        else:
            st.error("The Person has PCOS Disease.")

# Period Tracker Section
elif navigation == "📅 Period Tracker":
    st.header("Period Tracker")
    st.write("Track your periods and get predictions for the next cycle.")

    # Inputs for Period Tracking
    last_period_date = st.date_input("Last Period Date", datetime.date.today())
    average_cycle_length = st.number_input("Average Cycle Length (days)", min_value=20, max_value=40, value=28)

    # Calculate next period date
    if st.button("Track"):
        next_period_date = last_period_date + datetime.timedelta(days=average_cycle_length)
        st.success(f"Your next period is expected to start on: {next_period_date.strftime('%d %B, %Y')}")

# Diet Plan Section
elif navigation == "🥗 Diet Plan":
    st.header("PCOS Diet Plan Generator")
    st.write("Enter your weight, activity level, and dietary preferences to get a personalized PCOS-friendly diet plan.")

    weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=200, value=70)
    activity_level = st.selectbox("Activity level", ["low", "moderate", "high"])
    preference = st.selectbox("Dietary preference", ["regular", "vegetarian", "vegan"])

    if st.button("Generate Diet Plan"):
        # Diet Plan Calculation Logic
        def calculate_macronutrients(calories):
            protein_grams = (calories * 0.30) / 4  # 4 calories per gram of protein
            carb_grams = (calories * 0.40) / 4    # 4 calories per gram of carbohydrates
            fat_grams = (calories * 0.30) / 9     # 9 calories per gram of fats
            return {
                "Protein (g)": round(protein_grams, 2),
                "Carbs (g)": round(carb_grams, 2),
                "Fats (g)": round(fat_grams, 2)
            }

        def get_diet_plan(weight, activity_level, preference):
            calories = weight * 10
            if activity_level == "low":
                calories *= 1.2
            elif activity_level == "moderate":
                calories *= 1.5
            elif activity_level == "high":
                calories *= 1.8

            macros = calculate_macronutrients(calories)

            diet_plan = {
                "Breakfast": "Oats with almond milk and chia seeds",
                "Snack": "Handful of nuts and a small apple",
                "Lunch": "Grilled chicken/fish with quinoa and mixed vegetables",
                "Snack": "Greek yogurt with berries",
                "Dinner": "Lentil soup with a side of roasted veggies"
            }

            return macros, diet_plan

        macros, plan = get_diet_plan(weight, activity_level, preference)
        st.write(f"Your estimated daily calorie intake: {weight * 10} kcal")
        st.write("Macronutrient Breakdown:")
        st.write(macros)
        st.write("Recommended Diet Plan:")
        for meal, recipe in plan.items():
            st.write(f"{meal}: {recipe}")

# Image-based PCOS Detection Section
elif navigation == "🖼️ Image-based PCOS Detection":
    st.header("PCOS Detection via Image (Ultrasound)")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image and make prediction
        img = image.load_img(uploaded_image, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)
        if np.argmax(prediction[0]) == 0:
            st.success("PCOS Detected!")
        else:
            st.error("No PCOS Detected!")