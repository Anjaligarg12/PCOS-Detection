import pickle

def calculate_macronutrients(calories):
    """
    Calculate macronutrients based on standard PCOS-friendly ratios:
    - Protein: 30% of calories
    - Carbs: 40% of calories
    - Fats: 30% of calories
    """
    protein_grams = (calories * 0.30) / 4  # 4 calories per gram of protein
    carb_grams = (calories * 0.40) / 4    # 4 calories per gram of carbohydrates
    fat_grams = (calories * 0.30) / 9     # 9 calories per gram of fats
    return {
        "Protein (g)": round(protein_grams, 2),
        "Carbs (g)": round(carb_grams, 2),
        "Fats (g)": round(fat_grams, 2)
    }

def get_diet_plan(weight, activity_level, preference):
    # Base calorie calculation
    calories = weight * 10  # Basal Metabolic Rate approximation

    # Adjust calories based on activity level
    if activity_level == "low":
        calories *= 1.2
    elif activity_level == "moderate":
        calories *= 1.5
    elif activity_level == "high":
        calories *= 1.8

    # Macronutrient breakdown
    macros = calculate_macronutrients(calories)

    # Example diet plan based on calories and preferences
    diet_plan = {
        "Breakfast": "Oats with almond milk and chia seeds",
        "Snack": "Handful of nuts and a small apple",
        "Lunch": "Grilled chicken/fish with quinoa and mixed vegetables",
        "Snack": "Greek yogurt with berries",
        "Dinner": "Lentil soup with a side of spinach salad"
    }

    if preference == "vegetarian":
        diet_plan["Lunch"] = "Tofu stir-fry with quinoa and vegetables"
        diet_plan["Dinner"] = "Vegetable curry with brown rice"
    elif preference == "vegan":
        diet_plan["Breakfast"] = "Smoothie with spinach, almond milk, and banana"
        diet_plan["Snack"] = "Roasted chickpeas"
        diet_plan["Lunch"] = "Lentil stew with sweet potatoes"
        diet_plan["Dinner"] = "Quinoa bowl with roasted vegetables and tahini dressing"

    return calories, macros, diet_plan

# Input from the user
weight = float(input("Enter your weight (in kg): "))
activity_level = input("Enter your activity level (low, moderate, high): ").strip().lower()
preference = input("Enter your dietary preference (regular, vegetarian, vegan): ").strip().lower()

# Generate diet plan
calories, macros, diet_plan = get_diet_plan(weight, activity_level, preference)

# Display the diet plan
print("\nHere is your suggested diet plan:")
for meal, item in diet_plan.items():
    print(f"{meal}: {item}")

# Display macronutrient breakdown
print("\nMacronutrient Breakdown:")
for macro, value in macros.items():
    print(f"{macro}: {value}")

# Export the model to a .pkl file
model_data = {
    "weight": weight,
    "activity_level": activity_level,
    "preference": preference,
    "calories": calories,
    "macros": macros,
    "diet_plan": diet_plan
}

with open("pcos_diet_plan.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\nDiet plan model has been saved to 'pcos_diet_plan.pkl'.")