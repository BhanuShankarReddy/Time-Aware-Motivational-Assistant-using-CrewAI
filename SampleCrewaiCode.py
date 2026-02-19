from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import pytz

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#Generate with beam search
def generate_response(prompt: str, max_tokens: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
#time detection code
india_tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(india_tz)
current_hour = current_time.hour

if 5 <= current_hour < 12:
    time_of_day = "morning"
    motivation_prompt = "Write a positive message to start the morning with motivation and confidence."
    reminder_prompt = "Remind the user to drink warm water and do a few light exercises."
elif 12 <= current_hour < 17:
    time_of_day = "afternoon"
    motivation_prompt = "Write a short, inspiring message to help someone stay focused this afternoon."
    reminder_prompt = "Remind the user to eat lunch and stay hydrated."
else:
    time_of_day = "evening"
    motivation_prompt = "Give a relaxing and calming message to help someone unwind in the evening."
    reminder_prompt = "Remind the user to reduce screen time and prepare for bed."

# Generate
motivation_output = generate_response(motivation_prompt)
reminder_output = generate_response(reminder_prompt)

print("Current Time:", current_time.strftime("%Y-%m-%d %I:%M %p"))
print("Detected Time of Day:", time_of_day.capitalize())
print("\nMotivational Message:\n", motivation_output)
print("\nReminder Message:\n", reminder_output)
