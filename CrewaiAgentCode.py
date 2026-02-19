from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List
from pydantic import PrivateAttr
from crewai import Agent, Task, Crew
from datetime import datetime
import pytz


class LocalFlanT5(LLM):
    model_name: str = "google/flan-t5-small"
    max_new_tokens: int = 50

    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: AutoModelForSeq2SeqLM = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not prompt or not prompt.strip():
            return "Prompt is empty."
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"LLM Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "local_flan_t5"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

#Time detection
india_tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(india_tz)
current_hour = current_time.hour

if 5 <= current_hour < 12:
    time_of_day = "morning"
    motivation = "Generate a motivational message for the morning."
    reminder = "Remind user to drink warm water and exercise."
elif 12 <= current_hour < 17:
    time_of_day = "afternoon"
    motivation = "Give an afternoon motivation message."
    reminder = "Remind user to eat lunch and stay hydrated."
else:
    time_of_day = "evening"
    motivation = "Give a relaxing message to end the day."
    reminder = "Remind user to avoid screens and prepare for bed."

#for implementing LLM instance
llm = LocalFlanT5()

#Agents
motivational_agent = Agent(
    role="Motivational Coach",
    goal="Provide motivational message",
    backstory="Helps users stay positive",
    verbose=True,
    llm=llm
)

reminder_agent = Agent(
    role="Reminder Assistant",
    goal="Give helpful daily habit reminders",
    backstory="Focuses on healthy habits",
    verbose=True,
    llm=llm
)

#Tasks to assign
motivational_task = Task(
    description=motivation,
    expected_output="Motivational text",
    agent=motivational_agent
)

reminder_task = Task(
    description=reminder,
    expected_output="Habit reminder",
    agent=reminder_agent
)

#Crew
crew = Crew(
    agents=[motivational_agent, reminder_agent],
    tasks=[motivational_task, reminder_task],
    verbose=True
)

print("Current Time:", current_time.strftime("%Y-%m-%d %I:%M %p"))
print("Detected Time of Day:", time_of_day.capitalize())

try:
    result = crew.kickoff()
    print("\nFinal Output:")
    if isinstance(result, dict):
        for task, output in result.items():
            print(f"\nTask: {task}\nOutput: {output}")
    else:
        print(result)
except Exception as e:
    print("\nError during Crew kickoff:", str(e))
