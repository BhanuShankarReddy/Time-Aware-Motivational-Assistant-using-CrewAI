from datetime import datetime
import pytz

local_tz = pytz.timezone('Asia/Kolkata')
current_hour = datetime.now(local_tz).hour

if 5 <= current_hour < 12:
    time_of_day = "Morning"
elif 12 <= current_hour < 18:
    time_of_day = "Afternoon"
else:
    time_of_day = "Night"

print(f"Detected Time: {time_of_day}")
