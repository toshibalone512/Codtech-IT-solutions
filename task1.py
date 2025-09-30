import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


API_KEY = "2a844f0ea59dde37f63508e1a1f04e2c" 
CITY = "Nagpur"            
DAYS = 5                   

url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

# FETCH DATA
response = requests.get(url)
data = response.json()

if response.status_code != 200:
    print("Error fetching data:", data)
    exit()

# PROCESS DATA
forecast_list = data['list']

# Extract useful information
weather_data = []
for entry in forecast_list:
    dt = datetime.fromtimestamp(entry['dt'])
    temp = entry['main']['temp']
    feels_like = entry['main']['feels_like']
    humidity = entry['main']['humidity']
    weather = entry['weather'][0]['description']
    
    weather_data.append([dt, temp, feels_like, humidity, weather])


df = pd.DataFrame(weather_data, columns=["datetime", "temp", "feels_like", "humidity", "weather"])
print(df.head())

# VISUALIZATIONS

# 1. Temperature over time
plt.figure(figsize=(12,6))
plt.plot(df["datetime"], df["temp"], marker="o", label="Temperature (°C)")
plt.plot(df["datetime"], df["feels_like"], marker="x", linestyle="--", label="Feels Like (°C)")
plt.title(f"Temperature Forecast for {CITY}")
plt.xlabel("Date & Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Humidity over time
plt.figure(figsize=(12,6))
sns.lineplot(x="datetime", y="humidity", data=df, marker="o", color="blue")
plt.title(f"Humidity Forecast for {CITY}")
plt.xlabel("Date & Time")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Weather condition counts
plt.figure(figsize=(8,6))
sns.countplot(y="weather", data=df, palette="viridis")
plt.title(f"Weather Condition Frequency in {CITY}")
plt.xlabel("Count")
plt.ylabel("Weather Condition")
plt.tight_layout()
plt.show()
