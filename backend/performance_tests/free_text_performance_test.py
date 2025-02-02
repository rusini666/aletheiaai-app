import requests
import time
import matplotlib.pyplot as plt

# API Endpoint
BASE_URL = "http://127.0.0.1:5000/api/classify"

# Function to simulate API requests
def measure_response_times(request_count):
    response_times = []

    for _ in range(request_count):
        start_time = time.time()
        try:
            response = requests.post(BASE_URL, json={"text": "This is a sample text for performance testing."})
            response_time = time.time() - start_time
            response_times.append(response_time)
        except requests.exceptions.RequestException:
            response_times.append(None)  # Log None if the request fails

    return response_times

# Define varying load levels
request_counts = [10, 50, 100, 200, 500]
results = {}

# Perform the tests
for count in request_counts:
    print(f"Sending {count} requests...")
    response_times = measure_response_times(count)
    valid_times = [t for t in response_times if t is not None]  # Filter out failed requests
    avg_response_time = sum(valid_times) / len(valid_times) if valid_times else 0
    max_response_time = max(valid_times) if valid_times else 0

    results[count] = {
        "avg_time": avg_response_time,
        "max_time": max_response_time
    }

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), [v["avg_time"] for v in results.values()], label='Average Response Time', marker='o')
plt.plot(list(results.keys()), [v["max_time"] for v in results.values()], label='Maximum Response Time', marker='s')

plt.xlabel("Number of Requests")
plt.ylabel("Response Time (seconds)")
plt.title("API Performance Testing - Response Times Under Varying Loads")
plt.legend()
plt.grid(True)
plt.show()
