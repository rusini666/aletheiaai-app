import requests
import time
import matplotlib.pyplot as plt
import numpy as np

# API Endpoint for SHAP + LIME Explanation
BASE_URL = "http://127.0.0.1:5000"
API_ENDPOINT = f"{BASE_URL}/api/lime_explanation"  # Update if needed

# Test Text for API Calls
TEST_TEXT = "Artificial intelligence is transforming industries by automating tasks and providing data-driven insights."

# Function to Measure Response Time
def measure_response_time(num_requests):
    response_times = []
    
    for i in range(num_requests):
        start_time = time.time()

        # Send POST request to the API
        response = requests.post(API_ENDPOINT, json={"text": TEST_TEXT})

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        response_times.append(elapsed_time)

        # Debug log
        print(f"Request {i+1}/{num_requests}: {elapsed_time:.2f} ms")

    return response_times

# Run Performance Test for Different Load Levels
load_levels = [10, 20, 50, 100]  # Number of requests
all_response_times = {}

for load in load_levels:
    print(f"\n📊 Running performance test for {load} requests...")
    response_times = measure_response_time(load)
    all_response_times[load] = response_times

# 📈 Plotting Response Times
plt.figure(figsize=(10, 6))

for load, times in all_response_times.items():
    plt.plot(range(1, len(times) + 1), times, label=f'{load} Requests')

plt.xlabel('Request Number')
plt.ylabel('Response Time (ms)')
plt.title('LIME API Performance Testing')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the performance graph
plt.savefig("lime_performance_test.png")
plt.show()