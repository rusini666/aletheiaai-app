import requests
import time
import matplotlib.pyplot as plt
import numpy as np

# API Endpoint
BASE_URL = "http://127.0.0.1:5000"
API_ENDPOINT = f"{BASE_URL}/api/lime_explanation"

# Generate Texts of Increasing Length
def generate_text(length):
    return "AI " * length  # Simple repetitive pattern

# Test Function
def test_scalability(text_lengths, concurrent_users):
    response_times = {}

    for length in text_lengths:
        text = generate_text(length)
        times = []

        for _ in range(concurrent_users):
            start_time = time.time()
            response = requests.post(API_ENDPOINT, json={"text": text})
            elapsed_time = (time.time() - start_time) * 1000  # ms

            times.append(elapsed_time)
            print(f"Text Length: {length}, Response Time: {elapsed_time:.2f} ms")

        response_times[length] = times

    return response_times

# Test Parameters
text_lengths = [10, 50, 100, 200, 500]      # Increasing text length
concurrent_users = 5                        # Simulate 5 users per length

# Run Scalability Test
scalability_results = test_scalability(text_lengths, concurrent_users)

# 📊 Plotting Scalability Results
plt.figure(figsize=(10, 6))

for length, times in scalability_results.items():
    plt.plot(range(1, len(times) + 1), times, marker='o', label=f'Text Length {length}')

plt.xlabel('Request Number')
plt.ylabel('Response Time (ms)')
plt.title('Scalability Testing: API Response vs Text Length')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("scalability_test_results.png")
plt.show()
