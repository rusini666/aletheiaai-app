import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# API Endpoint
BASE_URL = "http://127.0.0.1:5000"
API_ENDPOINT = f"{BASE_URL}/api/lime_explanation"

# Generate Texts of Increasing Length
def generate_text(length):
    return "AI " * length  # Simple repetitive pattern

# Function to Measure Response Time with Retry Logic
def measure_response_time(text, retries=3):
    for attempt in range(retries):
        start_time = time.time()
        try:
            # Increased timeout to 10 minutes (600 seconds)
            response = requests.post(API_ENDPOINT, json={"text": text}, timeout=600)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            return elapsed_time
        except requests.exceptions.Timeout:
            print(f"⏳ Timeout (Attempt {attempt + 1}) for text length {len(text.split())}")
            time.sleep(2)  # Short wait before retry
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection Error (Attempt {attempt + 1}) for text length {len(text.split())}")
            time.sleep(2)
    return None  # Return None if all attempts fail

# Scalability Test with Concurrent Users
def test_scalability(text_lengths, concurrent_users_list):
    results = {}

    for length in text_lengths:
        text = generate_text(length)
        print(f"\n📝 Testing with Text Length: {length} words")

        for users in concurrent_users_list:
            print(f"👥 Simulating {users} concurrent users...")
            
            with ThreadPoolExecutor(max_workers=users) as executor:
                response_times = list(executor.map(measure_response_time, [text] * users))

            # Filter out None values (failed requests)
            response_times = [time for time in response_times if time is not None]
            results[(length, users)] = response_times

            if response_times:
                avg_time = np.mean(response_times)
                print(f"✅ Avg Response Time for {users} users: {avg_time:.2f} ms")
            else:
                print(f"⚠️ All requests failed for Text Length {length} with {users} users.")

    return results

# Test Parameters
text_lengths = [10, 50, 100, 500, 1000, 2000, 5000]     # Including 5000 words
concurrent_users_list = [5, 10, 20, 50]                 # Simulating up to 50 users

# Run Scalability Test
scalability_results = test_scalability(text_lengths, concurrent_users_list)

# 📊 Plotting Scalability Results
plt.figure(figsize=(12, 8))

# Plotting for each combination of text length and concurrent users
for (length, users), times in scalability_results.items():
    if times:  # Ensure there's data to plot
        plt.plot(range(1, len(times) + 1), times, marker='o', label=f'{length} words, {users} Users')

plt.xlabel('Request Number')
plt.ylabel('Response Time (ms)')
plt.title('Scalability Testing: API Response Time vs Text Length & Concurrent Users')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("enhanced_scalability_test_with_5000_words.png")
plt.show()
