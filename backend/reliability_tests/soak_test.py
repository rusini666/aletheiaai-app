import requests
import time
import matplotlib.pyplot as plt

# API Endpoint
BASE_URL = "http://127.0.0.1:5000"
API_ENDPOINT = f"{BASE_URL}/api/lime_explanation"

# Test Text
TEST_TEXT = "This is a reliability test to evaluate system stability over time."

# Function to Send Request with Retry Logic
def send_request(retries=2):
    for attempt in range(retries + 1):
        try:
            start_time = time.time()
            # Increased timeout to 10 minutes (600 seconds)
            response = requests.post(API_ENDPOINT, json={"text": TEST_TEXT}, timeout=600)
            response.raise_for_status()
            return (time.time() - start_time) * 1000  # Convert to milliseconds
        except requests.exceptions.Timeout:
            print(f"❌ Request timed out after 10 minutes on attempt {attempt + 1}.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed with error: {e}")
        
        # Retry logic
        if attempt < retries:
            print(f"🔄 Retrying request (Attempt {attempt + 2}/{retries + 1})...")

    return None  # Mark as failed after max retries

# Reliability Test
def reliability_test(total_requests=100, batch_size=10, delay_between_batches=10):
    response_times = []
    failures = 0

    print(f"🔍 Starting Reliability Test: {total_requests} Requests in Batches of {batch_size}")

    for batch in range(0, total_requests, batch_size):
        batch_times = []

        for _ in range(batch_size):
            response_time = send_request()
            if response_time is not None:
                batch_times.append(response_time)
            else:
                failures += 1

        response_times.extend(batch_times)
        print(f"✅ Batch {batch // batch_size + 1}: {len(batch_times)} Success, {batch_size - len(batch_times)} Failures")

        # Add delay between batches
        time.sleep(delay_between_batches)

    # Plotting Results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(response_times) + 1), response_times, marker='o')
    plt.xlabel("Request Number")
    plt.ylabel("Response Time (ms)")
    plt.title(f"Reliability Test Results: {total_requests} Requests")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("final_reliability_test_results.png")
    plt.show()

    print(f"⚠️ Total Failures: {failures}/{total_requests}")
    if response_times:
        print(f"📊 Average Response Time: {sum(response_times) / len(response_times):.2f} ms")

# Run the Final Reliability Test
reliability_test(total_requests=500, batch_size=10, delay_between_batches=10)
