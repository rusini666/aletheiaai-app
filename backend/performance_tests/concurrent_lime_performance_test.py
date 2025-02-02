import requests
import time
import threading
import matplotlib.pyplot as plt

# Configuration
BASE_URL = "http://127.0.0.1:5000/api/lime_explanation"  # Ensure Flask is running
NUM_REQUESTS = 50       # Total number of API requests
CONCURRENCY = 10        # Number of concurrent threads
TEST_TEXT = "Artificial intelligence is revolutionizing the world in various industries."

# Function to send a single request
def send_request(response_times):
    start_time = time.time()
    try:
        response = requests.post(BASE_URL, json={"text": TEST_TEXT})
        response.raise_for_status()  # Raise an error for bad responses
    except requests.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return
    end_time = time.time()
    response_times.append(end_time - start_time)

# Performance Testing Function
def run_performance_test():
    response_times = []
    threads = []

    print(f"Starting performance test with {NUM_REQUESTS} requests and {CONCURRENCY} concurrent threads...")

    # Run the requests with concurrency
    for i in range(NUM_REQUESTS):
        thread = threading.Thread(target=send_request, args=(response_times,))
        threads.append(thread)
        thread.start()

        # Limit the number of concurrent threads
        if len(threads) >= CONCURRENCY:
            for t in threads:
                t.join()
            threads = []  # Reset thread pool

    # Wait for remaining threads to finish
    for t in threads:
        t.join()

    print(f"✅ Completed {NUM_REQUESTS} requests.")
    return response_times

# Plotting the results
def plot_response_times(response_times):
    plt.figure(figsize=(10, 6))
    plt.plot(response_times, marker='o', linestyle='-', color='b')
    plt.xlabel("Request Number")
    plt.ylabel("Response Time (seconds)")
    plt.title("API Response Times for LIME Explanation")
    plt.grid(True)
    plt.show()

# Main Execution
if __name__ == "__main__":
    response_times = run_performance_test()

    if response_times:
        print(f"\n📊 Performance Summary:")
        print(f"Average Response Time: {sum(response_times) / len(response_times):.2f} seconds")
        print(f"Fastest Response Time: {min(response_times):.2f} seconds")
        print(f"Slowest Response Time: {max(response_times):.2f} seconds")

        plot_response_times(response_times)
    else:
        print("⚠️ No successful responses recorded.")
