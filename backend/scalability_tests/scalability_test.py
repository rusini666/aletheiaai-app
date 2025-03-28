import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# API Endpoint (using the LIME explanation endpoint for this test)
BASE_URL = "http://127.0.0.1:5000"
API_ENDPOINT = f"{BASE_URL}/api/lime_explanation"

def generate_test_text(length=500):
    """Generate a random text string of the specified length."""
    import random, string
    return "".join(random.choices(string.ascii_lowercase + " ", k=length))

def send_request(test_text, retries=1):
    """Send a single POST request to the API endpoint with retry logic."""
    for attempt in range(retries + 1):
        try:
            start_time = time.time()
            # Set timeout to 600 seconds, but for a simple test you may lower this if needed.
            response = requests.post(API_ENDPOINT, json={"text": test_text}, timeout=600)
            response.raise_for_status()
            elapsed_ms = (time.time() - start_time) * 1000  # Convert seconds to ms
            return elapsed_ms
        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(1)
    return None

def simple_scalability_test(text_length=500, concurrent_users=2, total_requests=10):
    """Perform a simple scalability test with low concurrency and few total requests."""
    test_text = generate_test_text(text_length)
    response_times = []
    
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(send_request, test_text) for _ in range(total_requests)]
        for future in futures:
            res = future.result()
            if res is not None:
                response_times.append(res)
    
    return response_times

def run_tests_and_plot():
    # Use a couple of low concurrency levels for a gentle test
    concurrency_levels = [1, 2, 3]
    overall_results = {}
    
    for users in concurrency_levels:
        print(f"\nTesting with {users} concurrent user(s)...")
        times = simple_scalability_test(text_length=500, concurrent_users=users, total_requests=10)
        overall_results[users] = times
        if times:
            avg_time = np.mean(times)
            print(f"  Avg Response Time: {avg_time:.2f} ms over {len(times)} successful requests.")
        else:
            print("  All requests failed.")
    
    # Plot average response times versus concurrency level
    avg_times = []
    for users in concurrency_levels:
        if overall_results[users]:
            avg_times.append(np.mean(overall_results[users]))
        else:
            avg_times.append(0)
    
    plt.figure(figsize=(8, 5))
    plt.plot(concurrency_levels, avg_times, marker='o')
    plt.xlabel("Number of Concurrent Users")
    plt.ylabel("Average Response Time (ms)")
    plt.title("Scalability Test")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simple_scalability_results.png")
    plt.show()

if __name__ == "__main__":
    run_tests_and_plot()