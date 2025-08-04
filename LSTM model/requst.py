import requests
import json

# Test the exact invoke URL from your stage details
base_url = "https://ufmjvh8aj6.execute-api.ap-south-1.amazonaws.com/prod"

print("Quick API Gateway Diagnostic Test")
print("=" * 50)

# Test 1: Check if API Gateway is responding at all
print("1. Testing base invoke URL...")
try:
    response = requests.get(base_url, timeout=10)
    print(f"   URL: {base_url}")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text}")
    print(f"   Headers: {dict(response.headers)}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "-" * 30)

# Test 2: Try the route we expect to work
expected_url = f"{base_url}/waterquality-lambda"
print("2. Testing expected route...")

test_data = {"input_sequence": [[7.5, 0.03, 5.0], [7.6, 0.031, 5.1]]}

try:
    response = requests.post(expected_url, json=test_data, timeout=30)
    print(f"   URL: {expected_url}")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text}")

    # Check response headers for clues
    if "x-amzn-requestid" in response.headers:
        print(f"   Request ID: {response.headers['x-amzn-requestid']}")
    if "x-amz-apigw-id" in response.headers:
        print(f"   API Gateway ID: {response.headers['x-amz-apigw-id']}")

except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50)
print("Next steps:")
print("1. Check the 'Resources' tab in your API Gateway prod stage")
print("2. Look for any error patterns in the response headers")
print("3. Check if the deployment actually included your routes")
