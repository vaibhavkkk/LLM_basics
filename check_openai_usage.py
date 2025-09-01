import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
try:
    from openai import OpenAI
except ImportError:
    print("⚠️  OpenAI library not found. Install with: pip install openai")
    OpenAI = None

def load_api_key():
    """Load OpenAI API key from environment variables"""
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("❌ No API key found. Please set OPENAI_API_KEY in your .env file.")
    elif not api_key.startswith('sk-'):
        raise ValueError("❌ Invalid API key format. OpenAI keys should start with 'sk-'")
    elif api_key.strip() != api_key:
        raise ValueError("❌ API key contains leading or trailing spaces. Please remove them.")
    
    print("✅ API key loaded successfully!")
    return api_key.strip()

def check_api_usage(api_key, date=None):
    """
    Check OpenAI API usage for a specific date
    """
    if date is None:
        date = datetime.now().date() - timedelta(days=1)  # Yesterday's data

    # OpenAI API endpoint for usage - updated to correct endpoint
    url = f"https://api.openai.com/v1/usage?date={date.strftime('%Y-%m-%d')}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        print(f"🔍 Checking usage for {date}...")
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            usage_data = response.json()
            display_usage_summary(usage_data, date)
            return usage_data
        elif response.status_code == 401:
            print("❌ Authentication failed. Please check your API key.")
        elif response.status_code == 403:
            print("❌ Access forbidden. Your API key may not have the required permissions.")
        elif response.status_code == 429:
            print("❌ Rate limit exceeded. Please try again later.")
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def check_billing_usage(api_key, days_back=30):
    """
    Alternative method to check billing usage using the billing endpoint
    """
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    # Try the billing usage endpoint
    url = "https://api.openai.com/v1/dashboard/billing/usage"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    params = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }

    try:
        print(f"🔍 Checking billing usage from {start_date} to {end_date}...")
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            billing_data = response.json()
            display_billing_summary(billing_data, start_date, end_date)
            return billing_data
        elif response.status_code == 401:
            print("❌ Authentication failed. Please check your API key.")
        elif response.status_code == 403:
            print("❌ Access forbidden. Your API key may not have billing access permissions.")
        elif response.status_code == 429:
            print("❌ Rate limit exceeded. Please try again later.")
        else:
            print(f"❌ Billing API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def display_usage_summary(usage_data, date):
    """Display a formatted summary of API usage for a specific date"""
    print("\n" + "="*60)
    print("📊 OPENAI API USAGE SUMMARY")
    print("="*60)
    print(f"Date: {date}")

    if usage_data and isinstance(usage_data, dict):
        print(f"\n� Usage Data Structure:")
        print(json.dumps(usage_data, indent=2))

        # Try to extract meaningful data from the response
        if 'object' in usage_data:
            print(f"\n📋 Object Type: {usage_data['object']}")

        # Look for common usage fields
        total_usage = usage_data.get('total_usage', 0)
        if total_usage:
            print(f"💰 Total Usage: ${total_usage / 100:.4f}")

    else:
        print("\n📭 No usage data found for the specified date.")
        print("This could mean:")
        print("   • No API calls were made on this date")
        print("   • The API key is new and hasn't been used yet")
        print("   • There might be a delay in usage reporting")

def display_billing_summary(billing_data, start_date, end_date):
    """Display a formatted summary of billing usage"""
    print("\n" + "="*60)
    print("💳 BILLING USAGE SUMMARY")
    print("="*60)
    print(f"Period: {start_date} to {end_date}")

    if billing_data and isinstance(billing_data, dict):
        print(f"\n� Billing Data:")
        print(json.dumps(billing_data, indent=2))

        # Extract total usage
        total_usage = billing_data.get('total_usage', 0)
        if total_usage:
            print(f"\n💰 Total Usage: ${total_usage / 100:.4f}")

        # Extract daily breakdown if available
        if 'daily_costs' in billing_data:
            print(f"\n📅 Daily Breakdown:")
            print("-" * 40)
            for daily in billing_data['daily_costs']:
                date = daily.get('timestamp')
                cost = daily.get('line_items', [{}])[0].get('cost', 0) / 100
                print(f"{date}: ${cost:.4f}")
    else:
        print("\n📭 No billing data found for the specified period.")

def check_account_info(api_key):
    """Check account information and billing details"""
    print("\n" + "="*60)
    print("💳 ACCOUNT INFORMATION")
    print("="*60)
    
    # Check account details
    url = "https://api.openai.com/v1/organizations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            org_data = response.json()
            if 'data' in org_data and org_data['data']:
                org = org_data['data'][0]
                print(f"Organization: {org.get('name', 'Unknown')}")
                print(f"Organization ID: {org.get('id', 'Unknown')}")
        else:
            print("⚠️  Could not retrieve organization information")
            
    except Exception as e:
        print(f"⚠️  Error retrieving account info: {e}")

def main():
    """Main function to run the usage checker"""
    try:
        print("🚀 OpenAI API Usage Checker")
        print("="*40)

        # Load API key
        api_key = load_api_key()

        # Check account information
        check_account_info(api_key)

        # Method 1: Check usage for specific dates (last few days)
        print(f"\n{'='*60}")
        print("📊 DAILY USAGE CHECK")
        print('='*60)

        for i in range(1, 4):  # Check last 3 days
            date = datetime.now().date() - timedelta(days=i)
            print(f"\n--- Day {i} ({date}) ---")
            check_api_usage(api_key, date)

        # Method 2: Try billing usage endpoint
        print(f"\n{'='*60}")
        print("� BILLING USAGE CHECK")
        print('='*60)
        check_billing_usage(api_key, days_back=30)

        # Method 3: Simple API test to verify key works
        print(f"\n{'='*60}")
        print("🔧 API KEY TEST")
        print('='*60)
        test_api_key(api_key)

        print(f"\n{'='*60}")
        print("✅ Usage check completed!")
        print("="*60)

    except Exception as e:
        print(f"❌ Error: {e}")

def test_api_key(api_key):
    """Test if the API key works with a simple request"""
    if OpenAI is None:
        print("❌ OpenAI library not available. Cannot test API key.")
        return

    try:
        client = OpenAI(api_key=api_key)

        # Make a minimal test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("✅ API key is working! Successfully connected to OpenAI.")
        print(f"📊 Test response: {response.choices[0].message.content}")

        # Try to get models list
        models = client.models.list()
        print(f"📋 Available models: {len(models.data)} models found")

    except Exception as e:
        print(f"❌ API key test failed: {str(e)}")
        print("   This might indicate:")
        print("   • Insufficient quota/credits")
        print("   • Invalid API key")
        print("   • Network connectivity issues")

if __name__ == "__main__":
    main()
