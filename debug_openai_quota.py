import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

def load_api_key():
    """Load and validate OpenAI API key"""
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("❌ No API key found. Please set OPENAI_API_KEY in your .env file.")
    elif not api_key.startswith('sk-'):
        raise ValueError("❌ Invalid API key format. OpenAI keys should start with 'sk-'")
    elif api_key.strip() != api_key:
        raise ValueError("❌ API key contains leading or trailing spaces. Please remove them.")

    print("✅ API key loaded successfully!")
    print(f"   Key preview: {api_key[:10]}...{api_key[-4:]}")

    # Analyze key structure for additional insights
    analyze_api_key_structure(api_key)

    return api_key.strip()

def analyze_api_key_structure(api_key):
    """Analyze API key structure to identify potential issues"""
    print(f"\n🔍 API Key Analysis:")
    print(f"   • Length: {len(api_key)} characters")
    print(f"   • Format: {api_key[:7]}...{api_key[-4:]}")

    # Check if it's a project key vs user key
    if api_key.startswith('sk-proj-'):
        print(f"   • Type: PROJECT API KEY")
        print(f"   • Note: Project keys have specific limitations and billing setup")
    elif api_key.startswith('sk-'):
        print(f"   • Type: USER/ORGANIZATION API KEY")

    # Check key segments
    segments = api_key.split('-')
    print(f"   • Segments: {len(segments)} parts")

    if len(segments) >= 3:
        if segments[1] == 'proj':
            print(f"   ⚠️  PROJECT KEY DETECTED - Common issues:")
            print(f"      - Project may not have billing enabled")
            print(f"      - Project may not have usage limits set")
            print(f"      - Project may be in wrong organization")
            print(f"      - Project may need explicit model access")

def check_billing_subscription(api_key):
    """Check billing subscription details"""
    print("\n" + "="*60)
    print("💳 BILLING SUBSCRIPTION CHECK")
    print("="*60)
    
    url = "https://api.openai.com/v1/dashboard/billing/subscription"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Subscription data retrieved successfully!")
            print(json.dumps(data, indent=2))
            
            # Extract key information
            if 'plan' in data:
                print(f"\n📋 Plan Details:")
                plan = data['plan']
                print(f"   • Plan ID: {plan.get('id', 'Unknown')}")
                print(f"   • Title: {plan.get('title', 'Unknown')}")
            
            if 'hard_limit_usd' in data:
                print(f"   • Hard Limit: ${data['hard_limit_usd']}")
            
            if 'soft_limit_usd' in data:
                print(f"   • Soft Limit: ${data['soft_limit_usd']}")
                
        else:
            print(f"❌ Failed to get subscription: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error checking subscription: {e}")

def check_billing_usage_detailed(api_key):
    """Check detailed billing usage"""
    print("\n" + "="*60)
    print("💰 DETAILED BILLING USAGE")
    print("="*60)
    
    # Check current month usage
    now = datetime.now()
    start_date = now.replace(day=1).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    
    url = "https://api.openai.com/v1/dashboard/billing/usage"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {
        "start_date": start_date,
        "end_date": end_date
    }
    
    try:
        print(f"Checking usage from {start_date} to {end_date}")
        response = requests.get(url, headers=headers, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Usage data retrieved successfully!")
            print(json.dumps(data, indent=2))
            
            # Calculate total usage
            total_usage = data.get('total_usage', 0) / 100  # Convert cents to dollars
            print(f"\n💰 Total Usage This Month: ${total_usage:.4f}")
            
        else:
            print(f"❌ Failed to get usage: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error checking usage: {e}")

def check_credit_grants(api_key):
    """Check available credit grants"""
    print("\n" + "="*60)
    print("🎁 CREDIT GRANTS CHECK")
    print("="*60)
    
    url = "https://api.openai.com/v1/dashboard/billing/credit_grants"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Credit grants data retrieved!")
            print(json.dumps(data, indent=2))
            
            if 'grants' in data:
                total_granted = sum(grant.get('grant_amount', 0) for grant in data['grants']) / 100
                total_used = sum(grant.get('used_amount', 0) for grant in data['grants']) / 100
                remaining = total_granted - total_used
                
                print(f"\n💰 Credit Summary:")
                print(f"   • Total Granted: ${total_granted:.2f}")
                print(f"   • Total Used: ${total_used:.2f}")
                print(f"   • Remaining: ${remaining:.2f}")
                
        else:
            print(f"❌ Failed to get credit grants: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error checking credit grants: {e}")

def test_minimal_request(api_key):
    """Test with the most minimal possible request"""
    print("\n" + "="*60)
    print("🧪 MINIMAL API TEST")
    print("="*60)
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Ultra minimal request
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "temperature": 0
    }
    
    try:
        print("Testing minimal chat completion request...")
        response = requests.post(url, headers=headers, json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            error_data = response.json()
            print(f"Error details: {json.dumps(error_data, indent=2)}")
            
            # Analyze the error
            if 'error' in error_data:
                error = error_data['error']
                error_type = error.get('type', 'unknown')
                error_code = error.get('code', 'unknown')
                error_message = error.get('message', 'No message')
                
                print(f"\n🔍 Error Analysis:")
                print(f"   • Type: {error_type}")
                print(f"   • Code: {error_code}")
                print(f"   • Message: {error_message}")
                
                if error_code == 'insufficient_quota':
                    print(f"\n💡 Quota Issue Detected:")
                    print(f"   • This means you've exceeded your usage limits")
                    print(f"   • Even with a $5 limit, you might have used it all")
                    print(f"   • Check your OpenAI dashboard for exact usage")
                    print(f"   • You may need to add more credits or wait for reset")
            
    except Exception as e:
        print(f"❌ Error during API test: {e}")

def check_models_access(api_key):
    """Check which models are accessible"""
    print("\n" + "="*60)
    print("🤖 MODELS ACCESS CHECK")
    print("="*60)
    
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            models = [model['id'] for model in data['data']]
            print(f"✅ Found {len(models)} accessible models:")
            
            # Show relevant models
            chat_models = [m for m in models if 'gpt' in m.lower()]
            print(f"\n🗣️  Chat Models Available:")
            for model in sorted(chat_models):
                print(f"   • {model}")
                
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")

def check_project_specific_issues(api_key):
    """Check for project-specific configuration issues"""
    print("\n" + "="*60)
    print("🏗️  PROJECT CONFIGURATION CHECK")
    print("="*60)

    if not api_key.startswith('sk-proj-'):
        print("ℹ️  Not a project key - skipping project-specific checks")
        return

    print("🔍 Detected PROJECT API KEY - checking common issues:")

    # Check organization access
    url = "https://api.openai.com/v1/organizations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        print(f"\n� Organization Access Check:")
        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            orgs = response.json()
            print(f"   ✅ Can access organizations")
            if 'data' in orgs and orgs['data']:
                for org in orgs['data']:
                    print(f"   • Org: {org.get('name', 'Unknown')} (ID: {org.get('id', 'Unknown')})")
            else:
                print(f"   ⚠️  No organizations found - this might be the issue!")
        else:
            print(f"   ❌ Cannot access organizations: {response.text}")

    except Exception as e:
        print(f"   ❌ Error checking organizations: {e}")

def check_payment_method_status():
    """Provide guidance on payment method issues"""
    print("\n" + "="*60)
    print("💳 PAYMENT METHOD DIAGNOSTIC")
    print("="*60)

    print("🔍 Common payment/billing issues that cause quota errors:")
    print("\n1. 📋 BILLING SETUP ISSUES:")
    print("   • No payment method added to account")
    print("   • Payment method declined or expired")
    print("   • Billing address issues")
    print("   • Account verification pending")

    print("\n2. 🏗️  PROJECT BILLING ISSUES:")
    print("   • Project not linked to billing account")
    print("   • Project usage limits set to $0")
    print("   • Project not enabled for API usage")
    print("   • Wrong organization selected for project")

    print("\n3. 🌍 GEOGRAPHIC RESTRICTIONS:")
    print("   • OpenAI not available in your country")
    print("   • VPN/proxy causing location issues")
    print("   • Account flagged for unusual activity")

    print("\n4. 📊 ACCOUNT STATUS ISSUES:")
    print("   • Account suspended or under review")
    print("   • Free trial expired without payment method")
    print("   • Account created but not fully activated")

def check_account_verification_status(api_key):
    """Check if account verification might be the issue"""
    print("\n" + "="*60)
    print("✅ ACCOUNT VERIFICATION CHECK")
    print("="*60)

    # Try a very basic endpoint that should work for any valid key
    url = "https://api.openai.com/v1/models/gpt-3.5-turbo"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        print(f"Model Info Check - Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✅ Basic API access works - account is verified")
        elif response.status_code == 401:
            print("❌ Authentication failed - API key might be invalid")
        elif response.status_code == 403:
            print("❌ Forbidden - Account might be suspended or restricted")
        elif response.status_code == 429:
            print("⚠️  Rate limited - but this confirms account exists")
        else:
            print(f"❌ Unexpected response: {response.text}")

    except Exception as e:
        print(f"❌ Error checking account: {e}")

def main():
    """Main debugging function"""
    try:
        print("🔍 OpenAI Quota Debugging Tool - Enhanced for Zero-Usage Issues")
        print("="*70)

        # Load API key
        api_key = load_api_key()

        # Enhanced diagnostic checks
        check_account_verification_status(api_key)
        check_project_specific_issues(api_key)
        check_payment_method_status()
        check_models_access(api_key)
        test_minimal_request(api_key)

        print(f"\n{'='*60}")
        print("🏁 ENHANCED DEBUGGING COMPLETED")
        print("="*60)
        print("📋 Likely Issues for Zero-Usage Quota Errors:")
        print("   1. 💳 NO PAYMENT METHOD: Add a credit card to your account")
        print("   2. 🏗️  PROJECT BILLING: Enable billing for your project")
        print("   3. 📊 ACCOUNT VERIFICATION: Complete account verification")
        print("   4. 🌍 GEOGRAPHIC: Check if OpenAI is available in your region")
        print("   5. 🔧 PROJECT SETUP: Ensure project has proper permissions")

        print(f"\n🔗 Essential Links:")
        print("   • Billing: https://platform.openai.com/account/billing")
        print("   • Projects: https://platform.openai.com/settings/organization/projects")
        print("   • Usage: https://platform.openai.com/usage")
        print("   • API Keys: https://platform.openai.com/api-keys")

    except Exception as e:
        print(f"❌ Debugging failed: {e}")

if __name__ == "__main__":
    main()
