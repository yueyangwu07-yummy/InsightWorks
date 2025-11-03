"""Test suite for all agent tools."""

import requests
import time
from typing import Optional


# Test configuration
BASE_URL = "http://localhost:8000/api/v1"
CHAT_ENDPOINT = f"{BASE_URL}/chatbot/chat"
AUTH_ENDPOINT = f"{BASE_URL}/auth"
REQUEST_TIMEOUT = 60  # Longer timeout for tool calls
DELAY_BETWEEN_TESTS = 2  # seconds


class ToolTester:
    """Test all agent tools."""
    
    def __init__(self):
        self.test_results = []
        self.auth_token: Optional[str] = None
        self.session_id: Optional[str] = None
    
    def _get_auth_token(self) -> bool:
        """Get authentication token by creating a test user and session."""
        try:
            # Try to register a test user
            register_data = {
                "email": f"test_tools_{int(time.time())}@example.com",
                "username": f"test_tools_{int(time.time())}",
                "password": "TestPassword123!",
            }
            
            register_response = requests.post(
                f"{AUTH_ENDPOINT}/register",
                json=register_data,
                timeout=10
            )
            
            # If registration fails, try to login (user might already exist)
            if register_response.status_code not in [200, 201]:
                # Try login instead
                login_response = requests.post(
                    f"{AUTH_ENDPOINT}/login",
                    data={
                        "username": register_data["email"],
                        "password": register_data["password"]
                    },
                    timeout=10
                )
                
                if login_response.status_code != 200:
                    print(f"❌ Authentication failed: {login_response.text}")
                    return False
                
                token_data = login_response.json()
                user_token = token_data["access_token"]
            else:
                register_data_response = register_response.json()
                user_token = register_data_response["token"]["access_token"]
            
            # Create a session
            session_response = requests.post(
                f"{AUTH_ENDPOINT}/session",
                headers={"Authorization": f"Bearer {user_token}"},
                timeout=10
            )
            
            if session_response.status_code != 200:
                print(f"❌ Session creation failed: {session_response.text}")
                return False
            
            session_data = session_response.json()
            self.auth_token = session_data["token"]["access_token"]
            self.session_id = session_data["session_id"]
            
            print(f"✅ Authentication successful (session: {self.session_id[:8]}...)")
            return True
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def _test_tool(self, query: str) -> bool:
        """Test a single tool by sending a query."""
        if not self.auth_token:
            if not self._get_auth_token():
                return False
        
        url = CHAT_ENDPOINT
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "messages": [{"role": "user", "content": query}]
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                response_length = len(str(data))
                print(f"   ✓ Response received ({response_length} chars)")
                
                if "messages" in data and data["messages"]:
                    last_msg = data["messages"][-1]
                    content = last_msg.get("content", "")
                    print(f"   Response preview: {content[:300]}...")
                
                return True
            else:
                print(f"   ✗ Error: {response.text[:500]}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️  Request timeout (> {REQUEST_TIMEOUT}s)")
            return False
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tool tests."""
        print("=" * 70)
        print("TOOL FUNCTIONALITY TEST SUITE")
        print("=" * 70)
        
        # Authenticate first
        print("\n🔐 Authenticating...")
        if not self._get_auth_token():
            print("❌ Failed to authenticate. Cannot proceed with tests.")
            return
        
        tests = [
            ("VIN Decoder", "Decode VIN 1HGBH41JXMN109186"),
            ("Straight Distance", "What's the straight-line distance from New York to Boston?"),
            ("Driving Distance", "How long does it take to drive from NYC to Boston?"),
            ("Unit Converter", "Convert 35 PSI to bar"),
            ("Timezone Converter", "What time is 3 PM EST in PST?"),
            ("Recall Checker", "Check recalls for 2020 Honda Accord"),
            ("Tire Pressure", "What tire pressure for my SUV?"),
            ("Traffic Incident", "Any traffic incidents in New York?"),
            ("Road Condition", "What are road conditions on I-95?"),
            ("Weather Alert", "Any weather alerts in Massachusetts?"),
            ("DuckDuckGo Search", "Search for best restaurants in Boston"),
        ]
        
        for tool_name, query in tests:
            print(f"\n{'='*70}")
            print(f"Testing: {tool_name}")
            print(f"Query: {query}")
            print(f"{'='*70}")
            
            try:
                result = self._test_tool(query)
                self.test_results.append((tool_name, result, None))
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"\nResult: {status}")
            except Exception as e:
                self.test_results.append((tool_name, False, str(e)))
                print(f"\n❌ EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(DELAY_BETWEEN_TESTS)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TOOL TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)
        
        print(f"\nTotal Tools Tested: {total}")
        print(f"Working: {passed} ✅")
        print(f"Broken: {total - passed} ❌")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        print("\nDetailed Results:")
        for tool_name, result, error in self.test_results:
            status = "✅" if result else "❌"
            print(f"{status} {tool_name}")
            if error:
                print(f"   Error: {error}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n🔧 Starting Tool Tests...")
    print("Make sure server is running!")
    print("\nStarting in 2 seconds...\n")
    time.sleep(2)
    
    tester = ToolTester()
    tester.run_all_tests()

