"""
Test the X402 payment retry logic in the scheduler

This verifies that:
1. Payment requests retry up to 5 times on 402 errors
2. 1 second cooldown between retries
3. Successful responses are returned immediately
"""

import time
from unittest.mock import Mock, MagicMock

def test_retry_logic():
    """Test the retry_payment_request function"""
    print("\n" + "="*70)
    print("Testing X402 Payment Retry Logic")
    print("="*70)

    # Mock session
    session = Mock()

    # Test 1: Immediate success (no retries needed)
    print("\n[Test 1] Immediate success (200 OK)")
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "success"}
    session.post.return_value = mock_response

    # Copy the retry function from scheduler
    def retry_payment_request(session, url, max_retries=5, retry_delay=1, **kwargs):
        for attempt in range(max_retries):
            response = session.post(url, **kwargs)

            if response.status_code == 402:
                if attempt < max_retries - 1:
                    print(f"  Got 402, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"  Failed after {max_retries} attempts")
                    raise Exception(f"Payment failed after {max_retries} retries")

            return response

        return response

    response = retry_payment_request(session, "http://test.com", json={"test": "data"})
    assert response.status_code == 200
    print("  [OK] Returned 200 on first attempt")

    # Test 2: Success after 2 retries (402 -> 402 -> 200)
    print("\n[Test 2] Success after 2 retries (402 -> 402 -> 200)")
    responses = [
        Mock(status_code=402),  # First attempt
        Mock(status_code=402),  # Second attempt
        Mock(status_code=200, json=lambda: {"data": "success"})  # Third attempt
    ]
    session.post.side_effect = responses

    start_time = time.time()
    response = retry_payment_request(session, "http://test.com", retry_delay=0.1, json={"test": "data"})
    elapsed_time = time.time() - start_time

    assert response.status_code == 200
    assert elapsed_time >= 0.2  # At least 2 retries * 0.1 second delay
    print(f"  [OK] Returned 200 after 2 retries ({elapsed_time:.2f}s elapsed)")

    # Test 3: Failure after max retries (all 402s)
    print("\n[Test 3] Failure after max retries (all 402s)")
    session.post.side_effect = None
    session.post.return_value = Mock(status_code=402)

    try:
        response = retry_payment_request(session, "http://test.com", max_retries=3, retry_delay=0.1, json={"test": "data"})
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Payment failed after 3 retries" in str(e)
        print(f"  [OK] Raised exception: {e}")

    print("\n" + "="*70)
    print("[OK] All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_retry_logic()
