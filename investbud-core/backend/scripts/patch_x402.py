"""
Patch x402 package to make error field optional in x402PaymentRequiredResponse

This is a workaround for a bug in x402==0.2.1 where the error field
is required but servers don't always return it.

Run this after installing x402:
    python scripts/patch_x402.py
"""

import sys
from pathlib import Path

def patch_x402():
    # Find the x402 types.py file in site-packages
    # Try multiple possible locations
    import sysconfig
    site_packages = Path(sysconfig.get_path('purelib'))
    types_file = site_packages / 'x402' / 'types.py'

    if not types_file.exists():
        print(f"[ERROR] x402 not found at {types_file}")
        print("  Make sure x402 is installed: uv pip install x402")
        return False

    # Read the file
    with open(types_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if 'error: Optional[str] = None' in content:
        print("[OK] x402 already patched")
        return True

    # Apply the patch
    if 'error: str\n' in content:
        content = content.replace('    error: str\n', '    error: Optional[str] = None\n')

        with open(types_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print("[OK] Patched x402 PaymentRequiredResponse.error to be optional")
        return True
    else:
        print("[ERROR] Could not find 'error: str' in x402/types.py")
        print("  x402 may have been updated. Check if patch is still needed.")
        return False

if __name__ == "__main__":
    success = patch_x402()
    sys.exit(0 if success else 1)
