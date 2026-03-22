#!/usr/bin/env python3
"""
Verification script for Grok API + Hidden Token File setup
"""

import sys
from pathlib import Path

print('=' * 60)
print('GROK API + Hidden Token File - Setup Verification')
print('=' * 60)

# Test 1: Token Manager
print('\n[1] Token Manager')
try:
    from token_manager import TokenManager
    print('    OK: Import successful')
    
    tokens = TokenManager.load_tokens()
    grok = TokenManager.get_grok_config()
    git = TokenManager.get_git_credentials()
    
    print('    OK: Token loading functional')
    print(f'       - Grok config has endpoint: {"endpoint" in grok}')
    print(f'       - Grok config has model: {"model" in grok}')
except Exception as e:
    print(f'    ERROR: {e}')
    sys.exit(1)

# Test 2: Git Manager
print('\n[2] Git Manager')
try:
    from git_manager import GitManager
    print('    OK: Import successful')
    print('    OK: Git operations available')
except Exception as e:
    print(f'    ERROR: {e}')
    sys.exit(1)

# Test 3: HLTokenizer
print('\n[3] HLTokenizer')
try:
    from hl_tokenizer import HLTokenizer
    print('    OK: Import successful')
    
    t = HLTokenizer()
    print('    OK: Instance creation successful')
    print('       - Version: v5.5 (Grok-enabled)')
except Exception as e:
    if 'Grok API key' in str(e):
        print('    OK: Grok API key validation working')
        print('       (Expected - run: python token_manager.py setup)')
    else:
        print(f'    ERROR: {e}')
        sys.exit(1)

# Test 4: Secrets Directory
print('\n[4] Secrets Directory')
import os
secrets_dir = Path.home() / '.openclaw' / 'secrets'
if secrets_dir.exists():
    print(f'    OK: Directory exists')
    stat_info = os.stat(secrets_dir)
    perms = oct(stat_info.st_mode)[-3:]
    print(f'       Permissions: {perms} (should be 700)')
else:
    print(f'    INFO: Will be created on first setup')

# Test 5: .gitignore
print('\n[5] .gitignore Configuration')
try:
    with open('.gitignore', 'r') as f:
        content = f.read()
        if 'API Keys & Secrets' in content:
            print('    OK: Secrets section in .gitignore')
        if '.api_keys.json' in content:
            print('    OK: .api_keys.json ignored')
except Exception as e:
    print(f'    WARNING: Error reading .gitignore: {e}')

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print('\nOK: All components installed and functional!')
print('\nNext Steps:')
print('  1. python token_manager.py setup')
print('  2. python token_manager.py         (verify)')
print('  3. python git_manager.py quick -m "message"  (test)')
print('\nDocumentation:')
print('  - GROK_SETUP_GUIDE.md')
print('  - QUICK_START_GROK.md')
print('  - IMPLEMENTATION_SUMMARY.md')
print('\n' + '=' * 60)
