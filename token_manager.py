"""
Token Manager for HyperLanguage Tokenizer

Securely loads API keys and credentials from hidden token file.
Location: ~/.openclaw/secrets/.api_keys.json

Supports:
- Grok API (xAI)
- DashScope/Qwen API (Alibaba) - fallback
- Git credentials
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional

class TokenManager:
    """Manage API keys and credentials from hidden token file."""
    
    SECRETS_DIR = Path.home() / '.openclaw' / 'secrets'
    TOKEN_FILE = SECRETS_DIR / '.api_keys.json'
    
    @staticmethod
    def ensure_secrets_dir():
        """Create secrets directory with restricted permissions if needed."""
        TokenManager.SECRETS_DIR.mkdir(parents=True, exist_ok=True)
        # Set restrictive permissions (rwx------)
        os.chmod(TokenManager.SECRETS_DIR, 0o700)
    
    @staticmethod
    def load_tokens() -> Dict:
        """Load API keys from hidden token file or environment variables."""
        tokens = {}
        
        # Try to load from token file first
        if TokenManager.TOKEN_FILE.exists():
            with open(TokenManager.TOKEN_FILE, 'r') as f:
                file_tokens = json.load(f)
                tokens.update(file_tokens)
        
        # Override with environment variables if set
        if os.getenv('GROK_API_KEY'):
            tokens.setdefault('grok', {})['api_key'] = os.getenv('GROK_API_KEY')
        
        if os.getenv('DASHSCOPE_API_KEY'):
            tokens.setdefault('dashscope', {})['api_key'] = os.getenv('DASHSCOPE_API_KEY')
        
        if os.getenv('GIT_TOKEN'):
            tokens.setdefault('git', {})['token'] = os.getenv('GIT_TOKEN')
        
        return tokens
    
    @staticmethod
    def save_tokens(tokens: Dict):
        """Save API keys to hidden token file."""
        TokenManager.ensure_secrets_dir()
        
        # Ensure file has restricted permissions (rw-------)
        with open(TokenManager.TOKEN_FILE, 'w') as f:
            json.dump(tokens, f, indent=2)
        
        os.chmod(TokenManager.TOKEN_FILE, 0o600)
        print(f"✓ Tokens saved to {TokenManager.TOKEN_FILE}")
        print(f"  Permissions: 0o600 (rw-------)")
    
    @staticmethod
    def get_grok_api_key() -> Optional[str]:
        """Get Grok API key from token file or environment."""
        tokens = TokenManager.load_tokens()
        return tokens.get('grok', {}).get('api_key')
    
    @staticmethod
    def get_grok_config() -> Dict:
        """Get full Grok configuration."""
        tokens = TokenManager.load_tokens()
        grok_config = tokens.get('grok', {})
        
        # Default values if not provided
        defaults = {
            'endpoint': 'https://api.x.ai/v1',
            'model': 'grok-4-1-fast-reasoning'
        }
        
        for key, value in defaults.items():
            grok_config.setdefault(key, value)
        
        return grok_config
    
    @staticmethod
    def get_dashscope_api_key() -> Optional[str]:
        """Get DashScope API key (fallback)."""
        tokens = TokenManager.load_tokens()
        return tokens.get('dashscope', {}).get('api_key')
    
    @staticmethod
    def get_git_credentials() -> Dict:
        """Get Git credentials for push operations."""
        tokens = TokenManager.load_tokens()
        return tokens.get('git', {})
    
    @staticmethod
    def create_token_file(grok_key: str, dashscope_key: Optional[str] = None, 
                         git_user: Optional[str] = None, git_token: Optional[str] = None):
        """Create token file with provided credentials."""
        tokens = {
            'grok': {
                'api_key': grok_key,
                'endpoint': 'https://api.x.ai/v1',
                'model': 'grok-4-1-fast-reasoning'
            }
        }
        
        if dashscope_key:
            tokens['dashscope'] = {
                'api_key': dashscope_key,
                'endpoint': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'model': 'qwen-turbo'
            }
        
        if git_user or git_token:
            tokens['git'] = {
                'user': git_user or '',
                'token': git_token or ''
            }
        
        TokenManager.save_tokens(tokens)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        # Setup wizard
        print("Setting up token file...")
        grok_key = input("Enter Grok API key (xAI): ").strip()
        
        if not grok_key:
            print("Grok API key is required!")
            sys.exit(1)
        
        dashscope_key = input("Enter DashScope API key (optional): ").strip() or None
        git_user = input("Enter Git username (optional): ").strip() or None
        git_token = input("Enter Git personal access token (optional): ").strip() or None
        
        TokenManager.create_token_file(grok_key, dashscope_key, git_user, git_token)
        print("\n✓ Token file created successfully!")
        print(f"  Location: {TokenManager.TOKEN_FILE}")
        print("  Permissions: 0o600 (rw-------) - readable only by you")
    else:
        # Check tokens
        tokens = TokenManager.load_tokens()
        print("Token Status:")
        print(f"  Grok API: {'✓' if TokenManager.get_grok_api_key() else '✗'}")
        print(f"  DashScope API: {'✓' if TokenManager.get_dashscope_api_key() else '✗'}")
        print(f"  Git credentials: {'✓' if TokenManager.get_git_credentials() else '✗'}")
