#!/usr/bin/env python3
"""
Git operations helper for HyperLanguage workspace.

Manages secure Git credentials and automated push operations using tokens
from the hidden token file (~/.openclaw/secrets/.api_keys.json).
"""

import subprocess
import sys
from pathlib import Path
from token_manager import TokenManager

class GitManager:
    """Manage Git operations with secure token handling."""
    
    @staticmethod
    def config_git_credentials():
        """Configure Git with credentials from token file."""
        tokens = TokenManager.load_tokens()
        git_creds = tokens.get('git', {})
        
        user = git_creds.get('user')
        token = git_creds.get('token')
        
        if not user or not token:
            print("❌ Git credentials not configured")
            print("   Run: python token_manager.py setup")
            return False
        
        try:
            # Configure git user
            subprocess.run(['git', 'config', 'user.name', user], 
                         capture_output=True, check=True)
            print(f"✓ Git user configured: {user}")
            
            # Store token in git config (safe location with restricted access)
            token_file = Path.home() / '.git-credentials'
            with open(token_file, 'w') as f:
                f.write(f"https://{user}:{token}@github.com\n")
            
            # Restrict file permissions (rw-------)
            import os
            os.chmod(token_file, 0o600)
            
            # Configure git to use credential file
            subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'],
                         capture_output=True, check=True)
            print(f"✓ Git credentials configured securely")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to configure Git: {e}")
            return False
    
    @staticmethod
    def add_all_and_commit(message: str) -> bool:
        """Add all changes and commit with message."""
        try:
            subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
            print("✓ Added all changes")
            
            subprocess.run(['git', 'commit', '-m', message], 
                         check=True, capture_output=True)
            print(f"✓ Committed: {message}")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Git commit failed: {e}")
            return False
    
    @staticmethod
    def push(branch: str = 'main') -> bool:
        """Push to remote repository."""
        try:
            # Make sure credentials are configured
            GitManager.config_git_credentials()
            
            subprocess.run(['git', 'push', 'origin', branch], 
                         check=True, capture_output=True)
            print(f"✓ Pushed to origin/{branch}")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Git push failed: {e}")
            return False
    
    @staticmethod
    def quick_commit_push(message: str, branch: str = 'main') -> bool:
        """Quick commit and push in one operation."""
        if GitManager.add_all_and_commit(message):
            return GitManager.push(branch)
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Git operations with secure token management')
    parser.add_argument('action', choices=['config', 'commit', 'push', 'quick'],
                       help='Action to perform')
    parser.add_argument('-m', '--message', default='Update', 
                       help='Commit message for commit/quick actions')
    parser.add_argument('-b', '--branch', default='main',
                       help='Branch to push to (default: main)')
    
    args = parser.parse_args()
    
    if args.action == 'config':
        GitManager.config_git_credentials()
    elif args.action == 'commit':
        GitManager.add_all_and_commit(args.message)
    elif args.action == 'push':
        GitManager.push(args.branch)
    elif args.action == 'quick':
        if GitManager.quick_commit_push(args.message, args.branch):
            print("\n✓ Git push completed successfully!")
        else:
            print("\n❌ Git push failed")
            sys.exit(1)
