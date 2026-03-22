# ✅ Grok API + Hidden Token File Setup - Complete

## What's Been Done

### 1. **Grok API Integration** (xAI)
- ✅ Replaced Qwen with Grok for language tagging
- ✅ Updated `_init_llm()` to use xAI Grok endpoint
- ✅ Updated model to use `grok-2`
- ✅ Added better error messages for missing API keys

### 2. **Token Manager** (`token_manager.py`)
- ✅ Secure token loading from hidden file
- ✅ Environment variable override support
- ✅ Interactive setup wizard
- ✅ Token status checking

### 3. **Git Manager** (`git_manager.py`)
- ✅ Secure Git credential management
- ✅ Automated commit and push operations
- ✅ Uses tokens from hidden file

### 4. **Security**
- ✅ Hidden secrets directory: `~/.openclaw/secrets/`
- ✅ File permissions: `0o600` (rw-------)
- ✅ Directory permissions: `0o700` (rwx------)
- ✅ Updated `.gitignore` to exclude secrets
- ✅ Added `.gitignore` to secrets directory

### 5. **Documentation**
- ✅ `GROK_SETUP_GUIDE.md` - Complete setup instructions
- ✅ Token file location and format documented
- ✅ API key acquisition guide
- ✅ Troubleshooting section
- ✅ Security best practices

---

## Quick Start

### Step 1: Initialize Tokens
```bash
cd /home/wanwa/.openclaw/workspace/hyper-language
python token_manager.py setup
```

You'll be prompted for:
- **Grok API Key** (required) - Get from https://console.x.ai/
- **DashScope API Key** (optional) - Fallback LLM
- **Git Username** (optional) - For GitHub operations
- **Git Personal Access Token** (optional) - For GitHub push

### Step 2: Verify Setup
```bash
python token_manager.py
# Output:
# Token Status:
#   Grok API: ✓
#   DashScope API: ✗ (or ✓)
#   Git credentials: ✗ (or ✓)
```

### Step 3: Use HLTokenizer with Grok
```python
from hl_tokenizer import HLTokenizer

tokenizer = HLTokenizer()

# Now uses Grok API automatically
text = "I went to 銀行 to check my 残高"
result = tokenizer._llm_language_tag(text)
```

### Step 4: Push to Git (Optional)
```bash
python git_manager.py quick -m "Update with Grok API" -b main
```

---

## File Structure

```
~/.openclaw/
└── secrets/                          # Hidden (chmod 700)
    ├── .gitignore                   # Prevents accidents
    └── .api_keys.json               # Token file (chmod 600)

/home/wanwa/.openclaw/workspace/hyper-language/
├── token_manager.py                 # NEW: Token loader
├── git_manager.py                   # NEW: Git operations
├── hl_tokenizer.py                  # UPDATED: Uses Grok
├── GROK_SETUP_GUIDE.md             # NEW: Setup guide
├── .gitignore                       # UPDATED: Ignores secrets
└── (other files)
```

---

## Key Features

### Token Manager
```bash
# Setup wizard
python token_manager.py setup

# Check token status
python token_manager.py

# Use in code
from token_manager import TokenManager
grok_config = TokenManager.get_grok_config()
git_creds = TokenManager.get_git_credentials()
```

### Grok API
```python
from hl_tokenizer import HLTokenizer

tokenizer = HLTokenizer()
# Auto-loads Grok config from ~/.openclaw/secrets/.api_keys.json

# Tokens loaded in order:
# 1. ~/.openclaw/secrets/.api_keys.json (file)
# 2. GROK_API_KEY environment variable (override)
```

### Git Operations
```bash
# Configure Git credentials
python git_manager.py config

# Create commit
python git_manager.py commit -m "Update tokenizer"

# Push to main
python git_manager.py push -b main

# Quick commit + push
python git_manager.py quick -m "message" -b main
```

---

## Security Highlights

✅ **No Hardcoded Tokens**
- All keys in hidden file outside Git repo
- Never committed to version control

✅ **File Permissions**
- Token file: `0o600` (readable only by you)
- Secrets directory: `0o700` (accessible only by you)

✅ **Environment Variable Fallback**
- Can use env vars instead of file
- Perfect for CI/CD environments

✅ **Git Credentials Protected**
- Stored in `.git-credentials` with `0o600` permissions
- Used via `git credential.helper`

---

## Updated HLTokenizer Changes

### Version
- **Before**: v5.4 (Qwen)
- **After**: v5.5 (Grok)

### Endpoint
- **Before**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **After**: `https://api.x.ai/v1`

### Model
- **Before**: `qwen-turbo`
- **After**: `grok-2`

### Token Loading
- **Before**: Only `DASHSCOPE_API_KEY` env var
- **After**: TokenManager (file + env var override)

### Error Handling
- **Before**: Generic OpenAI errors
- **After**: Clear message to setup token file

---

## Next Steps

1. **Get API Keys**
   - Grok: https://console.x.ai/
   - Optional DashScope: https://dashscope.aliyuncs.com/
   - Optional Git: https://github.com/settings/tokens

2. **Run Setup**
   ```bash
   python token_manager.py setup
   ```

3. **Test Everything**
   ```bash
   python token_manager.py      # Check tokens
   python -c "from hl_tokenizer import HLTokenizer; HLTokenizer()"
   ```

4. **Start Using**
   - Use HLTokenizer with Grok AI
   - Push updates via git_manager.py
   - All tokens stay secure and hidden

---

## Support

| Issue | Solution |
|-------|----------|
| "Grok API key not found" | Run `python token_manager.py setup` |
| "Invalid API key" | Regenerate key on xAI console |
| Import errors | Both files in same directory? |
| Permission denied | Run `chmod 600 ~/.openclaw/secrets/.api_keys.json` |
| Git push fails | Run `python git_manager.py config` |

---

## Version Info

- **Tokenizer**: v5.5+ (Grok-enabled)
- **Token Manager**: v1.0
- **Git Manager**: v1.0
- **LLM**: Grok (xAI)
- **Date**: 2026-03-14

---

**Status**: ✅ **Ready to Use**
- All dependencies installed
- Token manager functional
- HLTokenizer updated
- Git manager ready
- Security configured

**Next Action**: `python token_manager.py setup`
