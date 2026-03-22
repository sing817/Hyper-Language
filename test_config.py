from token_manager import TokenManager
config = TokenManager.get_grok_config()
print('Model:', config.get('model'))