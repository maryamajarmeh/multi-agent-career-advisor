memory_store = {}

def get_user_memory(user_id):
    return memory_store.get(user_id, {})

def save_user_memory(user_id, data):
    memory_store[user_id] = data