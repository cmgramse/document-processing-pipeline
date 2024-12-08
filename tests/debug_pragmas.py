import sqlite3

def test_pragma_settings():
    # Create a test database
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    
    # Test each setting
    print("Initial settings:")
    c.execute('PRAGMA temp_store')
    print(f"temp_store = {c.fetchone()[0]}")
    
    print("\nTrying to set temp_store = 2:")
    c.execute('PRAGMA temp_store = 2')
    c.execute('PRAGMA temp_store')
    print(f"temp_store = {c.fetchone()[0]}")
    
    print("\nTrying with isolation_level = None:")
    conn.isolation_level = None
    c.execute('PRAGMA temp_store = 2')
    c.execute('PRAGMA temp_store')
    print(f"temp_store = {c.fetchone()[0]}")
    
    print("\nTrying with immediate transaction:")
    c.execute('BEGIN IMMEDIATE')
    c.execute('PRAGMA temp_store = 2')
    c.execute('PRAGMA temp_store')
    print(f"temp_store = {c.fetchone()[0]}")
    
    conn.close()

if __name__ == '__main__':
    test_pragma_settings()
