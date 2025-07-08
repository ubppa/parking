import sqlite3

class Database:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE
            )
        ''')
        self.connection.commit()

    def insert_user(self, name, email):
        self.cursor.execute('''
            INSERT INTO users (name, email) VALUES (?, ?)
        ''', (name, email))
        self.connection.commit()

    def get_users(self):
        self.cursor.execute('SELECT * FROM users')
        return self.cursor.fetchall()

    def close(self):
        self.connection.close()

def main():
    db = Database('app.db')

    while True:
        action = input("Choose an action: [1] Add User [2] View Users [3] Exit: ")
        if action == '1':
            name = input("Enter name: ")
            email = input("Enter email: ")
            db.insert_user(name, email)
            print("User added.")
        elif action == '2':
            users = db.get_users()
            for user in users:
                print(f"ID: {user[0]}, Name: {user[1]}, Email: {user[2]}")
        elif action == '3':
            db.close()
            break
        else:
            print("Invalid action. Please try again.")

if __name__ == "__main__":
    main()