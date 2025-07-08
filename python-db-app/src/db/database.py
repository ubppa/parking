class Database:
    def __init__(self, db_name):
        import sqlite3
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def create_table(self, table_name, columns):
        columns_with_types = ', '.join([f"{col} {col_type}" for col, col_type in columns.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_with_types});"
        self.cursor.execute(query)
        self.connection.commit()

    def insert_data(self, table_name, data):
        placeholders = ', '.join(['?'] * len(data))
        query = f"INSERT INTO {table_name} VALUES ({placeholders});"
        self.cursor.execute(query, tuple(data))
        self.connection.commit()

    def retrieve_data(self, table_name):
        query = f"SELECT * FROM {table_name};"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        self.connection.close()