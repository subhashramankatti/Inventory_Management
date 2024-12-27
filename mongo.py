from pymongo import MongoClient

# Replace the connection string below with your Atlas connection string
connection_string = "mongodb+srv://sachinb665:sachinb665@cluster0.ofadrht.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a MongoClient to interact with MongoDB
client = MongoClient(connection_string)

# Access a specific database
db = client['dd']

# Access a collection within the database
collection = db['dd']

# Example query: Find one document in the collection
document = collection.find_one()

print(document)
