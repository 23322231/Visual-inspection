from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://april910909:U7Fm6Le9KyUeUU9n@test.ytyoqoz.mongodb.net/?appName=test"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)