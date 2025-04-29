import os
import pymongo
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    print("Error: MONGODB_URI not found in .env file")
else:
    try:
        print(f"Attempting to connect to: {MONGODB_URI[:30]}...") # Print partial URI for verification
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000) # 5 second timeout
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("MongoDB connection successful!")
        client.close()
    except pymongo.errors.ConfigurationError as e:
        print(f"MongoDB Configuration Error: {e}")
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"MongoDB Connection Timeout: {e}")
        print("Check firewall, IP allowlist, and if the server is running.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")