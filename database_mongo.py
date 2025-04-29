import os
import motor.motor_asyncio
import pymongo
import pandas as pd
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
import logging
import re # For regex search

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "ai_video_assistant")
CSV_FILE_PATH = "anshul copy data.csv"

client = None
db = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_db():
    """Creates MongoDB client and selects database."""
    global client, db
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        # You can add a check here to see if the server is available
        # await client.admin.command('ping') # Requires auth if enabled
        logger.info(f"Connected to MongoDB: {MONGODB_URI}, Database: {MONGODB_DB_NAME}")
        # Initialize collections and load data synchronously on startup
        init_db_sync()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        client = None
        db = None

def close_db():
    """Closes MongoDB client connection."""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed.")

def init_db_sync():
    """
    Synchronously ensures collections exist and loads landmarks from CSV
    if the landmarks collection is empty. Uses PyMongo for sync operations.
    Called once on startup.
    """
    try:
        sync_client = pymongo.MongoClient(MONGODB_URI)
        sync_db = sync_client[MONGODB_DB_NAME]
        logger.info("Running initial DB setup (synchronous)...")

        # Create Indexes (optional but recommended)
        try:
             # Index for faster interaction lookup
             sync_db.interactions.create_index([("session_id", pymongo.ASCENDING)], background=True)
             # Index for faster landmark name lookup (case-insensitive)
             # For text search, consider MongoDB's text indexes:
             # sync_db.landmarks.create_index([("landmark_name", pymongo.TEXT)], default_language='english', background=True)
             # Simple index for demo:
             sync_db.landmarks.create_index([("landmark_name", pymongo.ASCENDING)], background=True)
             # Unique index for landmarks based on original data structure
             sync_db.landmarks.create_index([("ulbcode", pymongo.ASCENDING), ("landmark_objectid", pymongo.ASCENDING)], unique=True, background=True)

             logger.info("Ensured indexes exist on 'interactions' and 'landmarks' collections.")
        except pymongo.errors.OperationFailure as e:
             logger.warning(f"Could not create indexes (might already exist or permissions issue): {e}")


        # Load Landmarks if collection is empty
        landmarks_collection = sync_db.landmarks
        if landmarks_collection.count_documents({}) == 0:
            logger.info("'landmarks' collection is empty. Attempting to load data from CSV...")
            try:
                df = pd.read_csv(CSV_FILE_PATH)
                # Clean column names
                df.columns = [col.strip().replace('"', '').strip() for col in df.columns]
                 # Clean string data
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].str.strip().str.replace('"', '').str.strip()

                # Convert specific columns to numeric if possible, handle errors
                for col in ['ulb_objectid', 'landmark_objectid']:
                     df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN
                # Latitude/Longitude - keep as string or convert carefully
                # df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                # df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

                # Remove rows with NaN in critical numeric fields if necessary
                # df.dropna(subset=['ulb_objectid', 'landmark_objectid'], inplace=True)

                # Convert DataFrame to list of dictionaries
                data_to_insert = df.to_dict('records')

                if data_to_insert:
                    logger.info(f"Inserting {len(data_to_insert)} records into 'landmarks' collection...")
                    landmarks_collection.insert_many(data_to_insert)
                    logger.info(f"Successfully loaded data into 'landmarks' collection.")
                else:
                    logger.warning("No valid data found in CSV after cleaning.")

            except FileNotFoundError:
                logger.error(f"CSV file not found: {CSV_FILE_PATH}")
            except pymongo.errors.BulkWriteError as bwe:
                logger.error(f"Error during bulk insert into landmarks (check for duplicates if unique index exists): {bwe.details}")
            except Exception as e:
                logger.error(f"Failed to load landmarks from CSV: {e}", exc_info=True)
        else:
            count = landmarks_collection.count_documents({})
            logger.info(f"'landmarks' collection already contains {count} documents. Skipping CSV load.")

        sync_client.close()
        logger.info("Initial DB setup finished.")

    except Exception as e:
        logger.error(f"Error during synchronous DB initialization: {e}", exc_info=True)


async def log_interaction_start(session_id: str | None = None) -> ObjectId | None:
    """Logs the start of an interaction and returns the MongoDB ObjectId."""
    if db is None: return None
    try:
        interaction_doc = {
            "session_id": session_id,
            "timestamp_start": datetime.utcnow(),
            "status": "pending",
            # Initialize other fields optionally
            "user_input_text": None,
            "stt_confidence": None,
            "llm_response_text": None,
            "generated_audio_path": None,
            "generated_video_path": None,
            "timestamp_end": None,
            "latency_stt_ms": None,
            "latency_llm_ms": None,
            "latency_tts_ms": None,
            "latency_lipsync_ms": None,
            "error_message": None
        }
        result = await db.interactions.insert_one(interaction_doc)
        return result.inserted_id
    except Exception as e:
        logger.error(f"Failed to log interaction start: {e}")
        return None

async def update_interaction(interaction_id: ObjectId, **kwargs):
    """Updates an interaction record with provided data using its ObjectId."""
    if db is None or not interaction_id: return

    update_fields = {}
    valid_keys = [
        "user_input_text", "stt_confidence", "llm_response_text",
        "generated_audio_path", "generated_video_path", "timestamp_end",
        "latency_stt_ms", "latency_llm_ms", "latency_tts_ms",
        "latency_lipsync_ms", "status", "error_message"
    ]

    for key, value in kwargs.items():
        if key in valid_keys:
            update_fields[key] = value
        else:
            logger.warning(f"Attempted to update interaction with unknown field: {key}")

    # Ensure timestamp_end is set when completing or failing
    if update_fields.get('status') in ['completed', 'completed_audio_only', 'failed'] and 'timestamp_end' not in update_fields:
        update_fields['timestamp_end'] = datetime.utcnow()

    if not update_fields:
        logger.warning("No valid fields provided for interaction update.")
        return

    try:
        await db.interactions.update_one(
            {"_id": interaction_id},
            {"$set": update_fields}
        )
    except Exception as e:
        logger.error(f"Failed to update interaction {interaction_id}: {e}")

async def get_landmarks_by_name(name_query: str) -> list[dict]:
    """Fetches landmarks using a case-insensitive regex search."""
    if db is None: return []
    try:
        # Use regex for flexible, case-insensitive matching
        # Escape special regex characters in the query
        safe_query = re.escape(name_query)
        regex = re.compile(safe_query, re.IGNORECASE)

        cursor = db.landmarks.find(
            {"landmark_name": {"$regex": regex}},
            # Projection: Specify fields to return (_id is included by default)
            {"_id": 0, "landmark_name": 1, "ulbname": 1, "latitude": 1, "longitude": 1}
        ).limit(5) # Limit results

        results = await cursor.to_list(length=5) # Convert cursor to list
        return results
    except Exception as e:
        logger.error(f"Error querying landmarks: {e}")
        return []

def get_db():
    """Dependency function to get DB instance (useful if needed in endpoints)."""
    if db is None:
         raise Exception("Database not initialized")
    return db