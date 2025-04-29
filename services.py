# # services.py

# import os
# import time
# import uuid
# import asyncio # <--- Import asyncio
# import logging
# from dotenv import load_dotenv
# import openai
# import torch
# # torchaudio might not be explicitly needed by Coqui TTS for simple generation,
# # but keep if Whisper or other audio processing might use it.
# # import torchaudio # Usually not needed directly if TTS manages it

# # --- Import Coqui TTS ---
# try:
#     # Rename class during import to avoid potential naming conflicts
#     from TTS.api import TTS as CoquiTTSApi
#     COQUI_AVAILABLE = True
#     logging.info("Successfully imported Coqui TTS components.")
# except ImportError as e:
#     logging.warning(f"Coqui TTS library (TTS) not found or failed to import: {e}. TTS will not work.")
#     CoquiTTSApi = None # Define as None if import fails
#     COQUI_AVAILABLE = False

# # Import the specific MongoDB functions needed
# from database_mongo import log_interaction_start, update_interaction, get_landmarks_by_name

# load_dotenv()

# # --- Configuration ---
# OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/")
# WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
# BASE_AVATAR_PATH = os.getenv("BASE_AVATAR_PATH", "./avatar/base_avatar.mp4")
# WAV2LIP_CHECKPOINT_PATH = os.getenv("WAV2LIP_CHECKPOINT_PATH") # Placeholder path
# DEVICE = os.getenv("DEVICE", "cpu") # Will be checked against torch.cuda.is_available()
# # --- Coqui specific config ---
# # Using VITS as specified in .env - good choice for quality/speed balance
# COQUI_MODEL_NAME = os.getenv("COQUI_MODEL_NAME", "tts_models/en/ljspeech/vits")

# # Ensure output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- Model Globals ---
# whisper_model = None
# tts_model = None # Will hold the Coqui TTS API instance
# wav2lip_model = None # Placeholder for Wav2Lip

# logger = logging.getLogger(__name__)

# # --- Model Loading ---
# def load_models():
#     """Load all AI models (call once at startup)."""
#     global whisper_model, tts_model, wav2lip_model
#     global DEVICE # Allow modification if CUDA check fails

#     # Determine effective device (Ensure PyTorch was installed correctly for this!)
#     if DEVICE.lower() == "cuda":
#         if not torch.cuda.is_available():
#             logger.warning("CUDA specified but torch.cuda.is_available() is False. Falling back to CPU.")
#             DEVICE = "cpu"
#         else:
#             logger.info("CUDA device is available via torch.cuda.is_available().")
#             try:
#                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
#             except Exception as gpu_err:
#                logger.warning(f"Could not get CUDA device name: {gpu_err}")
#     else:
#         logger.info("Using CPU device as configured.")

#     effective_device = DEVICE # Use the potentially updated DEVICE value
#     logger.info(f"Attempting to load models on device: {effective_device}")

#     # --- Load Whisper ---
#     try:
#         import whisper
#         logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
#         # Load Whisper model respecting the determined effective device
#         whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=effective_device)
#         logger.info(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on {effective_device}.")
#     except Exception as e:
#         logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
#         whisper_model = None

#     # --- Load Coqui TTS Model ---
#     if COQUI_AVAILABLE:
#         try:
#             if not COQUI_MODEL_NAME:
#                 raise ValueError("COQUI_MODEL_NAME is not set in environment variables.")

#             logger.info(f"Loading Coqui TTS model: {COQUI_MODEL_NAME}...")
#             # Determine if GPU should be used based on effective_device
#             use_cuda = (effective_device == 'cuda')

#             # Instantiate Coqui TTS API
#             tts_model = CoquiTTSApi(model_name=COQUI_MODEL_NAME, progress_bar=True, gpu=use_cuda)

#             logger.info(f"Coqui TTS model '{COQUI_MODEL_NAME}' loaded successfully.")
#             # Verify actual device Coqui is using (it might fallback)
#             actual_tts_device = "CPU" # Assume CPU unless proven otherwise
#             if hasattr(tts_model, 'device'):
#                  actual_tts_device = str(tts_model.device).upper()
#                  if use_cuda and 'cuda' in actual_tts_device.lower():
#                      logger.info(f"Coqui TTS confirmed running on GPU ({actual_tts_device}).")
#                  elif use_cuda and 'cuda' not in actual_tts_device.lower():
#                       logger.warning(f"Coqui TTS was requested on GPU, but seems to be running on {actual_tts_device}. Check model compatibility/logs.")
#                  else:
#                       logger.info(f"Coqui TTS confirmed running on CPU ({actual_tts_device}).")

#         except Exception as e:
#             logger.error(f"Failed during Coqui TTS model instantiation for '{COQUI_MODEL_NAME}': {e}", exc_info=True)
#             tts_model = None
#     else:
#         logger.warning("Coqui TTS library (TTS) not available. Skipping TTS model load.")
#         tts_model = None

#     # --- Load Wav2Lip (Placeholder) ---
#     # Keep the placeholder logic for Wav2Lip loading
#     try:
#         logger.info("Loading Wav2Lip model (Placeholder)...")
#         # You would load your actual Wav2Lip model here if implemented
#         wav2lip_model = "Wav2LipPlaceholder" # Simulate loaded state
#         logger.info("Wav2Lip model loaded (Placeholder).")
#     except Exception as e:
#         logger.error(f"Failed to load Wav2Lip model placeholder: {e}", exc_info=True)
#         wav2lip_model = None

#     # --- Configure OpenAI ---
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     if not openai.api_key:
#          logger.warning("OPENAI_API_KEY not found in environment variables.")
#     else:
#         # Initialize the async client here if preferred, or do it in query_llm
#         # openai.AsyncOpenAI() # You might initialize the client here if needed globally
#         pass


# # --- Service Functions ---

# async def transcribe_audio(audio_path: str) -> tuple[str | None, float | None, int]:
#     """Transcribes audio using Whisper."""
#     start_time = time.time()
#     if not whisper_model:
#         logger.error("Whisper model not loaded.")
#         return None, None, 0
#     try:
#         logger.info(f"Transcribing audio file: {audio_path}")
#         # Use the effective device determined during loading
#         effective_device = DEVICE
#         # Run transcription in a separate thread to avoid blocking if it's unexpectedly long
#         result = await asyncio.to_thread(
#             whisper_model.transcribe, audio_path, fp16=(effective_device == 'cuda')
#         )
#         text = result['text']
#         # Whisper output format may vary; check 'segments' for segment-level confidence if needed
#         # Overall confidence is not directly provided by standard transcribe method.
#         confidence = result.get('avg_logprob') # Example if available, often not
#         latency = int((time.time() - start_time) * 1000)
#         logger.info(f"Transcription result (Whisper): {text[:100]}...")
#         return text, confidence, latency
#     except Exception as e:
#         logger.error(f"Error during transcription: {e}", exc_info=True)
#         return None, None, int((time.time() - start_time) * 1000)

# async def query_llm(user_text: str, session_id: str | None = None) -> tuple[str | None, int]:
#     """Gets response from GPT-4, potentially using context/history."""
#     start_time = time.time()
#     if not openai.api_key:
#          logger.error("OpenAI API key not set.")
#          return None, 0
#     try:
#         logger.info(f"Querying LLM for text: {user_text[:100]}...")
#         landmark_context = ""
#         # Basic keyword check (same as before, can be improved with NER)
#         keywords = ["where is", "landmark", "find", "location of", "tell me about"]
#         normalized_text = user_text.lower()
#         found_keyword = False
#         potential_name = user_text # Default

#         for keyword in keywords:
#              if keyword in normalized_text:
#                  found_keyword = True
#                  parts = normalized_text.split(keyword, 1)
#                  if len(parts) > 1:
#                       potential_name = parts[1].replace("?","").strip()
#                       # Try to get the original casing back if possible, otherwise use lowercased
#                       original_parts = re.split(keyword, user_text, maxsplit=1, flags=re.IGNORECASE)
#                       if len(original_parts) > 1:
#                           potential_name = original_parts[1].replace("?","").strip()
#                       break # Take the first match

#         if found_keyword and potential_name:
#              logger.info(f"Searching landmarks for: '{potential_name}'")
#              # *** USE THE MONGODB FUNCTION ***
#              landmarks = await get_landmarks_by_name(potential_name)
#              if landmarks:
#                   landmark_context = "\n\nRelevant Landmark Information (up to 5):\n"
#                   for lm in landmarks:
#                        name = lm.get('landmark_name', 'N/A')
#                        ulb = lm.get('ulbname', 'N/A')
#                        lat = lm.get('latitude', 'N/A')
#                        lon = lm.get('longitude', 'N/A')
#                        landmark_context += f"- {name} in {ulb} (Lat: {lat}, Lon: {lon})\n"
#                   logger.info(f"Added landmark context for query.")
#              else:
#                  logger.info(f"No landmarks found matching '{potential_name}'.")
#         else:
#             logger.info("No landmark keywords detected or name extraction failed, querying LLM without specific landmark context.")


#         messages = [
#             {"role": "system", "content": "You are a helpful AI video assistant. Be concise and friendly." + landmark_context},
#             {"role": "user", "content": user_text}
#             # Add history management here if needed
#         ]

#         # Initialize client here if not done globally
#         client = openai.AsyncOpenAI()
#         response = await client.chat.completions.create(
#             model="gpt-3.5-turbo", # Consider "gpt-3.5-turbo" for faster/cheaper responses if sufficient
#             messages=messages,
#             max_tokens=150
#         )
#         llm_response = response.choices[0].message.content.strip()
#         latency = int((time.time() - start_time) * 1000)
#         logger.info(f"LLM response received: {llm_response[:100]}...")
#         return llm_response, latency

#     except Exception as e:
#         logger.error(f"Error querying LLM: {e}", exc_info=True)
#         # Check for specific OpenAI errors like authentication or rate limits
#         if isinstance(e, openai.AuthenticationError):
#             logger.error("OpenAI Authentication Error: Check your API key.")
#         elif isinstance(e, openai.RateLimitError):
#             logger.error("OpenAI Rate Limit Error: Please check your usage limits.")
#         # Return None or raise a specific exception
#         return None, int((time.time() - start_time) * 1000)


# async def generate_speech(text: str) -> tuple[str | None, int]:
#     """Generates speech using the loaded Coqui TTS model asynchronously."""
#     start_time = time.time()
#     if tts_model is None:
#         logger.error("Coqui TTS model is not loaded or available. Cannot generate speech.")
#         return None, 0
#     if not text:
#          logger.warning("No text provided for TTS.")
#          return None, 0

#     try:
#         output_filename = f"tts_{uuid.uuid4()}.wav"
#         output_path = os.path.join(OUTPUT_DIR, output_filename)
#         logger.info(f"Requesting speech generation with Coqui TTS for text: '{text[:100]}...'")
#         logger.info(f"Using model: '{COQUI_MODEL_NAME}'")

#         # --- Run Coqui TTS Inference in a separate thread ---
#         # This prevents blocking the main FastAPI event loop
#         await asyncio.to_thread(
#             tts_model.tts_to_file,
#             text=text,
#             file_path=output_path,
#             # --- Optional Parameters ---
#             # speaker_wav="path/to/reference.wav", # Needed for models like XTTS
#             # language="en", # Needed for multilingual models like XTTS
#             # speaker=tts_model.speakers[0], # If using a multi-speaker model and know the index
#         )
#         # --------------------------------------------------

#         latency = int((time.time() - start_time) * 1000)

#         # Check if file was actually created (tts_to_file doesn't always raise exceptions on failure)
#         if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
#              logger.error(f"Coqui TTS call completed but output file is missing or empty: {output_path}")
#              return None, latency

#         logger.info(f"Speech generated and saved to: {output_path} in {latency}ms")
#         return output_path, latency

#     except Exception as e:
#         logger.error(f"Error during Coqui TTS speech generation thread: {e}", exc_info=True)
#         return None, int((time.time() - start_time) * 1000)


# async def generate_lipsync(audio_path: str) -> tuple[str | None, int]:
#     """Generates lip-synced video using Wav2Lip. (Placeholder Implementation)"""
#     start_time = time.time()
#     if not wav2lip_model: # Check placeholder loaded state
#         logger.error("Wav2Lip model/environment not ready (placeholder check).")
#         return None, 0
#     if not BASE_AVATAR_PATH or not os.path.exists(BASE_AVATAR_PATH):
#         logger.error(f"Base avatar video not found or path not set: {BASE_AVATAR_PATH}")
#         return None, 0
#     if not audio_path or not os.path.exists(audio_path):
#         logger.error(f"Input audio for lipsync not found or path not set: {audio_path}")
#         return None, 0

#     try:
#         output_filename = f"lipsync_{uuid.uuid4()}.mp4"
#         output_path = os.path.join(OUTPUT_DIR, output_filename)
#         logger.info(f"Generating lipsync for audio: {audio_path} (Placeholder)")

#         # --- Start Placeholder Logic ---
#         # In a real scenario, you would call your Wav2Lip inference script/function here.
#         # This would likely involve running a separate process or using asyncio.to_thread
#         # if the Wav2Lip code can be called as a Python function.
#         logger.warning("Lipsync generation is using a placeholder (copying base avatar). Replace with actual Wav2Lip call.")
#         # Simulate generation time - replace with actual call
#         # Example: await asyncio.to_thread(run_wav2lip_inference, audio_path, BASE_AVATAR_PATH, output_path, WAV2LIP_CHECKPOINT_PATH, DEVICE)
#         await asyncio.sleep(1.0 + len(open(audio_path, 'rb').read()) / 50000) # Simulate time based on audio size
#         import shutil
#         shutil.copyfile(BASE_AVATAR_PATH, output_path)
#         # --- End Placeholder Logic ---

#         latency = int((time.time() - start_time) * 1000)

#         if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
#              logger.error(f"Lipsync placeholder failed to create output file: {output_path}")
#              return None, latency

#         logger.info(f"Lipsync video generated (placeholder) and saved to: {output_path} in {latency}ms")
#         return output_path, latency
#     except Exception as e:
#         logger.error(f"Error during lipsync generation placeholder: {e}", exc_info=True)
#         return None, int((time.time() - start_time) * 1000)


# # --- Helper for regex in LLM function (optional, depends on import) ---
# import re










# ////////////////////////////////////////////////////////////////////////////////////























# services.py

import os
import time
import uuid
import asyncio
import logging
from dotenv import load_dotenv
import groq   # <--- Import Groq
import torch
# torchaudio might not be explicitly needed by Coqui TTS for simple generation,
# but keep if Whisper or other audio processing might use it.
# import torchaudio # Usually not needed directly if TTS manages it
import re # <--- Ensure re is imported for query_llm

# --- Import Coqui TTS ---
try:
    # Rename class during import to avoid potential naming conflicts
    from TTS.api import TTS as CoquiTTSApi
    COQUI_AVAILABLE = True
    logging.info("Successfully imported Coqui TTS components.")
except ImportError as e:
    logging.warning(f"Coqui TTS library (TTS) not found or failed to import: {e}. TTS will not work.")
    CoquiTTSApi = None # Define as None if import fails
    COQUI_AVAILABLE = False

# Import the specific MongoDB functions needed
from database_mongo import get_landmarks_by_name
# Removed log_interaction_start, update_interaction as they are called from main.py/database_mongo

load_dotenv()

# --- Configuration ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
BASE_AVATAR_PATH = os.getenv("BASE_AVATAR_PATH", "./avatar/base_avatar.mp4")
WAV2LIP_CHECKPOINT_PATH = os.getenv("WAV2LIP_CHECKPOINT_PATH") # Placeholder path
DEVICE = os.getenv("DEVICE", "cpu") # Will be checked against torch.cuda.is_available()
# --- Coqui specific config ---
COQUI_MODEL_NAME = os.getenv("COQUI_MODEL_NAME", "tts_models/en/ljspeech/vits")
# --- Groq Config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Specify the Groq model to use (can also be set via env var if preferred)
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Globals ---
whisper_model = None
tts_model = None # Will hold the Coqui TTS API instance
wav2lip_model = None # Placeholder for Wav2Lip

logger = logging.getLogger(__name__)

# --- Model Loading ---
def load_models():
    """Load all AI models (call once at startup)."""
    global whisper_model, tts_model, wav2lip_model
    global DEVICE # Allow modification if CUDA check fails

    # Determine effective device (Ensure PyTorch was installed correctly for this!)
    if DEVICE.lower() == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA specified but torch.cuda.is_available() is False. Falling back to CPU.")
            DEVICE = "cpu"
        else:
            logger.info("CUDA device is available via torch.cuda.is_available().")
            try:
               logger.info(f"Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            except Exception as gpu_err:
               logger.warning(f"Could not get CUDA device name: {gpu_err}")
    else:
        logger.info("Using CPU device as configured.")

    effective_device = DEVICE # Use the potentially updated DEVICE value
    logger.info(f"Attempting to load models on device: {effective_device}")

    # --- Load Whisper ---
    try:
        import whisper
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
        # Load Whisper model respecting the determined effective device
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=effective_device)
        logger.info(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on {effective_device}.")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
        whisper_model = None

    # --- Load Coqui TTS Model ---
    if COQUI_AVAILABLE:
        try:
            if not COQUI_MODEL_NAME:
                raise ValueError("COQUI_MODEL_NAME is not set in environment variables.")

            logger.info(f"Loading Coqui TTS model: {COQUI_MODEL_NAME}...")
            # Determine if GPU should be used based on effective_device
            use_cuda = (effective_device == 'cuda')

            # Instantiate Coqui TTS API
            tts_model = CoquiTTSApi(model_name=COQUI_MODEL_NAME, progress_bar=True, gpu=use_cuda)

            logger.info(f"Coqui TTS model '{COQUI_MODEL_NAME}' loaded successfully.")
            # Verify actual device Coqui is using (it might fallback)
            actual_tts_device = "CPU" # Assume CPU unless proven otherwise
            if hasattr(tts_model, 'device'):
                 actual_tts_device = str(tts_model.device).upper()
                 if use_cuda and 'cuda' in actual_tts_device.lower():
                     logger.info(f"Coqui TTS confirmed running on GPU ({actual_tts_device}).")
                 elif use_cuda and 'cuda' not in actual_tts_device.lower():
                      logger.warning(f"Coqui TTS was requested on GPU, but seems to be running on {actual_tts_device}. Check model compatibility/logs.")
                 else:
                      logger.info(f"Coqui TTS confirmed running on CPU ({actual_tts_device}).")

        except Exception as e:
            logger.error(f"Failed during Coqui TTS model instantiation for '{COQUI_MODEL_NAME}': {e}", exc_info=True)
            tts_model = None
    else:
        logger.warning("Coqui TTS library (TTS) not available. Skipping TTS model load.")
        tts_model = None

    # --- Load Wav2Lip (Placeholder) ---
    try:
        logger.info("Loading Wav2Lip model (Placeholder)...")
        # You would load your actual Wav2Lip model here if implemented
        wav2lip_model = "Wav2LipPlaceholder" # Simulate loaded state
        logger.info("Wav2Lip model loaded (Placeholder).")
    except Exception as e:
        logger.error(f"Failed to load Wav2Lip model placeholder: {e}", exc_info=True)
        wav2lip_model = None

    # --- Check for Groq API Key ---
    if not GROQ_API_KEY:
         logger.warning("GROQ_API_KEY not found in environment variables. LLM queries will fail.")
    # No need to configure the client globally here, we do it in query_llm


# --- Service Functions ---

async def transcribe_audio(audio_path: str) -> tuple[str | None, float | None, int]:
    """Transcribes audio using Whisper."""
    start_time = time.time()
    if not whisper_model:
        logger.error("Whisper model not loaded.")
        return None, None, 0
    try:
        logger.info(f"Transcribing audio file: {audio_path}")
        # Use the effective device determined during loading
        effective_device = DEVICE
        # Run transcription in a separate thread to avoid blocking if it's unexpectedly long
        result = await asyncio.to_thread(
            whisper_model.transcribe, audio_path, fp16=(effective_device == 'cuda')
        )
        text = result['text']
        # Whisper output format may vary; check 'segments' for segment-level confidence if needed
        # Overall confidence is not directly provided by standard transcribe method.
        confidence = result.get('avg_logprob') # Example if available, often not
        latency = int((time.time() - start_time) * 1000)
        logger.info(f"Transcription result (Whisper): {text[:100]}...")
        return text, confidence, latency
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None, None, int((time.time() - start_time) * 1000)

async def query_llm(user_text: str, session_id: str | None = None) -> tuple[str | None, int]:
    """Gets response from Groq Llama 3, potentially using context/history."""
    start_time = time.time()
    if not GROQ_API_KEY: # <--- Check Groq key
         logger.error("Groq API key not set.")
         return None, 0
    try:
        logger.info(f"Querying Groq LLM (model: {GROQ_MODEL_NAME}) for text: {user_text[:100]}...")
        landmark_context = ""
        # --- Landmark searching logic remains the same ---
        keywords = ["where is", "landmark", "find", "location of", "tell me about"]
        normalized_text = user_text.lower()
        found_keyword = False
        potential_name = user_text # Default name is the full user text

        for keyword in keywords:
             keyword_pattern = r'\b' + re.escape(keyword) + r'\b' # Match whole word
             match = re.search(keyword_pattern, user_text, re.IGNORECASE)
             if match:
                 found_keyword = True
                 # Extract text after the keyword
                 potential_name_part = user_text[match.end():].strip()
                 # Simple cleanup (remove question mark, leading punctuation)
                 potential_name = re.sub(r"^[?.,! ]+", "", potential_name_part)

                 if potential_name: # If something remains after the keyword
                     break # Take the first keyword match that yields a potential name
                 else: # If keyword was at the very end, reset potential name
                     potential_name = user_text # Fallback if nothing follows keyword

        if found_keyword and potential_name and potential_name != user_text: # Only search if name likely extracted
             logger.info(f"Landmark keyword '{match.group()}' detected. Searching landmarks for: '{potential_name}'")
             landmarks = await get_landmarks_by_name(potential_name)
             if landmarks:
                  landmark_context = "\n\nRelevant Landmark Information (up to 5):\n"
                  for lm in landmarks:
                       name = lm.get('landmark_name', 'N/A')
                       ulb = lm.get('ulbname', 'N/A')
                       lat = lm.get('latitude', 'N/A')
                       lon = lm.get('longitude', 'N/A')
                       landmark_context += f"- {name} in {ulb} (Lat: {lat}, Lon: {lon})\n"
                  logger.info(f"Added landmark context for query.")
             else:
                 logger.info(f"No landmarks found matching '{potential_name}'.")
        else:
            if found_keyword:
                 logger.info(f"Landmark keyword detected, but failed to extract a distinct name. Querying LLM without specific context.")
            else:
                 logger.info("No landmark keywords detected, querying LLM without specific landmark context.")
        # --- End of landmark logic ---

        messages = [
            # Llama 3 often responds better with a slightly more direct system prompt
            {"role": "system", "content": "You are a helpful AI video assistant. Provide concise and friendly answers." + landmark_context},
            {"role": "user", "content": user_text}
            # Add history management here if needed
        ]

        # --- Use Groq Client ---
        client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        response = await client.chat.completions.create(
            # Use the model name specified in config
            model=GROQ_MODEL_NAME,
            messages=messages,
            max_tokens=150, # Keep max_tokens or adjust as needed
            # Optional: Add temperature, top_p etc. if desired
            # temperature=0.7,
        )
        llm_response = response.choices[0].message.content.strip()
        # ------------------------

        latency = int((time.time() - start_time) * 1000)
        logger.info(f"Groq LLM response received: {llm_response[:100]}...")
        return llm_response, latency

    # Update Error Handling for Groq if needed (optional, generic works)
    except groq.AuthenticationError:
        logger.error("Groq Authentication Error: Check your API key.")
        return None, int((time.time() - start_time) * 1000)
    except groq.RateLimitError:
         logger.error("Groq Rate Limit Error: Please check your usage limits.")
         return None, int((time.time() - start_time) * 1000)
    except groq.APIConnectionError as e:
        logger.error(f"Groq API Connection Error: {e}", exc_info=True)
        return None, int((time.time() - start_time) * 1000)
    except groq.APIStatusError as e:
        logger.error(f"Groq API Status Error - Status: {e.status_code}, Response: {e.response}", exc_info=True)
        return None, int((time.time() - start_time) * 1000)
    except Exception as e:
        logger.error(f"Error querying Groq LLM: {type(e).__name__} - {e}", exc_info=True)
        return None, int((time.time() - start_time) * 1000)


async def generate_speech(text: str) -> tuple[str | None, int]:
    """Generates speech using the loaded Coqui TTS model asynchronously."""
    start_time = time.time()
    if tts_model is None:
        logger.error("Coqui TTS model is not loaded or available. Cannot generate speech.")
        return None, 0
    if not text:
         logger.warning("No text provided for TTS.")
         return None, 0

    try:
        output_filename = f"tts_{uuid.uuid4()}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        logger.info(f"Requesting speech generation with Coqui TTS for text: '{text[:100]}...'")
        logger.info(f"Using model: '{COQUI_MODEL_NAME}'")

        # --- Run Coqui TTS Inference in a separate thread ---
        # This prevents blocking the main FastAPI event loop
        await asyncio.to_thread(
            tts_model.tts_to_file,
            text=text,
            file_path=output_path,
            # --- Optional Parameters ---
            # speaker_wav="path/to/reference.wav", # Needed for models like XTTS
            # language="en", # Needed for multilingual models like XTTS
            # speaker=tts_model.speakers[0], # If using a multi-speaker model and know the index
        )
        # --------------------------------------------------

        latency = int((time.time() - start_time) * 1000)

        # Check if file was actually created (tts_to_file doesn't always raise exceptions on failure)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             logger.error(f"Coqui TTS call completed but output file is missing or empty: {output_path}")
             return None, latency

        logger.info(f"Speech generated and saved to: {output_path} in {latency}ms")
        return output_path, latency

    except Exception as e:
        logger.error(f"Error during Coqui TTS speech generation thread: {e}", exc_info=True)
        return None, int((time.time() - start_time) * 1000)


async def generate_lipsync(audio_path: str) -> tuple[str | None, int]:
    """Generates lip-synced video using Wav2Lip. (Placeholder Implementation)"""
    start_time = time.time()
    if not wav2lip_model: # Check placeholder loaded state
        logger.error("Wav2Lip model/environment not ready (placeholder check).")
        return None, 0
    if not BASE_AVATAR_PATH or not os.path.exists(BASE_AVATAR_PATH):
        logger.error(f"Base avatar video not found or path not set: {BASE_AVATAR_PATH}")
        return None, 0
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Input audio for lipsync not found or path not set: {audio_path}")
        return None, 0

    try:
        output_filename = f"lipsync_{uuid.uuid4()}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        logger.info(f"Generating lipsync for audio: {audio_path} (Placeholder)")

        # --- Start Placeholder Logic ---
        # In a real scenario, you would call your Wav2Lip inference script/function here.
        # This would likely involve running a separate process or using asyncio.to_thread
        # if the Wav2Lip code can be called as a Python function.
        logger.warning("Lipsync generation is using a placeholder (copying base avatar). Replace with actual Wav2Lip call.")
        # Simulate generation time - replace with actual call
        # Example: await asyncio.to_thread(run_wav2lip_inference, audio_path, BASE_AVATAR_PATH, output_path, WAV2LIP_CHECKPOINT_PATH, DEVICE)
        # Simulate time based on audio size (rough estimate)
        audio_size_mb = 0
        try:
             audio_size_mb = os.path.getsize(audio_path) / (1024*1024)
        except OSError:
             pass # Ignore if file size can't be read
        await asyncio.sleep(1.0 + audio_size_mb * 0.5) # Simulate 1s + 0.5s per MB
        import shutil
        shutil.copyfile(BASE_AVATAR_PATH, output_path)
        # --- End Placeholder Logic ---

        latency = int((time.time() - start_time) * 1000)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             logger.error(f"Lipsync placeholder failed to create output file: {output_path}")
             return None, latency

        logger.info(f"Lipsync video generated (placeholder) and saved to: {output_path} in {latency}ms")
        return output_path, latency
    except Exception as e:
        logger.error(f"Error during lipsync generation placeholder: {e}", exc_info=True)
        return None, int((time.time() - start_time) * 1000)