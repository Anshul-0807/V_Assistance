import os
import time
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from bson import ObjectId # Import ObjectId

# Import models - ProcessResponse now expects string ID
from models import TextInput, ProcessResponse, HealthCheckResponse

# Import MongoDB functions
from database_mongo import connect_db, close_db, log_interaction_start, update_interaction, client as db_client # Use client to check connection
# Import services
from services import (
    load_models,
    transcribe_audio,
    query_llm,
    generate_speech,
    generate_lipsync,
    OUTPUT_DIR,
    whisper_model,
    tts_model,
    wav2lip_model,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup...")
    connect_db() # Connects and initializes DB sync part
    load_models()
    yield
    # Shutdown
    logger.info("Application shutdown...")
    close_db()

app = FastAPI(lifespan=lifespan, title="AI Video Assistant API (MongoDB)")

# --- Helper Function (Updated to handle ObjectId) ---
async def process_interaction_flow(
    interaction_id: ObjectId, # Expect ObjectId here
    background_tasks: BackgroundTasks,
    audio_file: UploadFile | None = None,
    input_text: str | None = None,
    session_id: str | None = None,
):
    """Handles the core STT -> LLM -> TTS -> LipSync flow using MongoDB ObjectId."""
    update_data = {}
    start_overall_time = time.time()
    audio_path = None
    transcribed_text = input_text
    stt_latency = 0
    stt_confidence = None

    try:
        # 1. STT (if audio provided)
        if audio_file:
            temp_audio_filename = f"input_{uuid.uuid4()}_{audio_file.filename}"
            audio_path = os.path.join(OUTPUT_DIR, temp_audio_filename)
            with open(audio_path, "wb") as buffer:
                buffer.write(await audio_file.read())
            logger.info(f"Audio file saved temporarily to {audio_path}")

            transcribed_text, stt_confidence, stt_latency = await transcribe_audio(audio_path)
            update_data['latency_stt_ms'] = stt_latency
            if transcribed_text is None:
                raise HTTPException(status_code=500, detail="Speech-to-Text failed")
            update_data['user_input_text'] = transcribed_text
            update_data['stt_confidence'] = stt_confidence
            logger.info(f"STT completed in {stt_latency}ms")
        elif input_text:
             update_data['user_input_text'] = input_text
             logger.info("Processing text input directly.")
        else:
            raise HTTPException(status_code=400, detail="No audio or text input provided")

        # Update DB (using background task)
        background_tasks.add_task(update_interaction, interaction_id, **update_data)

        # 2. LLM Query
        llm_response, llm_latency = await query_llm(transcribed_text, session_id)
        update_data['latency_llm_ms'] = llm_latency
        if llm_response is None:
            raise HTTPException(status_code=500, detail="Language Model query failed")
        update_data['llm_response_text'] = llm_response
        background_tasks.add_task(update_interaction, interaction_id, llm_response_text=llm_response, latency_llm_ms=llm_latency)
        logger.info(f"LLM completed in {llm_latency}ms")

        # 3. TTS Generation
        generated_audio_path, tts_latency = await generate_speech(llm_response)
        update_data['latency_tts_ms'] = tts_latency
        if generated_audio_path is None:
            raise HTTPException(status_code=500, detail="Text-to-Speech generation failed")
        update_data['generated_audio_path'] = generated_audio_path
        background_tasks.add_task(update_interaction, interaction_id, generated_audio_path=generated_audio_path, latency_tts_ms=tts_latency)
        logger.info(f"TTS completed in {tts_latency}ms")

        # 4. Lip Sync Generation
        generated_video_path, lipsync_latency = await generate_lipsync(generated_audio_path)
        update_data['latency_lipsync_ms'] = lipsync_latency
        if generated_video_path is None:
            logger.error("Lipsync generation failed, proceeding without video.")
            update_data['status'] = 'completed_audio_only'
            update_data['error_message'] = 'Lipsync failed'
        else:
            update_data['generated_video_path'] = generated_video_path
            update_data['status'] = 'completed'
            logger.info(f"LipSync completed in {lipsync_latency}ms")

        # Final update
        update_data['timestamp_end'] = 'NOW()' # Use database time (or set here)
        background_tasks.add_task(update_interaction, interaction_id, **update_data)

        total_time = time.time() - start_overall_time
        logger.info(f"Interaction {interaction_id} processed successfully in {total_time:.2f}s.")

        # Return response with stringified ObjectId
        return ProcessResponse(
            interaction_id=str(interaction_id), # Convert ObjectId to string
            status=update_data.get('status', 'completed'),
            message="Interaction processed successfully.",
            transcribed_text=transcribed_text,
            llm_response=llm_response,
            audio_output_path=generated_audio_path,
            video_output_path=update_data.get('generated_video_path'),
        )

    except Exception as e:
        logger.error(f"Error processing interaction {interaction_id}: {e}", exc_info=True)
        error_message = f"Failed processing interaction: {type(e).__name__} - {e}"
        background_tasks.add_task(
            update_interaction,
            interaction_id,
            status='failed',
            error_message=error_message
            # timestamp_end will be set in update_interaction
        )
        if isinstance(e, HTTPException):
             raise e
        else:
             raise HTTPException(status_code=500, detail=error_message)

    finally:
         if audio_path and os.path.exists(audio_path):
              try:
                  background_tasks.add_task(os.remove, audio_path)
                  logger.info(f"Scheduled cleanup for temporary file: {audio_path}")
              except Exception as cleanup_error:
                  logger.error(f"Failed to schedule cleanup for {audio_path}: {cleanup_error}")

# --- API Endpoints ---

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Checks the status of the API and its dependencies."""
    db_connected = False
    if db_client:
        try:
            # The ismaster command is cheap and does not require auth.
            await db_client.admin.command('ismaster')
            db_connected = True
            logger.info("MongoDB connection check successful.")
        except Exception as e:
            logger.error(f"MongoDB connection check failed: {e}")
            db_connected = False

    return HealthCheckResponse(
        database_connected=db_connected,
        whisper_loaded=whisper_model is not None,
        tts_loaded=tts_model is not None,
        lipsync_loaded=wav2lip_model is not None,
    )

@app.post("/process_audio", response_model=ProcessResponse)
async def process_audio_endpoint(
    background_tasks: BackgroundTasks,
    session_id: str | None = None,
    audio_file: UploadFile = File(...)
):
    """ Accepts audio, processes full pipeline (STT -> LLM -> TTS -> LipSync). """
    logger.info(f"Received audio processing request. File: {audio_file.filename}, Session: {session_id}")
    interaction_id = await log_interaction_start(session_id=session_id)
    if interaction_id is None:
        raise HTTPException(status_code=500, detail="Failed to log interaction start in database")

    # Pass ObjectId to the flow
    return await process_interaction_flow(
        interaction_id=interaction_id,
        background_tasks=background_tasks,
        audio_file=audio_file,
        session_id=session_id,
    )

@app.post("/process_text", response_model=ProcessResponse)
async def process_text_endpoint(
    payload: TextInput,
    background_tasks: BackgroundTasks,
):
    """ Accepts text, processes pipeline (LLM -> TTS -> LipSync). """
    logger.info(f"Received text processing request. Text: {payload.text[:50]}..., Session: {payload.session_id}")
    interaction_id = await log_interaction_start(session_id=payload.session_id)
    if interaction_id is None:
        raise HTTPException(status_code=500, detail="Failed to log interaction start in database")

    # Pass ObjectId to the flow
    return await process_interaction_flow(
        interaction_id=interaction_id,
        background_tasks=background_tasks,
        input_text=payload.text,
        session_id=payload.session_id,
    )

# Optional: Endpoint to serve files (add security checks if used)
# from fastapi.responses import FileResponse
# @app.get("/output/{filename}")
# async def get_output_file(filename: str):
#     # ... (same as before) ...


# --- Run the server ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)