"""
Utils module for Semantic Video Search app.
Contains model loading and video processing functions.
"""

import cv2
import json
import os
import tempfile
import re
import torch
import numpy as np
import streamlit as st
import yt_dlp
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# ============================================================================
# LOTTIE ANIMATION LOADER
# ============================================================================

def load_lottiefile(filepath: str):
    """Load Lottie animation from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_vision_model(model_id="LiquidAI/LFM2-VL-450M"):
    """
    Load the vision-language model and processor for analyzing video frames.
    Cached to avoid reloading on every interaction.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor




@st.cache_resource
def load_embedding_model(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load the sentence transformer model for semantic search.
    Used to find frames by meaning rather than exact keywords.
    """
    embedding_model = SentenceTransformer(model_id)
    return embedding_model


# ============================================================================
# VIDEO DOWNLOAD FUNCTION
# ============================================================================

def download_youtube_video(url, output_path=None):
    """
    Download video from YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL
        output_path: Directory to save video (creates temp dir if None)

    Returns:
        tuple: (video_path, video_title) or (None, None) on error
    """
    if output_path is None:
        output_path = tempfile.mkdtemp()

    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            video_title = info.get('title', 'video')
            return video_path, video_title
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None, None


# ============================================================================
# VIDEO PROCESSING FUNCTION (WITH TEMPORAL CONTEXT + EMBEDDINGS)
# ============================================================================

def process_video(video_path, model, processor, embedding_model, smart_filtering=True):
    """
    Process video frames with temporal context AND generate embeddings.
    This enables both VLM descriptions and semantic search.

    Args:
        video_path: Path to video file
        model: Vision-language model
        processor: Model processor
        embedding_model: Sentence transformer for embeddings
        smart_filtering: Whether to skip similar frames

    Returns:
        list: Frame data with descriptions and embeddings
    """
    all_frames = []
    skipped_count = 0

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_interval = int(fps)  # Extract one frame per second

    st.info(f"📹 FPS: {fps:.2f}, extracting 1 frame per second with temporal context")
    if smart_filtering:
        st.info(f"🧠 Smart filtering enabled - VLM checks for changes")

    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    # -------------------------------------------------------------------------
    # STEP 1: Load all video frames into memory buffer
    # -------------------------------------------------------------------------
    status_text.text("📹 Loading video frames into memory...")
    all_video_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_video_frames.append(frame)

    cap.release()
    st.success(f"✓ Loaded {len(all_video_frames)} frames into memory")

    # -------------------------------------------------------------------------
    # STEP 2: Process frames with temporal context
    # -------------------------------------------------------------------------
    second_count = 0

    for i in range(0, len(all_video_frames), frame_interval):
        second_count += 1
        timestamp = i / fps

        # Update progress
        progress = i / len(all_video_frames)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {second_count} at {timestamp:.2f}s...")

        # Get temporal context frames (before, current, after)
        temporal_frames = []

        before_idx = max(0, i - 15)
        frame_rgb = cv2.cvtColor(all_video_frames[before_idx], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))

        frame_rgb = cv2.cvtColor(all_video_frames[i], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))

        after_idx = min(len(all_video_frames) - 1, i + 15)
        frame_rgb = cv2.cvtColor(all_video_frames[after_idx], cv2.COLOR_BGR2RGB)
        temporal_frames.append(Image.fromarray(frame_rgb))

        # =====================================================================
        # ALWAYS ADD FIRST 5 FRAMES
        # =====================================================================
        if len(all_frames) < 5:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Frame BEFORE:"},
                        {"type": "image", "image": temporal_frames[0]},
                        {"type": "text", "text": "Frame CURRENT (describe this one):"},
                        {"type": "image", "image": temporal_frames[1]},
                        {"type": "text", "text": "Frame AFTER:"},
                        {"type": "image", "image": temporal_frames[2]},
                        {"type": "text", "text": "In MAXIMUM 40 words, describe CURRENT frame. Focus on: actions, text/numbers visible, main subjects. Be concise."},
                    ],
                },
            ]

            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(model.device)

            input_length = inputs['input_ids'].shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.2
            )

            new_tokens = outputs[0][input_length:]
            description = processor.decode(new_tokens, skip_special_tokens=True).strip()

            # Generate embedding for this description
            embedding = embedding_model.encode(description, convert_to_numpy=True)

            # Save frame data WITH embedding
            all_frames.append({
                "frame_number": second_count,
                "timestamp_seconds": round(timestamp, 2),
                "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                "description": description,
                "embedding": embedding.tolist()  # Convert numpy to list for JSON
            })

            status_text.text(f"✓ Frame {second_count} added (building baseline)")

        # =====================================================================
        # FROM FRAME 6 ONWARDS: Smart filtering
        # =====================================================================
        else:
            if smart_filtering:
                previous_context = ""
                for prev_frame in all_frames[-5:]:
                    previous_context += f"F{prev_frame['frame_number']}: {prev_frame['description'][:60]}...\n"

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Previous 5 frames:\n{previous_context}\n\nNow look at these 3 frames:"},
                            {"type": "text", "text": "BEFORE:"},
                            {"type": "image", "image": temporal_frames[0]},
                            {"type": "text", "text": "CURRENT:"},
                            {"type": "image", "image": temporal_frames[1]},
                            {"type": "text", "text": "AFTER:"},
                            {"type": "image", "image": temporal_frames[2]},
                            {"type": "text", "text": "Is CURRENT frame DIFFERENT from previous? Answer ONLY 'YES' or 'NO'."},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(model.device)

                input_length = inputs['input_ids'].shape[1]

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.2
                )

                new_tokens = outputs[0][input_length:]
                response = processor.decode(new_tokens, skip_special_tokens=True).strip()

                response_upper = response.upper()
                is_different = "YES" in response_upper[:100] or "NEW" in response_upper[:100] or "DIFFERENT" in response_upper[:100]

                if is_different:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Frame BEFORE:"},
                                {"type": "image", "image": temporal_frames[0]},
                                {"type": "text", "text": "Frame CURRENT (describe this one):"},
                                {"type": "image", "image": temporal_frames[1]},
                                {"type": "text", "text": "Frame AFTER:"},
                                {"type": "image", "image": temporal_frames[2]},
                                {"type": "text", "text": "In MAXIMUM 40 words, describe CURRENT frame. Focus on: actions, text/numbers visible, main subjects. Be concise."},
                            ],
                        },
                    ]

                    inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        tokenize=True,
                    ).to(model.device)

                    input_length = inputs['input_ids'].shape[1]

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.2
                    )

                    new_tokens = outputs[0][input_length:]
                    description = processor.decode(new_tokens, skip_special_tokens=True).strip()

                    # Generate embedding
                    embedding = embedding_model.encode(description, convert_to_numpy=True)

                    all_frames.append({
                        "frame_number": second_count,
                        "timestamp_seconds": round(timestamp, 2),
                        "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                        "description": description,
                        "embedding": embedding.tolist()
                    })

                    status_text.text(f"✓ Frame {second_count} added (NEW content detected)")
                else:
                    skipped_count += 1
                    status_text.text(f"⏭️ Frame {second_count} skipped (similar to previous) - {skipped_count} skipped")

            else:
                # Smart filtering disabled - add all frames
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Frame BEFORE:"},
                            {"type": "image", "image": temporal_frames[0]},
                            {"type": "text", "text": "Frame CURRENT (describe this one):"},
                            {"type": "image", "image": temporal_frames[1]},
                            {"type": "text", "text": "Frame AFTER:"},
                            {"type": "image", "image": temporal_frames[2]},
                            {"type": "text", "text": "In MAXIMUM 40 words, describe CURRENT frame. Focus on: actions, text/numbers, main subjects. Be concise."},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(model.device)

                input_length = inputs['input_ids'].shape[1]

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.2
                )

                new_tokens = outputs[0][input_length:]
                description = processor.decode(new_tokens, skip_special_tokens=True).strip()

                # Generate embedding
                embedding = embedding_model.encode(description, convert_to_numpy=True)

                all_frames.append({
                    "frame_number": second_count,
                    "timestamp_seconds": round(timestamp, 2),
                    "timestamp_formatted": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                    "description": description,
                    "embedding": embedding.tolist()
                })

    progress_bar.progress(1.0)
    status_text.text(f"✓ Complete! Saved {len(all_frames)} frames, skipped {skipped_count} similar frames")

    return all_frames


# ============================================================================
# VIDEO CLIP EXTRACTION FUNCTION
# ============================================================================

def extract_video_clip(video_path, start_time, end_time, output_path):
    """
    Extract a video clip from start_time to end_time.

    Args:
        video_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Where to save the clip

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure even dimensions
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        # Use H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calculate frame numbers
        start_frame = max(0, int(start_time * fps))
        end_frame = int(end_time * fps)

        # Set position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        st.error(f"Error extracting clip: {e}")
        return False


# ============================================================================
# COSINE SIMILARITY FOR EMBEDDINGS
# ============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ============================================================================
# EMBEDDING-BASED SEARCH (SEMANTIC SEARCH)
# ============================================================================

def search_by_embedding(query, video_data, embedding_model, top_k=3):
    """
    Search frames using semantic similarity (embeddings).
    This finds frames by MEANING, not exact keywords.

    Args:
        query: User's search query
        video_data: Video frame data with embeddings
        embedding_model: Sentence transformer model
        top_k: Number of top results to return

    Returns:
        list: Top matching frames with similarity scores
    """
    frames = video_data['frames']

    # Encode the query
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)

    # Compute similarities with all frames
    results = []
    for frame in frames:
        frame_embedding = np.array(frame['embedding'])
        similarity = cosine_similarity(query_embedding, frame_embedding)
        results.append({
            'frame_number': frame['frame_number'],
            'timestamp_seconds': frame['timestamp_seconds'],
            'timestamp_formatted': frame['timestamp_formatted'],
            'description': frame['description'],
            'similarity': float(similarity)
        })

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    return results[:top_k]


# ============================================================================
# TEXT-BASED SEARCH WITH TIME RANGES (FLEXIBLE)
# ============================================================================

def search_by_text_flexible(user_question, video_data, text_model, tokenizer):
    """
    Search using text model with FLEXIBLE time ranges.
    Instead of exact frame, returns a time range where answer likely is.

    Args:
        user_question: User's search query
        video_data: Video frame data
        text_model: Text generation model
        tokenizer: Model tokenizer

    Returns:
        str: Response with time ranges and frame suggestions
    """
    frames = video_data['frames']

    # Build context from ALL frames (we need full picture for ranges)
    context = "AVAILABLE FRAMES:\n\n"
    for frame in frames:
        context += f"Frame {frame['frame_number']} at {frame['timestamp_formatted']} ({frame['timestamp_seconds']}s): {frame['description'][:80]}...\n"

    # Ultra-strict prompt for time range responses
    search_prompt = f"""{context}

USER QUESTION: {user_question}

INSTRUCTIONS:
1. Find frames that answer the question
2. Give a TIME RANGE (not just one frame)
3. Format: "Between [time1] and [time2]" or "Around [time] (frames X-Y)"
4. If multiple possible answers, list all time ranges
5. ONLY reference frames listed above
6. If NOT FOUND, say "NOT FOUND"

Answer:"""

    # Generate with VERY LOW temperature for accuracy
    input_ids = tokenizer.encode(search_prompt, return_tensors="pt", truncation=True, max_length=2048).to(text_model.device)

    output = text_model.generate(
        input_ids,
        do_sample=True,
        temperature=0.05,  # Almost deterministic
        top_p=0.9,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Validate: Check if mentioned frame numbers actually exist
    mentioned_frames = re.findall(r'[Ff]rame\s+(\d+)', response)
    valid = True
    max_frame = max([f['frame_number'] for f in frames])

    for frame_num in mentioned_frames:
        if int(frame_num) > max_frame:
            valid = False
            break

    if not valid or "NOT FOUND" in response.upper():
        return "I couldn't find frames matching your query in the available video data."

    return response


# ============================================================================
# CHAT FUNCTION - 10-SECOND CLIP QA
# ============================================================================

def chat_with_video_clip(user_question, video_data, text_model, tokenizer, start_time, end_time, conversation_history=""):
    """
    Simple QA on a 10-second video clip.
    Detects if user wants facts OR generation, and adjusts accordingly.
    """

    # Get frames within the time window
    frames = video_data['frames']
    clip_frames = [f for f in frames if start_time <= f['timestamp_seconds'] <= end_time]

    if not clip_frames:
        return "No frames found in the selected time window."

    # =========================================================================
    # DETECT: Is this a GENERATION request or FACTUAL question?
    # =========================================================================
    question_lower = user_question.lower()

    generation_keywords = [
        'generate', 'create', 'make', 'similar', 'practice', 'flashcard',
        'quiz', 'questions', 'problems', 'variations', 'examples', 'like this'
    ]

    is_generation = any(keyword in question_lower for keyword in generation_keywords)

    # =========================================================================
    # GENERATION MODE - Minimal context, high temperature
    # =========================================================================
    if is_generation:
        # Only give SUMMARY, not detailed descriptions
        summary = f"VIDEO CLIP ({start_time:.1f}s - {end_time:.1f}s) SUMMARY:\n"
        summary += f"This clip shows: {clip_frames[0]['description'][:120]}...\n"
        if len(clip_frames) > 1:
            summary += f"It continues with: {clip_frames[-1]['description'][:120]}...\n"

        prompt = f"""{summary}

USER REQUEST: {user_question}

CRITICAL INSTRUCTIONS FOR CONTENT GENERATION:

1. ❌ DO NOT repeat or copy frame descriptions
2. ❌ DO NOT say "Frame X at..."
3. ❌ DO NOT include timestamps
4. ✅ CREATE completely NEW content inspired by the topic
5. ✅ Generate ORIGINAL questions/problems/flashcards
6. ✅ Keep the same style/difficulty as the video topic

EXAMPLE:
Bad: "Frame 1 shows 4x + 4x + 4 = 192..."
Good: "1. If 3x + 3x + 3 = 99, what is x?"

Now generate NEW content:"""

        temperature = 0.9  # HIGH for creativity
        max_tokens = 350

    # =========================================================================
    # FACTUAL MODE - Full context, low temperature
    # =========================================================================
    else:
        # Give full frame descriptions for factual questions
        context = f"VIDEO CLIP CONTENT ({start_time:.1f}s - {end_time:.1f}s):\n\n"
        for frame in clip_frames:
            context += f"Frame {frame['frame_number']} at {frame['timestamp_formatted']} ({frame['timestamp_seconds']}s):\n{frame['description']}\n\n"

        if len(conversation_history) == 0:
            prompt = f"""{context}

USER QUESTION: {user_question}

RULES:
1. Answer ONLY using frame descriptions above
2. If not shown, say "I don't see that in this clip"
3. Cite frame numbers when helpful
4. Be concise and accurate

Answer:"""
            temperature = 0.1  # LOW for accuracy
        else:
            prompt = f"""{context}

PREVIOUS CONVERSATION:
{conversation_history}

USER: {user_question}

Answer:"""
            temperature = 0.15

        max_tokens = 250

    # =========================================================================
    # GENERATE RESPONSE
    # =========================================================================
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1800).to(text_model.device)

    output = text_model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=50,  # Add diversity
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Penalize repetition
    )

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    # =========================================================================
    # POST-PROCESSING: Remove frame references if generation mode
    # =========================================================================
    if is_generation:
        # Clean up if model still mentions frames
        response = re.sub(r'Frame \d+ at \d+:\d+.*?:', '', response)
        response = re.sub(r'Frame \d+:', '', response)
        response = response.replace('(0.0s)', '').replace('(0.97s)', '')
        response = '\n'.join([line for line in response.split('\n') if 'timestamp' not in line.lower()])

    return response.strip()
