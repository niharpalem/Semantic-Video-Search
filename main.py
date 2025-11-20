import json
import os
import re
import tempfile
import streamlit as st
from streamlit_lottie import st_lottie
from styles import get_app_css, get_header_html
from utils import (
    load_lottiefile,
    load_vision_model,
    load_embedding_model,
    download_youtube_video,
    process_video,
    extract_video_clip,
    search_by_embedding,
    search_by_text_flexible,
)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Video Frame Analyzer",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for minimal B&W design
    st.markdown(get_app_css(), unsafe_allow_html=True)

    # Header with title on black bar
    st.markdown(get_header_html(), unsafe_allow_html=True)

    # Lottie animation below
    lottie_animation = load_lottiefile("Animation.json")
    if lottie_animation:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st_lottie(lottie_animation, height=180, key="header_animation")

    # Initialize session state
    if 'video_data' not in st.session_state:
        st.session_state.video_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'clip_chat_history' not in st.session_state:
        st.session_state.clip_chat_history = []
    if 'current_clip_time' not in st.session_state:
        st.session_state.current_clip_time = None
    if 'video_file_path' not in st.session_state:
        st.session_state.video_file_path = None

    # Create tabs - ONLY 2 TABS
    tab1, tab2 = st.tabs(["📹 Analyze Video", "🔍 Search & Clips"])

    # =========================================================================
    # TAB 1: VIDEO ANALYSIS
    # =========================================================================
    with tab1:
        col1, col2 = st.columns([3, 1])

        with col1:
            input_method = st.radio(
                "Choose input method:",
                ["📁 Upload File", "🔗 YouTube URL"],
                horizontal=True
            )

        with col2:
            smart_filtering = st.checkbox("🧠 Smart Filtering", value=True)

        video_path = None
        video_name = None

        if input_method == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Drag and drop file here",
                type=['mov', 'mp4', 'avi', 'mkv', 'mpeg4'],
                help="Limit 200MB per file • MOV, MP4, AVI, MKV, MPEG4"
            )

            if uploaded_file is not None:
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(uploaded_file)
                st.markdown('</div>', unsafe_allow_html=True)
                video_name = uploaded_file.name

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            video_path = tmp_file.name
                            st.session_state.video_file_path = video_path

        else:  # YouTube URL
            youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")

            if youtube_url:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Download & Analyze", type="primary", use_container_width=True):
                        with st.spinner("⏳ Downloading from YouTube..."):
                            video_path, video_name = download_youtube_video(youtube_url)

                        if video_path:
                            st.success(f"✓ {video_name}")
                            st.markdown('<div class="video-container">', unsafe_allow_html=True)
                            st.video(video_path)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.session_state.video_file_path = video_path

        # Process video
        if video_path:
            st.markdown("---")

            with st.spinner("🔄 Loading AI models..."):
                vision_model, processor = load_vision_model()
                embedding_model = load_embedding_model()

            results = process_video(video_path, vision_model, processor, embedding_model, smart_filtering)

            if results:
                output_dir = "video_frames_analysis"
                os.makedirs(output_dir, exist_ok=True)

                output_data = {
                    "video_name": video_name,
                    "video_path": video_path,
                    "total_frames": len(results),
                    "smart_filtering_enabled": smart_filtering,
                    "frames": results
                }

                st.session_state.video_data = output_data

                json_filename = os.path.join(output_dir, "video_analysis.json")
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)

                st.success("✅ Analysis Complete!")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Frames", len(results))
                with col2:
                    st.metric("⏱️ Duration", f"{results[-1]['timestamp_seconds']:.0f}s")
                with col3:
                    st.metric("🎯 Quality", "Filtered" if smart_filtering else "All")
                with col4:
                    st.metric("💾 Size", f"{len(results) * 0.5:.1f}KB")

                # Frame preview
                st.markdown("### 📊 Frame Analysis")

                with st.expander(f"View all {len(results)} frames", expanded=False):
                    for frame in results:
                        st.markdown(f"""
                        <div class="frame-item">
                            <b>Frame {frame['frame_number']}</b>
                            <span style="color: #888;">• {frame['timestamp_formatted']}</span>
                            <br>
                            <span style="font-size: 0.9rem; color: #555;">{frame['description']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Download button
                json_str = json.dumps(output_data, indent=4)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Analysis (JSON)",
                        data=json_str,
                        file_name="video_analysis.json",
                        mime="application/json",
                        use_container_width=True
                    )

                st.info("✨ Navigate to **Search & Clips** tab to search and extract video clips!")

    # =========================================================================
    # TAB 2: SEARCH & CLIPS
    # =========================================================================
    with tab2:
        # Load existing analysis
        with st.expander("📂 Load Previous Analysis"):
            st.markdown('<p style="color: #000000; margin-bottom: 8px;">Upload JSON</p>', unsafe_allow_html=True)
            uploaded_json = st.file_uploader("Upload JSON", type=['json'], key="json_uploader_search", label_visibility="collapsed")
            if uploaded_json:
                loaded_data = json.load(uploaded_json)
                st.session_state.video_data = loaded_data
                st.markdown(f'<p style="color: #000000; background-color: #d4edda; padding: 10px; border-radius: 5px;">✓ Loaded: {loaded_data["video_name"]}</p>', unsafe_allow_html=True)

        if st.session_state.video_data is None:
            st.info("👆 Please analyze a video in the **Analyze Video** tab or load an existing JSON file above")
        else:
            st.markdown(f"**📹 {st.session_state.video_data['video_name']}** • {st.session_state.video_data['total_frames']} frames")

            # Load models
            if 'embedding_model' not in st.session_state:
                with st.spinner("Loading search models..."):
                    st.session_state.embedding_model = load_embedding_model()

            st.markdown("---")

            # Search interface
            col1, col2, col3 = st.columns([4, 2, 1])

            with col1:
                query = st.text_input("Search Query", placeholder="🔍 Search your video...", label_visibility="collapsed")

            with col2:
                search_method = st.selectbox("Search Method", ["Semantic"], label_visibility="collapsed")

            with col3:
                search_button = st.button("Search", type="primary", use_container_width=True)

            # Settings
            with st.expander("⚙️ Search Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    padding_seconds = st.slider("Clip Padding (seconds)", 0.0, 10.0, 3.0, 0.5)
                with col2:
                    top_k = st.slider("Number of Results", 1, 10, 3)

            if query and search_button:
                # SEMANTIC SEARCH
                if search_method == "Semantic":
                    results = search_by_embedding(
                        query,
                        st.session_state.video_data,
                        st.session_state.embedding_model,
                        top_k=top_k
                    )

                    st.markdown(f'<h3 style="color: #000000;">📊 Top {len(results)} Results</h3>', unsafe_allow_html=True)

                    for i, result in enumerate(results, 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Result {i}** • Frame {result['frame_number']} • {result['timestamp_formatted']}")
                            with col2:
                                st.markdown(f"**Similarity:** {result['similarity']:.2%}")

                            with st.expander("📝 Description"):
                                st.write(result['description'])

                            # Extract clip
                            start_time = max(0, result['timestamp_seconds'] - padding_seconds)
                            end_time = result['timestamp_seconds'] + padding_seconds

                            if st.session_state.video_file_path and os.path.exists(st.session_state.video_file_path):
                                clips_dir = "extracted_clips"
                                os.makedirs(clips_dir, exist_ok=True)

                                clip_filename = f"clip_{i}_frame{result['frame_number']}.mp4"
                                clip_path = os.path.join(clips_dir, clip_filename)

                                with st.spinner("Extracting clip..."):
                                    success = extract_video_clip(
                                        st.session_state.video_file_path,
                                        start_time,
                                        end_time,
                                        clip_path
                                    )

                                if success:
                                    col1, col2, col3 = st.columns([1, 3, 1])
                                    with col2:
                                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                                        st.video(clip_path)
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    st.download_button(
                                        f"📥 Download Clip {i}",
                                        open(clip_path, 'rb'),
                                        file_name=clip_filename,
                                        mime="video/mp4",
                                        key=f"download_embed_{i}",
                                        use_container_width=True
                                    )

                            st.markdown("---")

                # TEXT SEARCH
                else:
                    response = search_by_text_flexible(
                        query,
                        st.session_state.video_data,
                        st.session_state.text_model,
                        st.session_state.text_tokenizer
                    )

                    st.markdown("### 📊 Search Results")
                    st.info(response)

                    # Show clips for mentioned frames
                    frame_matches = re.findall(r'[Ff]rame\s+(\d+)', response)

                    if frame_matches:
                        st.markdown("### 🎬 Relevant Clips")

                        frames = st.session_state.video_data['frames']
                        for frame_num_str in set(frame_matches)[:3]:
                            frame_num = int(frame_num_str)

                            matching_frame = next((f for f in frames if f['frame_number'] == frame_num), None)

                            if matching_frame and st.session_state.video_file_path:
                                st.markdown(f"**Frame {frame_num}** • {matching_frame['timestamp_formatted']}")

                                start_time = max(0, matching_frame['timestamp_seconds'] - padding_seconds)
                                end_time = matching_frame['timestamp_seconds'] + padding_seconds

                                clips_dir = "extracted_clips"
                                os.makedirs(clips_dir, exist_ok=True)

                                clip_filename = f"clip_frame{frame_num}.mp4"
                                clip_path = os.path.join(clips_dir, clip_filename)

                                success = extract_video_clip(
                                    st.session_state.video_file_path,
                                    start_time,
                                    end_time,
                                    clip_path
                                )

                                if success:
                                    col1, col2, col3 = st.columns([1, 3, 1])
                                    with col2:
                                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                                        st.video(clip_path)
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    st.download_button(
                                        f"📥 Download",
                                        open(clip_path, 'rb'),
                                        file_name=clip_filename,
                                        mime="video/mp4",
                                        key=f"download_text_{frame_num}",
                                        use_container_width=True
                                    )

                                st.markdown("---")


if __name__ == "__main__":
    main()
