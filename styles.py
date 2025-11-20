"""
Styles module for Semantic Video Search app.
Contains all CSS styles and HTML templates.
"""


def get_app_css():
    """Return the CSS styles for the application."""
    return """
<style>
/* Main background */
.stApp {
    background-color: #f5f5f5;
}

/* REMOVE BLACK BAR - Hide header completely */
header {
    visibility: hidden;
    height: 0;
    padding: 0;
    margin: 0;
}

.main > div {
    padding-top: 1rem;
}

/* Remove extra padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* PROGRESS BAR - Dark text on light background */
.stProgress > div > div > div > div {
    background-color: #4A90E2;
}

/* Progress text - make it visible */
.stProgress {
    background-color: #e0e0e0;
}

/* Status text during processing */
.element-container div[data-testid="stMarkdownContainer"] p {
    color: #333;
}

/* Spinner text */
.stSpinner > div {
    border-top-color: #4A90E2 !important;
}

.stSpinner > div > div {
    color: #333 !important;
}

/* Clean tabs - EQUALLY SPACED */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: transparent;
    padding: 0;
    border-bottom: none;
    justify-content: space-evenly;
}

.stTabs [data-baseweb="tab"] {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 12px 24px;
    border: 1px solid #e0e0e0;
    color: #333;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    min-width: 200px;
}

.stTabs [aria-selected="true"] {
    background-color: #4A90E2;
    color: white;
    border: 1px solid #4A90E2;
    box-shadow: 0 4px 8px rgba(74, 144, 226, 0.3);
}

/* File uploader styling */
.stFileUploader {
    background-color: #2d2d2d;
    border-radius: 12px;
    padding: 2rem;
    border: 2px dashed #555;
}

.stFileUploader label {
    color: #ffffff !important;
}

.stFileUploader [data-testid="stFileUploadDropzone"] {
    background-color: #2d2d2d;
}

.stFileUploader [data-testid="stFileUploadDropzone"] section {
    background-color: #2d2d2d;
    border: none;
}

.stFileUploader small {
    color: #999 !important;
}

/* Buttons */
.stButton button {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton button:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

/* Primary button */
.stButton button[kind="primary"] {
    background-color: #4A90E2;
    color: white;
    border: none;
}

/* Radio buttons */
.stRadio > label {
    font-weight: 500;
    color: #333;
}

.stRadio [role="radiogroup"] {
    gap: 1rem;
}

/* Input fields - BLACK TEXT */
.stTextInput input, .stNumberInput input {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
    color: #000000 !important;
    font-size: 1rem;
}

/* Placeholder text - gray */
.stTextInput input::placeholder {
    color: #999 !important;
}

/* Text area - BLACK TEXT */
.stTextArea textarea {
    color: #000000 !important;
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

.stTextArea textarea::placeholder {
    color: #999 !important;
}

/* Select box - BLACK TEXT */
.stSelectbox div[data-baseweb="select"] > div {
    color: #000000 !important;
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

.stSelectbox [data-baseweb="select"] {
    background-color: #ffffff;
}

.stSelectbox label {
    color: #333;
    font-weight: 500;
}

/* Dropdown menu items - BLACK TEXT */
[role="listbox"] [role="option"] {
    color: #000000 !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    color: #2c3e50;
    font-weight: 600;
}

[data-testid="stMetricLabel"] {
    color: #666;
    font-size: 0.9rem;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    font-weight: 500;
    color: #000000 !important;
}

.streamlit-expanderHeader p,
.streamlit-expanderHeader span,
[data-testid="stExpander"] summary {
    color: #000000 !important;
}

.streamlit-expanderContent {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-top: none;
    border-radius: 0 0 8px 8px;
}

/* Frame display */
.frame-item {
    background-color: #ffffff;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 3px solid #4A90E2;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Chat messages */
.stChatMessage {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Video container */
.video-container {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

/* Info/Success/Warning boxes */
.stAlert {
    border-radius: 8px;
    border-left: 4px solid #4A90E2;
    background-color: #ffffff;
}

/* Success message */
.element-container:has(.stSuccess) {
    color: #2d862d;
}

/* Info message */
.element-container:has(.stInfo) {
    color: #333;
}

/* Download button */
.stDownloadButton button {
    background-color: #2d2d2d;
    color: white;
    border: none;
    font-weight: 500;
}

.stDownloadButton button:hover {
    background-color: #1a1a1a;
}

/* Checkbox */
.stCheckbox {
    color: #333;
}

.stCheckbox label {
    color: #333 !important;
}

/* Slider */
.stSlider {
    color: #333;
}

[data-testid="stSlider"] label {
    color: #333;
}

/* Caption text */
.caption {
    color: #666 !important;
}
</style>
"""


def get_header_html():
    """Return the header HTML with title."""
    return """
    <div style="background-color: #1a1a1a; padding: 1.5rem 2rem; border-radius: 12px; margin: 0 auto 1rem auto; max-width: 600px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="text-align: center; color: #ffffff; margin: 0; font-size: 2rem; font-weight: 600; letter-spacing: 2px;">
            Semantic Video Search
        </h1>
    </div>
"""
