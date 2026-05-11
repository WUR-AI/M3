"""Streamlit page for the metadata generation demo.

The page owns only UI concerns: collecting user inputs, showing a data preview,
triggering the workflow, caching results in Streamlit session state, and
rendering the workflow output.
"""

import sys
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.workflows.metadata_generation import (
    SUPPORTED_FILE_TYPES,
    available_metadata_standards,
    execution_details,
    extract_metadata,
    generate_metadata,
    load_preview,
    uploaded_file_key,
)


def render_controls() -> tuple[UploadedFile | None, str]:
    """Render upload and standard selection controls.

    Returns:
        The uploaded file, if present, and the selected metadata standard name.
    """
    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=SUPPORTED_FILE_TYPES,
        help="CSV and TSV files are supported by the demo context.",
    )
    standard_name = st.selectbox(
        "Metadata standard",
        options=available_metadata_standards(),
        index=0,
    )
    return uploaded_file, standard_name


def render_preview(uploaded_file: UploadedFile, file_bytes: bytes) -> None:
    """Render a preview table for the uploaded dataset.

    Args:
        uploaded_file: Streamlit uploaded file object.
        file_bytes: Raw uploaded file bytes.
    """
    st.subheader("Data preview")
    try:
        st.dataframe(
            load_preview(uploaded_file.name, file_bytes),
            use_container_width=True,
        )
        st.caption(f"{uploaded_file.name} - {len(file_bytes):,} bytes")
    except Exception as exc:
        st.warning(f"Preview unavailable: {exc}")


def run_generation(
    uploaded_file: UploadedFile,
    file_bytes: bytes,
    standard_name: str,
    file_key: str,
) -> dict[str, Any]:
    """Run the metadata workflow and cache the result in session state.

    Args:
        uploaded_file: Streamlit uploaded file object.
        file_bytes: Raw uploaded file bytes.
        standard_name: Selected metadata standard name.
        file_key: Stable cache key for the uploaded file and standard.

    Returns:
        JSON-friendly metadata generation result.
    """
    progress = st.status("Generating metadata", expanded=True)

    def show_progress(stage: str, payload: Any | None) -> None:
        if stage == "context_created":
            progress.write("Execution context created.")
            with progress:
                with st.expander("Context"):
                    st.json(payload)
        elif stage == "plan_generated":
            progress.write("Execution plan generated.")
            with progress:
                with st.expander("Generated plan"):
                    st.json(payload)
        elif stage == "execution_complete":
            progress.write("Plan execution complete.")

    try:
        progress.write("File uploaded and staged for analysis.")
        result = generate_metadata(
            uploaded_file.name,
            file_bytes,
            standard_name,
            progress_callback=show_progress,
        )
        st.session_state.metadata_generation_results[file_key] = result
        progress.update(label="Metadata generated", state="complete")
        return result
    except Exception as exc:
        progress.update(label="Metadata generation failed", state="error")
        st.exception(exc)
        st.stop()


def render_result(result: dict[str, Any]) -> None:
    """Render generated metadata and compact execution details.

    Args:
        result: JSON-friendly metadata generation result.
    """
    metadata = extract_metadata(result)

    st.subheader("Generated metadata")
    if metadata:
        st.json(metadata)
    else:
        st.warning("No final metadata artifact was found in the result.")

    plan = result.get("generated_plan") or []
    if plan:
        with st.expander("Generated plan", expanded=True):
            st.json(plan)

    step_results = result.get("step_results") or []
    if step_results:
        st.subheader("Execution steps")
        st.dataframe(
            [
                {
                    "step": step.get("step_index", 0) + 1,
                    "task": step.get("task"),
                    "player": step.get("player_role"),
                    "success": step.get("success"),
                    "artifacts": ", ".join((step.get("artifacts") or {}).keys()),
                    "error": step.get("error"),
                }
                for step in step_results
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Execution details"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Plan steps", result.get("plan_steps_count", 0))
        col2.metric("Completed", result.get("steps_completed", 0))
        col3.metric("Success", "Yes" if result.get("success") else "No")

        if result.get("error"):
            st.error(result["error"])

        st.json(execution_details(result))


def main() -> None:
    """Render the metadata generation Streamlit page."""
    st.set_page_config(page_title="Metadata Generation", page_icon="MD", layout="wide")
    st.title("Metadata Generation")

    controls_col, preview_col = st.columns([1, 2], gap="large")

    with controls_col:
        uploaded_file, standard_name = render_controls()

    if uploaded_file is None:
        with preview_col:
            st.info("Upload a CSV or TSV file to start.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    file_key = uploaded_file_key(file_bytes, standard_name)

    with preview_col:
        render_preview(uploaded_file, file_bytes)

    if "metadata_generation_results" not in st.session_state:
        st.session_state.metadata_generation_results = {}

    with controls_col:
        run_clicked = st.button("Generate metadata", type="primary", use_container_width=True)

    cached_result = st.session_state.metadata_generation_results.get(file_key)

    if run_clicked and cached_result is None:
        with preview_col:
            cached_result = run_generation(uploaded_file, file_bytes, standard_name, file_key)

    if cached_result is None:
        with preview_col:
            st.caption("Click Generate metadata to run the agent.")
        st.stop()

    with preview_col:
        render_result(cached_result)


main()
