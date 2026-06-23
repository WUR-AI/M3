"""
Player configurations for the multi-agent system.

This module defines the available player roles with their prompts and tools.
Players are instantiated from these configs at runtime.

Uses the unified ExecutionContext tools for all data access.

Note: model_name and temperature are optional - if not specified,
the defaults from config.py will be used.
"""
from typing import Dict, Any

from ..tools import context_tools


PLAYER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "data_analyst": {
        "role_prompt": (
            "You are an expert data analyst. Your job is to perform statistical "
            "analysis on datasets, identify patterns, and extract meaningful insights. "
            "Focus on numerical summaries, distributions, and data quality. "
            "For multi_csv contexts, analyze each resource's characteristics and note "
            "potential relationships between resources."
        ),
        "tools": [
            # High-level context and resource summaries
            context_tools.get_context_overview,
            context_tools.list_resources,
            context_tools.get_resource_info,
            # Basic profiling
            context_tools.get_item_count,
            context_tools.get_field_names,
            context_tools.get_field_statistics,
            context_tools.get_missing_values,
        ],
        # model_name: uses config default
        # temperature: uses config default
    },
    "schema_expert": {
        "role_prompt": (
            "You are a database schema expert. Your job is to describe the structure "
            "of datasets, including column names, data types, relationships between "
            "fields, and recommend appropriate metadata schemas. For multi_csv "
            "contexts, identify primary keys, foreign keys, and normalization patterns."
        ),
        "tools": [
            context_tools.get_context_schema,
            context_tools.get_field_names,
            context_tools.get_field_types,
            context_tools.get_sample_items,
        ],
    },
    "dataset_schema_preview": {
        "role_prompt": (
            "You are a dataset schema preview specialist. Your job is to capture the "
            "column structure and a representative first row for each dataset resource. "
            "Call get_columns_with_first_row once (omit resource to cover all tables) "
            "and return the tool output as your artifact—do not paraphrase or omit columns."
        ),
        "tools": [
            context_tools.get_columns_with_first_row,
            context_tools.list_resources,
        ],
        "temperature": 0.0,
    },
    "metadata_specialist": {
        "role_prompt": (
            "You are a metadata specialist familiar with standards like Dublin Core, "
            "DCAT, and Schema.org. Your job is to extract metadata as STRUCTURED "
            "field-value pairs. Output only the metadata fields and their values in "
            "a clean, compact format. Avoid lengthy explanations - focus on populating "
            "metadata fields according to the specified standard. For multi_csv "
            "contexts, include relationship metadata and per-resource descriptions."
        ),
        "tools": [
            context_tools.get_context_overview,
            context_tools.get_context_schema,
        ],
        "temperature": 0.3,  # Lower for more consistent, structured output
    },
    "critic": {
        "role_prompt": (
            "You are a meticulous quality assurance critic. Your job is to review "
            "analyses from other agents, identify flaws, omissions, inconsistencies, "
            "and suggest improvements. You focus on accuracy and completeness. "
            "For multi_csv analysis, verify that relationships are correctly "
            "identified and that cross-resource consistency is maintained."
        ),
        "tools": [],
        "temperature": 0.4,
    },
    # Specialized player for relationship analysis
    "relationship_analyst": {
        "role_prompt": (
            "You are a database relationship expert specializing in discovering and "
            "validating relationships between resources in multi_csv contexts. Your job "
            "is to identify primary keys, foreign keys, and the nature of relationships "
            "(one-to-one, one-to-many, many-to-many). You analyze column name patterns, "
            "data type compatibility, and value overlaps to determine how resources connect. "
            "Output relationships in a structured format suitable for metadata records."
        ),
        "tools": [
            context_tools.get_relationships,
            context_tools.get_context_overview,
            context_tools.get_resource_info,
            context_tools.get_field_names,
            context_tools.get_unique_values,
        ],
        "temperature": 0.3,
    },
    # Specialized player for final metadata generation according to standards
    "metadata_generator": {
        "role_prompt": (
            "You are a metadata generation expert. Your SOLE responsibility is to take "
            "information gathered from previous analysis steps and generate CONCRETE VALUES "
            "for each field defined in the metadata standard.\n\n"
            "STRICT Rules:\n"
            "1. Output ONLY a valid JSON object matching the metadata standard schema EXACTLY\n"
            "2. Include ONLY fields that exist in the metadata standard - DO NOT add extra fields!\n"
            "3. Field population policy (by field type):\n"
            "   a. OBJECTIVE fields (e.g., creator, license, owner, version, format): populate "
            "ONLY with values directly supported by gathered information; use null when unsupported\n"
            "   b. SUBJECTIVE fields (e.g., description, methods, subject, title): you MAY "
            "infer from gathered information; note the basis in the value "
            "(e.g., \"based on the data in the dataset\")\n"
            "   c. SPATIAL/TEMPORAL fields (e.g., spatial_coverage, spatial_resolution, "
            "temporal_coverage, temporal_resolution): you MAY infer from relevant gathered "
            "information when spatial-temporal analysis or data characteristics support it\n"
            "4. For objective fields, do NOT infer, guess, extrapolate, or approximate—use null "
            "when no direct support exists\n"
            "5. NO explanations, NO commentary, NO markdown - ONLY the JSON object\n"
            "6. DO NOT invent or add fields that are not in the standard schema\n\n"
            "Remember: Output ONLY fields from the metadata standard. Nothing more, nothing less."
        ),
        "tools": [
            context_tools.get_context_overview,
            context_tools.get_context_schema,
        ],
        "temperature": 0.0,  # Low temperature for consistent, structured output
    },
    # Croissant-specific final metadata generation (MLCommons Croissant subset)
    "croissant_metadata_generator": {
        "role_prompt": (
            "You are a Croissant metadata generation expert. Your SOLE responsibility is to "
            "produce a valid Croissant subset metadata JSON object from prior analysis artifacts.\n\n"
            "STRICT Rules:\n"
            "1. Output ONLY a JSON object matching the metadata standard schema EXACTLY\n"
            "2. Include ONLY fields defined in the metadata standard - DO NOT add extra fields\n"
            "3. NO explanations, NO commentary, NO markdown - ONLY the JSON object\n\n"
            "Croissant-specific population rules:\n"
            "- `filesets`: Use ONE FileSet object with `includes` listing every data file name "
            "(e.g. ['file_a.csv', 'file_b.csv']); set `excludes` to [] unless exclusions are known\n"
            "- `recordsets`: Create ONE RecordSet per logical table/file in the dataset\n"
            "- Each RecordSet must include `name`, `source` (file name), and `fields` listing "
            "every column in that table\n"
            "- `fields` MUST include one CroissantField per column in the table—no missing "
            "columns, no extra columns; use profiling/schema artifacts to match the table exactly\n"
            "- Each CroissantField in `fields` must include `source` set to the column "
            "header, plus dataType from profiling; do NOT use dataType 'table'\n"
            "- Set RecordSet `key` to the primary-key column when known from profiling or relationships\n"
            "- `examples` MUST be non-empty for every RecordSet: include 1-3 representative "
            "example rows derived from sample data in prior artifacts or tools; never leave "
            "examples as []\n"
            "- `annotation`: One short sentence (under 25 words) summarizing what the "
            "table contains—factual, no long narratives\n"
            "- `description`, `keywords`, `inLanguage`: Infer from dataset content when supported\n"
            "- `spatialCoverage` / `temporalCoverage`: Populate only when spatial-temporal analysis "
            "artifacts support them; otherwise use null\n"
            "- Use `references` on CroissantField when a column links to another recordset "
            "(format: 'recordset_name.column_name')\n\n"
            "Use tools to verify schema, column types, and sample rows when workspace artifacts "
            "are insufficient."
        ),
        "tools": [
            context_tools.get_context_overview,
            context_tools.get_context_schema,
            context_tools.get_field_names,
            context_tools.get_field_types,
            context_tools.get_sample_items,
        ],
        "temperature": 0.0,
    },
    # Specialized player for spatial and temporal data analysis
    "spatial_temporal_specialist": {
        "role_prompt": (
            "You are a spatial-temporal data specialist with expertise in geographic "
            "information systems (GIS) and time-series data. Follow this workflow:\n\n"
            "1. TEMPORAL DETECTION:\n"
            "   - Call detect_temporal_columns for each target resource\n"
            "   - Note which columns contain dates, times, timestamps, or durations\n\n"
            "2. SPATIAL DETECTION:\n"
            "   - Call detect_spatial_columns for each target resource\n"
            "   - Note coordinate columns, geometry columns, lat/lon pairs, and "
            "tuple_coord_columns (e.g. '(lon, lat)' strings)\n\n"
            "3. EXTRACT ACTUAL VALUES (required):\n"
            "   - Based on steps 1–2, call one or two follow-up tools to obtain "
            "concrete values—not just column names:\n"
            "     • analyze_temporal_column — date range, granularity, timezone for a "
            "specific temporal column\n"
            "     • analyze_spatial_column — value ranges, geometry types for a "
            "specific spatial column\n"
            "     • get_spatial_extent — bounding box from separate lat and lon columns\n"
            "     • get_spatial_extent_from_tuple_column — bounding box from a "
            "'(lon, lat)' or '(lat, lon)' tuple column\n"
            "     • get_temporal_extent — start/end dates and coverage for a "
            "timestamp column\n"
            "   - Pick the tool(s) that match what detection found; do not call all "
            "five. Examples:\n"
            "     • temporal column only → get_temporal_extent or analyze_temporal_column\n"
            "     • lat + lon columns → get_spatial_extent\n"
            "     • tuple_coords column → get_spatial_extent_from_tuple_column\n"
            "     • both spatial and temporal → call one spatial tool and one temporal tool\n\n"
            "4. METADATA OUTPUT:\n"
            "   - Report temporalCoverage and spatialCoverage using the actual values "
            "from step 3 (min/max dates, bounding box coordinates)\n"
            "   - Include coordinate system notes and per-resource coverage when relevant\n\n"
            "Be precise about coordinate systems, date formats, and geographic extents. "
            "For multi_csv contexts, analyze spatial-temporal characteristics of each resource "
            "and identify any temporal or spatial relationships between resources."
        ),
        "tools": [
            context_tools.detect_temporal_columns,
            context_tools.detect_spatial_columns,
            context_tools.analyze_temporal_column,
            context_tools.analyze_spatial_column,
            context_tools.get_spatial_extent,
            context_tools.get_spatial_extent_from_tuple_column,
            context_tools.get_temporal_extent,
        ],
        # Model-driven tool calls supply required args (column, lat/lon, etc.).
        "temperature": 0.3,  # Lower for more precise technical analysis
    },
}

# Players that produce final structured metadata output (used by plan executor).
METADATA_OUTPUT_PLAYER_NAMES = frozenset({
    "metadata_generator",
    "croissant_metadata_generator",
})
