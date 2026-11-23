"""
DEPRECATED: This module is kept for backwards compatibility only.

Use `datasource_tools.py` instead, which provides a unified interface
through the DataSource abstraction.

This file contains legacy tools that work directly with file paths.
New code should use the DataSource-based tools in datasource_tools.py.

Migration guide:
    Old: from src.tools import pandas_tools
         pandas_tools.get_row_count("./data.csv")
    
    New: from src.tools import get_row_count, register_datasource
         from src.datasource import create_datasource
         ds = create_datasource("./data.csv")
         register_datasource("my_ds", ds)
         get_row_count("my_ds", "data")
"""
import warnings

# Emit deprecation warning when this module is imported
warnings.warn(
    "pandas_tools is deprecated. Use datasource_tools instead, which provides "
    "a unified interface through the DataSource abstraction.",
    DeprecationWarning,
    stacklevel=2
)
import pandas as pd
from langchain_core.tools import tool
from typing import Dict, Any, List, Optional, Tuple
import os
import json

# Threshold for considering a file "large" (50MB)
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024
# Default chunk size for processing large files
DEFAULT_CHUNK_SIZE = 10000


def _is_large_file(file_path: str) -> bool:
    """Check if file exceeds the large file threshold."""
    try:
        return os.path.getsize(file_path) > LARGE_FILE_THRESHOLD_BYTES
    except OSError:
        return False


@tool
def get_row_count(file_path: str) -> int:
    """
    Returns the total number of rows in a structured data file (e.g., CSV).
    This is useful for getting a basic sense of the dataset's size.
    Optimized: counts rows without loading entire file into memory.
    The input must be a valid file path.
    """
    try:
        # Count rows by iterating through chunks - memory efficient
        count = 0
        for chunk in pd.read_csv(file_path, chunksize=DEFAULT_CHUNK_SIZE, usecols=[0]):
            count += len(chunk)
        return count
    except FileNotFoundError:
        return "Error: File not found at the specified path."
    except Exception as e:
        return f"Error processing file: {e}"


@tool
def get_column_names(file_path: str) -> List[str]:
    """
    Returns a list of column names from a structured data file (e.g., CSV).
    This is the first step to understanding the schema of the dataset.
    Optimized: reads only the header row.
    The input must be a valid file path.
    """
    try:
        # Read only the header row - no data loaded
        df = pd.read_csv(file_path, nrows=0)
        return df.columns.tolist()
    except FileNotFoundError:
        return ["Error: File not found at the specified path."]
    except Exception as e:
        return [f"Error processing file: {e}"]


@tool
def get_data_types(file_path: str) -> Dict[str, str]:
    """
    Returns a dictionary mapping column names to their data types.
    Useful for understanding the schema and structure of the dataset.
    Optimized: infers types from a sample of rows for large files.
    The input must be a valid file path.
    """
    try:
        if _is_large_file(file_path):
            # Sample first 10000 rows to infer types
            df = pd.read_csv(file_path, nrows=DEFAULT_CHUNK_SIZE)
        else:
            df = pd.read_csv(file_path)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    except FileNotFoundError:
        return {"error": "File not found at the specified path."}
    except Exception as e:
        return {"error": f"Error processing file: {e}"}


@tool
def get_column_statistics(file_path: str) -> Dict[str, Any]:
    """
    Returns basic statistics for all columns in a structured data file.
    For numeric columns: count, mean, std, min, max, quartiles.
    For non-numeric columns: count, unique, top, freq.
    Optimized: uses chunked processing for large files.
    The input must be a valid file path.
    """
    try:
        if _is_large_file(file_path):
            # For large files, compute statistics using chunked processing
            numeric_stats = {}
            categorical_stats = {}
            total_count = 0
            
            # First pass: get column types from sample
            sample_df = pd.read_csv(file_path, nrows=1000)
            numeric_cols = sample_df.select_dtypes(include=['number']).columns.tolist()
            non_numeric_cols = sample_df.select_dtypes(exclude=['number']).columns.tolist()
            
            # Initialize accumulators for numeric columns
            for col in numeric_cols:
                numeric_stats[col] = {
                    'count': 0, 'sum': 0, 'sum_sq': 0, 
                    'min': float('inf'), 'max': float('-inf')
                }
            
            # Initialize value counters for categorical columns
            cat_value_counts = {col: {} for col in non_numeric_cols}
            
            # Process in chunks
            for chunk in pd.read_csv(file_path, chunksize=DEFAULT_CHUNK_SIZE):
                total_count += len(chunk)
                
                # Numeric statistics
                for col in numeric_cols:
                    if col in chunk.columns:
                        valid_data = chunk[col].dropna()
                        if len(valid_data) > 0:
                            numeric_stats[col]['count'] += len(valid_data)
                            numeric_stats[col]['sum'] += valid_data.sum()
                            numeric_stats[col]['sum_sq'] += (valid_data ** 2).sum()
                            numeric_stats[col]['min'] = min(numeric_stats[col]['min'], valid_data.min())
                            numeric_stats[col]['max'] = max(numeric_stats[col]['max'], valid_data.max())
                
                # Categorical value counts (limit tracking to avoid memory issues)
                for col in non_numeric_cols:
                    if col in chunk.columns:
                        for val, cnt in chunk[col].value_counts().items():
                            if len(cat_value_counts[col]) < 1000:  # Limit unique values tracked
                                cat_value_counts[col][val] = cat_value_counts[col].get(val, 0) + cnt
            
            # Compute final statistics
            result = {}
            for col in numeric_cols:
                stats = numeric_stats[col]
                if stats['count'] > 0:
                    mean = stats['sum'] / stats['count']
                    variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                    std = variance ** 0.5 if variance > 0 else 0
                    result[col] = {
                        'count': stats['count'],
                        'mean': round(mean, 4),
                        'std': round(std, 4),
                        'min': stats['min'],
                        'max': stats['max']
                    }
            
            for col in non_numeric_cols:
                counts = cat_value_counts[col]
                if counts:
                    top_val = max(counts, key=counts.get)
                    result[col] = {
                        'count': sum(counts.values()),
                        'unique': len(counts),
                        'top': top_val,
                        'freq': counts[top_val]
                    }
            
            return result
        else:
            df = pd.read_csv(file_path)
            stats = df.describe(include='all').to_dict()
            return stats
    except FileNotFoundError:
        return {"error": "File not found at the specified path."}
    except Exception as e:
        return {"error": f"Error processing file: {e}"}


@tool
def get_missing_values(file_path: str) -> Dict[str, int]:
    """
    Returns a dictionary mapping column names to their count of missing values.
    Useful for data quality assessment.
    Optimized: uses chunked processing for large files.
    The input must be a valid file path.
    """
    try:
        if _is_large_file(file_path):
            # Process in chunks and accumulate missing counts
            missing_counts = None
            for chunk in pd.read_csv(file_path, chunksize=DEFAULT_CHUNK_SIZE):
                chunk_missing = chunk.isnull().sum()
                if missing_counts is None:
                    missing_counts = chunk_missing
                else:
                    missing_counts = missing_counts.add(chunk_missing, fill_value=0)
            return missing_counts.astype(int).to_dict() if missing_counts is not None else {}
        else:
            df = pd.read_csv(file_path)
            return df.isnull().sum().to_dict()
    except FileNotFoundError:
        return {"error": "File not found at the specified path."}
    except Exception as e:
        return {"error": f"Error processing file: {e}"}


@tool
def get_sample_rows(file_path: str, n: int = 5) -> str:
    """
    Returns a string representation of the first n rows of the dataset.
    Useful for understanding the actual data content.
    Optimized: reads only the required rows.
    The input must be a valid file path.
    """
    try:
        # Only read the rows we need - efficient for any file size
        df = pd.read_csv(file_path, nrows=n)
        return df.to_string()
    except FileNotFoundError:
        return "Error: File not found at the specified path."
    except Exception as e:
        return f"Error processing file: {e}"


@tool
def get_unique_values(file_path: str, column_name: str, max_unique: int = 100) -> List[Any]:
    """
    Returns a list of unique values in a specific column.
    Useful for understanding categorical columns.
    Optimized: uses chunked processing for large files.
    The input must be a valid file path and column name.
    """
    try:
        # First check if column exists
        header = pd.read_csv(file_path, nrows=0)
        if column_name not in header.columns:
            return [f"Error: Column '{column_name}' not found in dataset."]
        
        if _is_large_file(file_path):
            # Collect unique values across chunks
            unique_set = set()
            for chunk in pd.read_csv(file_path, usecols=[column_name], chunksize=DEFAULT_CHUNK_SIZE):
                unique_set.update(chunk[column_name].dropna().unique())
                # Early exit if we've found enough unique values
                if len(unique_set) > max_unique:
                    break
            
            unique_vals = list(unique_set)
        else:
            df = pd.read_csv(file_path, usecols=[column_name])
            unique_vals = df[column_name].unique().tolist()
        
        # Limit output to avoid huge responses
        if len(unique_vals) > max_unique:
            return unique_vals[:max_unique] + [f"... and {len(unique_vals) - max_unique} more"]
        return unique_vals
    except FileNotFoundError:
        return ["Error: File not found at the specified path."]
    except Exception as e:
        return [f"Error processing file: {e}"]


@tool
def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Returns basic file information including size and estimated row count.
    Useful as a first step to understand the dataset scale.
    The input must be a valid file path.
    """
    try:
        file_size = os.path.getsize(file_path)
        
        # Get column count and estimate row count from sample
        sample = pd.read_csv(file_path, nrows=100)
        num_columns = len(sample.columns)
        
        # Estimate total rows based on file size and sample
        sample_file_pos = sample.memory_usage(deep=True).sum()
        estimated_rows = int((file_size / sample_file_pos) * 100) if sample_file_pos > 0 else "unknown"
        
        return {
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "num_columns": num_columns,
            "column_names": sample.columns.tolist(),
            "estimated_rows": estimated_rows,
            "is_large_file": _is_large_file(file_path)
        }
    except FileNotFoundError:
        return {"error": "File not found at the specified path."}
    except Exception as e:
        return {"error": f"Error processing file: {e}"}


# ===================================================================
#  MULTI-FILE / RELATIONAL ANALYSIS TOOLS
# ===================================================================

@tool
def get_multi_file_overview(file_paths_json: str) -> Dict[str, Any]:
    """
    Get an overview of multiple files in a dataset.
    Input must be a JSON string mapping table names to file paths.
    Example: '{"users": "./data/users.csv", "orders": "./data/orders.csv"}'
    Returns summary info for each table including row counts and columns.
    """
    try:
        file_paths = json.loads(file_paths_json)
        overview = {}
        
        for table_name, file_path in file_paths.items():
            try:
                # Get basic info for each file
                sample = pd.read_csv(file_path, nrows=100)
                row_count = sum(1 for _ in pd.read_csv(file_path, chunksize=DEFAULT_CHUNK_SIZE, usecols=[0]))
                row_count = sum(chunk.shape[0] for chunk in pd.read_csv(file_path, chunksize=DEFAULT_CHUNK_SIZE, usecols=[0]))
                
                overview[table_name] = {
                    "file_path": file_path,
                    "row_count": row_count,
                    "column_count": len(sample.columns),
                    "columns": sample.columns.tolist(),
                    "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
                }
            except Exception as e:
                overview[table_name] = {"error": str(e)}
        
        return overview
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input. Provide a JSON string mapping table names to file paths."}
    except Exception as e:
        return {"error": f"Error processing files: {e}"}


@tool
def find_common_columns(file_paths_json: str) -> Dict[str, Any]:
    """
    Find columns that appear in multiple tables - potential join keys.
    Input must be a JSON string mapping table names to file paths.
    Returns columns grouped by how many tables they appear in.
    """
    try:
        file_paths = json.loads(file_paths_json)
        column_to_tables: Dict[str, List[str]] = {}
        
        for table_name, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path, nrows=0)
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if col_lower not in column_to_tables:
                        column_to_tables[col_lower] = []
                    column_to_tables[col_lower].append(f"{table_name}.{col}")
            except Exception as e:
                continue
        
        # Group by number of occurrences
        shared_columns = {col: tables for col, tables in column_to_tables.items() if len(tables) > 1}
        unique_columns = {col: tables[0] for col, tables in column_to_tables.items() if len(tables) == 1}
        
        return {
            "shared_columns": shared_columns,
            "shared_column_count": len(shared_columns),
            "unique_columns_count": len(unique_columns),
            "potential_join_keys": list(shared_columns.keys())
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input. Provide a JSON string mapping table names to file paths."}
    except Exception as e:
        return {"error": f"Error processing files: {e}"}


@tool
def analyze_column_relationship(
    file_path_a: str,
    column_a: str,
    file_path_b: str,
    column_b: str
) -> Dict[str, Any]:
    """
    Analyze the relationship between two columns from different files.
    Determines if they could be a foreign key relationship and what type.
    Returns match rate, cardinality, and relationship type suggestion.
    """
    try:
        # Read the columns efficiently
        if _is_large_file(file_path_a):
            values_a = set()
            for chunk in pd.read_csv(file_path_a, usecols=[column_a], chunksize=DEFAULT_CHUNK_SIZE):
                values_a.update(chunk[column_a].dropna().unique())
        else:
            df_a = pd.read_csv(file_path_a, usecols=[column_a])
            values_a = set(df_a[column_a].dropna().unique())
        
        if _is_large_file(file_path_b):
            values_b = set()
            for chunk in pd.read_csv(file_path_b, usecols=[column_b], chunksize=DEFAULT_CHUNK_SIZE):
                values_b.update(chunk[column_b].dropna().unique())
        else:
            df_b = pd.read_csv(file_path_b, usecols=[column_b])
            values_b = set(df_b[column_b].dropna().unique())
        
        # Calculate overlap
        intersection = values_a & values_b
        
        match_rate_a_to_b = len(intersection) / len(values_a) if values_a else 0
        match_rate_b_to_a = len(intersection) / len(values_b) if values_b else 0
        
        # Determine relationship type based on cardinality
        unique_count_a = len(values_a)
        unique_count_b = len(values_b)
        
        # Get actual row counts for cardinality analysis
        total_rows_a = sum(1 for _ in open(file_path_a)) - 1  # minus header
        total_rows_b = sum(1 for _ in open(file_path_b)) - 1
        
        # Determine if columns are unique (potential primary keys)
        is_unique_a = unique_count_a >= total_rows_a * 0.99  # 99% threshold
        is_unique_b = unique_count_b >= total_rows_b * 0.99
        
        # Infer relationship type
        if is_unique_a and is_unique_b:
            relationship_type = "one-to-one"
        elif is_unique_a and not is_unique_b:
            relationship_type = "one-to-many (A -> B)"
        elif not is_unique_a and is_unique_b:
            relationship_type = "one-to-many (B -> A)"
        else:
            relationship_type = "many-to-many"
        
        # Calculate confidence score
        confidence = (match_rate_a_to_b + match_rate_b_to_a) / 2
        
        return {
            "column_a": column_a,
            "column_b": column_b,
            "values_in_a": unique_count_a,
            "values_in_b": unique_count_b,
            "common_values": len(intersection),
            "match_rate_a_to_b": round(match_rate_a_to_b, 4),
            "match_rate_b_to_a": round(match_rate_b_to_a, 4),
            "is_unique_in_a": is_unique_a,
            "is_unique_in_b": is_unique_b,
            "suggested_relationship_type": relationship_type,
            "confidence_score": round(confidence, 4),
            "likely_foreign_key": confidence > 0.5 and (is_unique_a or is_unique_b),
            "sample_common_values": list(intersection)[:10]
        }
    except FileNotFoundError as e:
        return {"error": f"File not found: {e}"}
    except KeyError as e:
        return {"error": f"Column not found: {e}"}
    except Exception as e:
        return {"error": f"Error analyzing relationship: {e}"}


@tool
def discover_relationships(file_paths_json: str, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Automatically discover potential relationships between tables.
    Analyzes columns with similar names and overlapping values.
    Input must be a JSON string mapping table names to file paths.
    Returns a list of potential relationships with confidence scores.
    """
    try:
        file_paths = json.loads(file_paths_json)
        tables_data: Dict[str, Dict[str, Any]] = {}
        
        # Load sample data from each table
        for table_name, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path, nrows=sample_size)
                tables_data[table_name] = {
                    "path": file_path,
                    "columns": df.columns.tolist(),
                    "data": df
                }
            except Exception as e:
                continue
        
        relationships = []
        table_names = list(tables_data.keys())
        
        # Compare each pair of tables
        for i, table_a in enumerate(table_names):
            for table_b in table_names[i+1:]:
                data_a = tables_data[table_a]
                data_b = tables_data[table_b]
                
                # Find columns with similar names
                for col_a in data_a["columns"]:
                    col_a_normalized = col_a.lower().replace("_", "").replace("-", "")
                    
                    for col_b in data_b["columns"]:
                        col_b_normalized = col_b.lower().replace("_", "").replace("-", "")
                        
                        # Check for name similarity
                        name_match = (
                            col_a_normalized == col_b_normalized or
                            col_a_normalized.endswith("id") and col_b_normalized.endswith("id") and 
                            col_a_normalized[:-2] == col_b_normalized[:-2] or
                            col_a_normalized in col_b_normalized or col_b_normalized in col_a_normalized
                        )
                        
                        if not name_match:
                            continue
                        
                        # Analyze value overlap
                        values_a = set(data_a["data"][col_a].dropna().unique())
                        values_b = set(data_b["data"][col_b].dropna().unique())
                        
                        if not values_a or not values_b:
                            continue
                        
                        intersection = values_a & values_b
                        if not intersection:
                            continue
                        
                        match_rate = len(intersection) / min(len(values_a), len(values_b))
                        
                        if match_rate > 0.1:  # At least 10% overlap
                            # Determine relationship type
                            unique_ratio_a = len(values_a) / len(data_a["data"])
                            unique_ratio_b = len(values_b) / len(data_b["data"])
                            
                            if unique_ratio_a > 0.9 and unique_ratio_b > 0.9:
                                rel_type = "one-to-one"
                            elif unique_ratio_a > 0.9:
                                rel_type = "one-to-many"
                            elif unique_ratio_b > 0.9:
                                rel_type = "many-to-one"
                            else:
                                rel_type = "many-to-many"
                            
                            relationships.append({
                                "from_table": table_a,
                                "from_column": col_a,
                                "to_table": table_b,
                                "to_column": col_b,
                                "relationship_type": rel_type,
                                "match_rate": round(match_rate, 4),
                                "confidence": round(match_rate * 0.8 + 0.2 if name_match else match_rate * 0.5, 4),
                                "common_value_count": len(intersection),
                                "sample_common_values": list(intersection)[:5]
                            })
        
        # Sort by confidence
        relationships.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "tables_analyzed": list(tables_data.keys()),
            "relationships_found": len(relationships),
            "relationships": relationships
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input. Provide a JSON string mapping table names to file paths."}
    except Exception as e:
        return {"error": f"Error discovering relationships: {e}"}


@tool
def get_join_preview(
    file_path_a: str,
    column_a: str,
    file_path_b: str,
    column_b: str,
    join_type: str = "inner",
    preview_rows: int = 5
) -> str:
    """
    Preview the result of joining two tables.
    Useful for validating a suspected relationship.
    join_type can be: 'inner', 'left', 'right', 'outer'
    Returns a string representation of the joined preview.
    """
    try:
        # Read samples
        df_a = pd.read_csv(file_path_a, nrows=1000)
        df_b = pd.read_csv(file_path_b, nrows=1000)
        
        # Perform join
        if join_type not in ["inner", "left", "right", "outer"]:
            join_type = "inner"
        
        joined = pd.merge(
            df_a, df_b,
            left_on=column_a,
            right_on=column_b,
            how=join_type,
            suffixes=("_table_a", "_table_b")
        )
        
        result_info = f"Join result: {len(joined)} rows from {len(df_a)} x {len(df_b)} samples\n"
        result_info += f"Join type: {join_type}\n"
        result_info += f"Columns: {joined.columns.tolist()}\n\n"
        result_info += f"Preview ({preview_rows} rows):\n"
        result_info += joined.head(preview_rows).to_string()
        
        return result_info
    except FileNotFoundError as e:
        return f"Error: File not found - {e}"
    except KeyError as e:
        return f"Error: Column not found - {e}"
    except Exception as e:
        return f"Error performing join: {e}"


@tool
def compare_table_schemas(file_paths_json: str) -> Dict[str, Any]:
    """
    Compare schemas across multiple tables to identify patterns.
    Useful for understanding dataset structure and identifying normalization patterns.
    Input must be a JSON string mapping table names to file paths.
    """
    try:
        file_paths = json.loads(file_paths_json)
        schemas = {}
        
        for table_name, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path, nrows=100)
                schemas[table_name] = {
                    "columns": {
                        col: {
                            "dtype": str(df[col].dtype),
                            "sample_values": df[col].dropna().head(3).tolist(),
                            "null_count_in_sample": int(df[col].isnull().sum())
                        }
                        for col in df.columns
                    },
                    "column_count": len(df.columns)
                }
            except Exception as e:
                schemas[table_name] = {"error": str(e)}
        
        # Find schema patterns
        all_columns = set()
        for table, schema in schemas.items():
            if "columns" in schema:
                all_columns.update(schema["columns"].keys())
        
        # Identify potential ID columns (usually end with 'id' or '_id')
        potential_ids = [col for col in all_columns if col.lower().endswith('id') or col.lower().endswith('_id')]
        
        # Identify timestamp/date columns
        date_keywords = ['date', 'time', 'created', 'updated', 'timestamp']
        potential_dates = [col for col in all_columns if any(kw in col.lower() for kw in date_keywords)]
        
        return {
            "schemas": schemas,
            "total_unique_columns": len(all_columns),
            "potential_id_columns": potential_ids,
            "potential_date_columns": potential_dates,
            "all_columns": sorted(list(all_columns))
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input. Provide a JSON string mapping table names to file paths."}
    except Exception as e:
        return {"error": f"Error comparing schemas: {e}"}
