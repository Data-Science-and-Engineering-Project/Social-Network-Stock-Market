# Machine Learning Pipeline & Model Generation Prompt

**Role & Objective:**
You are an expert Data Engineer and Data Scientist specializing in large-scale data processing, graph networks, and financial temporal data. Your objective is to write a complete, production-ready, and highly optimized Python script that implements a specific Machine Learning model or baseline. 

The script must handle the entire pipeline: from loading data from a remote server environment (Slurm cluster), building the necessary data structures (e.g., graphs), processing the data using temporal sliding windows, evaluating the model, and saving the results.

---

## 1. Environment & Infrastructure Context
*   **System Type:** Linux-based Slurm cluster (`slurm-login-02`).
*   **Hardware Constraints:** The server relies heavily on CPU. Operations must be highly memory-efficient to prevent out-of-memory (OOM) crashes.
*   **Data Scale:** Approximately 86 Million rows of temporal data spanning multiple years.
*   **Coding Philosophy:** 
    *   Strict memory management is required (use `del` and `gc.collect()` explicitly).
    *   Avoid loading massive structures into RAM if lazy evaluation or chunking is possible.
    *   Include extensive `print()` logging to track progress, stages, and memory usage.

---

## 2. Data Architecture & Pathing
The script must interact with the following directory structure and file formats exactly as specified. Do not invent mock data.

**Base Paths:**
*   Root Directory: `~/Social-Network-Stock-Market/SocialNetwork/parquet_files`
*   Combined Parquet Files: `./generated_combined_parquet/`

**Input Files:**
1.  **Holdings Data (Features & Labels):**
    *   Path: `./generated_combined_parquet/holdings_filtered_new_period_start_{YYYY-MM-DD}.parquet`
    *   Frequency: Quarterly files (e.g., `..._2013-04-01.parquet`, `..._2013-07-01.parquet`, etc.) spanning from 2013 to 2025.
    *   *Schema Details:* Columns are strictly lowercase in the source files but must be converted to UPPERCASE upon loading (`CIK`, `CUSIP`, `SSHPRNAMT`, `PERIOD_START` or `PERIOD_DATE`). Note: `VALUE` column might be missing and should be derived from `SSHPRNAMT`.
2.  **Reference Data:**
    *   `./ticker_prices.parquet`
    *   `./ticker_to_cusip.parquet`

---

## 3. Data Processing & Methodology Rules
*   **Temporal Logic (Strict Causality):** The data represents time-series events (Quarterly). The model must utilize a **Sliding Window Approach** (e.g., Train on N quarters, Test on N+1 quarter). Future data leakage is strictly prohibited.
*   **Graph Construction (If applicable):** The data forms a bipartite network (Funds/CIK <-> Stocks/CUSIP). If the model requires graph topology, construct it per quarter using lightweight libraries (e.g., `networkx`, but optimize for speed).
*   **Negative Sampling:** For link prediction tasks, explicitly define a memory-efficient negative sampling strategy for the test sets (e.g., 1:1 ratio) to evaluate non-existent links.

---

## 4. The Model / Task Specification
*Provide the specific details of the model you want the AI to build in this section.*

*   **Model Type / Algorithm:** [e.g., Random Forest, Matrix Factorization, GraphSAGE, Simple Heuristics]
*   **Target Variable:** [e.g., Predict if a Fund (CIK) will hold a Stock (CUSIP) in the target quarter].
*   **Feature Engineering Required:** [e.g., Compute rolling averages, degree centrality, node embeddings].
*   **Libraries Allowed:** [e.g., pandas, numpy, scikit-learn, networkx. *Specify if heavy libraries like torch or DGL should be avoided due to CPU limits*].

---

## 5. Output & Evaluation Requirements
The script must conclude by generating comprehensive evaluation metrics and saving artifacts.

*   **Metrics Required:** Calculate and print AUC-ROC, Precision, Recall, and F1-Score for *every* sliding window iteration, and output an aggregate summary at the end.
*   **Result Persistence:** 
    *   Save detailed metric reports as a CSV file to a `./results/` directory (e.g., `./results/model_name_evaluation.csv`).
    *   If trained models need to be saved, serialize them (e.g., using `joblib` or `pickle`) into a `./models/` directory.

---

## 6. Output Format Expectation
Generate a single, cohesive Python script (`.py`). The code must be clean, well-commented, and ready to be executed via a standard `python script_name.py` or submitted as an `sbatch` job. Do not provide fragmented code blocks. Include standard Python error handling where necessary.