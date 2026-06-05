# Stock Market Social Network - Temporal Link Prediction

This project analyzes the stock holding network of investment funds over time (2021-2024) and performs Temporal Link Prediction by combining Graph Neural Networks (GNN) with decision-tree-based models.

## Pipeline Architecture

The system operates using a **Sliding Window** structure, where it learns from a two-year window (8 quarters) to predict connections in the subsequent quarter.

### Workflow Diagram

```mermaid
graph TD
    subgraph "1. Data Setup"
        A[Load Parquet Files] --> B[Clean & Filter 2021-2024]
        B --> C[Normalize Column Names to Uppercase]
    end

    subgraph "2. Graph Construction"
        C --> D[Create Quarterly Bipartite Graphs]
        D --> E[Funds <--> Stocks]
    end

    subgraph "3. Sliding Window Loop"
        E --> F{Is there a next window?}
        F -- Yes --> G[Combine Training Window Graphs - 8 Quarters]
        
        subgraph "Parallel Training Process"
            G --> H[Compute Topological Features: Centrality, Community]
            G --> I[Train GraphSAGE: Generate Embeddings]
            I -.-> |Transfer Learning| I
        end
        
        H & I --> J[Negative Sampling: Positive & Negative Samples]
        J --> K[Train LightGBM Model]
    end

    subgraph "4. Evaluation & Output"
        K --> L[Evaluate against Target Quarter - Test Quarter]
        L --> M[Compute Metrics: AUC, Precision, Recall]
        M --> N[Save Model to Pickle & Log Results to CSV]
        N --> F
    end

    subgraph "5. Future Usage - Inference"
        N --> O[Load Trained Model via NodeConnectionPredictor]
        O --> P[Predict New Connections for Specific Fund or Stock]
    end

    F -- No --> Q[End: Final Report final_scores_report.csv]