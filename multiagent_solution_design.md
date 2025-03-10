# **AWS Bedrock-Based Text-to-SQL Application**

## **1. Overview**

This document outlines the architecture for a Text-to-SQL application using AWS Bedrock with a multi-agent system. The system translates natural language queries into SQL, executes them in Amazon Athena, and iteratively refines queries in case of errors.

## **2. Architecture Design**

### **Agents in the System**

1. **Orchestrator Agent**

   - Handles **Natural Language Understanding (NLU)** and extracts intent.
   - Retrieves schema metadata from AWS Glue Data Catalog or OpenSearch.
   - Passes structured query intent to the SQL Generation Agent.

2. **SQL Generation Agent**

   - Constructs SQL queries using retrieved metadata.
   - References a **knowledge layer (JSON/CSV)** to ensure robust query generation:
     - **Join conditions** between tables.
     - **Column alias mapping** for logically identical columns.
     - **Data type constraints** for compatibility.
   - Outputs a well-formed SQL query to the SQL Executor Handling Agent.

3. **SQL Executor Handling Agent**

   - Executes queries in **Amazon Athena**.
   - Handles **error detection and iterative refinement**:
     - Identifies syntax errors, join issues, or datatype mismatches.
     - Modifies the SQL query using knowledge layer rules.
     - Retries up to **5 times** before returning an error message.

### **Data Flow**

#### **System Workflow**

1. **User Query Submission** → User submits a natural language query through the UI.
2. **Intent Extraction & Schema Retrieval** → The Orchestrator Agent extracts intent and retrieves schema details.
3. **SQL Generation** → SQL Generation Agent constructs a query using metadata and the knowledge layer.
4. **Execution & Refinement** → SQL Executor Handling Agent executes the query in Athena.
   - If errors occur, it refines and retries up to **5 times** before failing gracefully.
5. **Final Output** → The system returns results or an error message to the UI.

#### **Architecture Diagram**

```
+-------------------+       +------------------------+       +-------------------+
|  User Query UI   | --->  |  Orchestrator Agent    | --->  |  SQL Generation  |
|                 |       | (NLU & Schema Retrieval)|       |      Agent       |
+-------------------+       +------------------------+       +-------------------+
                                      |                           |
                                      v                           v
                        +----------------------+     +--------------------------+
                        |  Knowledge Layer     |     |  SQL Executor Handling   |
                        | (JSON / CSV)         |     | Agent (Exec & Refinement)|
                        +----------------------+     +--------------------------+
                                      |                           |
                                      v                           v
                               +-------------------+        +------------------+
                               |  Amazon Athena   | -----> |  Response to UI  |
                               +-------------------+        +------------------+
```

## **3. AWS Services Used**

- **Amazon Bedrock** → Hosts LLM-based agents for NLU, query generation, and execution handling.
- **Amazon Athena** → Executes dynamically generated SQL queries.
- **AWS Glue Data Catalog** → Stores schema metadata for accurate query generation.
- **Amazon S3** → Stores DDL files and query execution logs.
- **Amazon OpenSearch (Optional)** → Enhances metadata retrieval using vector search.
- **Amazon CloudWatch** → Monitors execution logs and errors.

## **4. Error Handling & Query Refinement**

1. SQL Execution errors (syntax, join, datatype mismatch) are caught by the **SQL Executor Handling Agent**.
2. The agent modifies the query using the **knowledge layer** (JSON/CSV mappings).
3. Query is **retried up to 5 times**.
4. If still unsuccessful, an error message is sent to the UI.

## **5. Knowledge Layer (JSON/CSV Sample)**

### **Sample JSON Format**

```json
{
  "tables": [
    {
      "table_name": "orders",
      "aliases": ["order_details"],
      "columns": [
        {"name": "order_id", "alias": "id", "data_type": "INT"},
        {"name": "customer_name", "alias": "client_name", "data_type": "STRING"}
      ],
      "joins": [
        {"target_table": "customers", "on": "orders.customer_id = customers.id"}
      ]
    }
  ]
}
```

### **Sample CSV Format**

| table_name | column_name      | alias         | data_type | join_table | join_condition                  |
|------------|-----------------|--------------|-----------|------------|---------------------------------|
| orders     | order_id        | id           | INT       | customers  | orders.customer_id = customers.id |
| orders     | customer_name   | client_name  | STRING    |            |                                 |

## **6. Pros and Cons of Single-Agent vs Multi-Agent System**

### **Single-Agent System**

#### ✅ Pros:
- Simpler to implement and maintain.
- Lower latency since all tasks are handled in a single execution.
- Fewer inter-agent communication overheads.

#### ❌ Cons:
- Difficult to scale as responsibilities increase.
- Harder to debug since all logic resides in one agent.
- Performance bottleneck if agent handles multiple tasks simultaneously.

### **Multi-Agent System (Chosen Approach)**

#### ✅ Pros:
- **Scalability:** Each agent handles a specialized task, making it easier to scale.
- **Modular Debugging:** Issues can be traced to a specific agent.
- **Better Performance:** Tasks are distributed, reducing bottlenecks.

#### ❌ Cons:
- **Inter-agent communication overhead** can introduce latency.
- **More complex implementation** compared to a single-agent system.

## **7. Benefits of this Architecture**

✅ **Optimized Query Generation** → Using schema metadata and predefined join conditions.
✅ **Efficient Error Handling** → SQL execution is self-healing via iterative refinement.
✅ **Scalability** → Serverless AWS architecture with auto-scaling agents.
✅ **Production-Grade** → Uses AWS-native services for security, monitoring, and high availability.

---


