---
title: Data Pipelines with Airflow
category: Data Engineering
date: 2021-07-17
client: Udacity (Project)
description: Automating ETL processes for Sparkify's data warehouse using Apache Airflow to orchestrate workflows and ensure data quality.
imageUrl: /images/airflow-pipeline.png
link: "https://github.com/shrikantnaidu/Data-Pipelines-with-Airflow"
tags:
  - Apache Airflow
  - ETL
  - AWS Redshift
  - Python
---

Sparkify, a music streaming startup, has a growing need to manage and analyze vast amounts of user activity data. To facilitate this, the analytics team requires a robust ETL (Extract, Transform, Load) pipeline that can efficiently process and load data into a data warehouse. This project leverages Apache Airflow to orchestrate the ETL workflow, ensuring that data is consistently and reliably processed.

The challenge lies in automating the data pipeline to handle various data sources, transformations, and loading processes while maintaining data integrity and performance.

---

### Technical Architecture

The architecture of the data pipeline is designed to ensure data quality and efficient processing. The source datasets consist of JSON logs detailing user activity and song metadata.

#### Project Structure

```
data_pipelines_airflow/
│
├── dags/                         # Directory for DAG definitions
│   └── udac_example_dag.py       # Defines the DAG for scheduling tasks
│
├── plugins/                      # Custom plugins for Airflow
│   ├── operators/                # Custom operator plugins
│   │   ├── stage_redshift.py     # Stages data from S3 to Redshift
│   │   ├── load_fact.py          # Loads data into the fact table
│   │   ├── load_dimension.py     # Loads data into dimension tables
│   │   └── data_quality.py       # Performs data quality checks
│   └── helpers/                  # Helper modules for the plugins
│
└── README.md                     # Project documentation
```

#### Core Components

**DAGs (Directed Acyclic Graphs)**
- Define the workflow and task dependencies
- Schedule and manage task execution
- Handle retries and failure notifications

**Custom Operators**
- Encapsulate logic for specific tasks
- `StageToRedshiftOperator` - Stages data from S3 to Redshift
- `LoadFactOperator` - Loads data into fact tables
- `LoadDimensionOperator` - Loads data into dimension tables
- `DataQualityOperator` - Performs data quality checks

**Data Quality Checks**
- Ensures integrity of data after ETL steps
- Catches discrepancies early in the pipeline
- Validates record counts and null constraints

---

### Database Schema

#### Fact Table

**songplays** - Records of song plays, capturing user interactions

```sql
CREATE TABLE IF NOT EXISTS songplays (
    songplay_id SERIAL PRIMARY KEY,
    start_time timestamp NOT NULL,
    user_id int NOT NULL,
    level varchar,
    song_id varchar,
    artist_id varchar,
    session_id int,
    location varchar,
    user_agent varchar,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (song_id) REFERENCES songs (song_id),
    FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
    FOREIGN KEY (start_time) REFERENCES time (start_time)
);
```

#### Dimension Tables

**users** - User information

```sql
CREATE TABLE IF NOT EXISTS users (
    user_id int PRIMARY KEY,
    first_name varchar,
    last_name varchar,
    gender varchar,
    level varchar
);
```

**songs** - Song metadata

```sql
CREATE TABLE IF NOT EXISTS songs (
    song_id varchar PRIMARY KEY,
    title varchar,
    artist_id varchar,
    year int,
    duration float
);
```

**artists** - Artist information

```sql
CREATE TABLE IF NOT EXISTS artists (
    artist_id varchar PRIMARY KEY,
    name varchar,
    location varchar,
    latitude float,
    longitude float
);
```

**time** - Timestamps of records broken down

```sql
CREATE TABLE IF NOT EXISTS time (
    start_time timestamp PRIMARY KEY,
    hour int,
    day int,
    week int,
    month int,
    year int,
    weekday int
);
```

---

### ETL Pipeline Implementation

#### 1. Staging Data

The ETL pipeline first stages data from S3 to Redshift. The `StageToRedshiftOperator` handles this process, ensuring that data is loaded into staging tables before being transformed into the final schema.

#### 2. Loading Fact and Dimension Tables

The `LoadFactOperator` and `LoadDimensionOperator` are responsible for loading data into the fact and dimension tables. These operators utilize SQL commands to insert data into the appropriate tables.

#### 3. Data Quality Checks

The `DataQualityOperator` runs validation checks after data loading:
- Verifies that tables contain records
- Checks for null values in critical columns
- Validates referential integrity

---

### Example Queries and Results

#### 1. Most Active Users

```sql
SELECT u.first_name, u.last_name, COUNT(*) as play_count
FROM songplays sp
JOIN users u ON sp.user_id = u.user_id
GROUP BY u.user_id, u.first_name, u.last_name
ORDER BY play_count DESC
LIMIT 5;
```

#### 2. Popular Music Hours

```sql
SELECT t.hour, COUNT(*) as play_count
FROM songplays sp
JOIN time t ON sp.start_time = t.start_time
GROUP BY t.hour
ORDER BY play_count DESC;
```

---

### Key Achievements

- Designed a robust ETL pipeline using Apache Airflow for data orchestration
- Implemented custom operators for modular and reusable pipeline components
- Created data quality checks to ensure data integrity after each ETL step
- Built a scalable architecture that can handle various data sources and transformations
- Configured proper task dependencies and retry logic for fault tolerance

---

### Technologies Used

**Apache Airflow**
- DAG-based workflow orchestration
- Task scheduling and monitoring
- Custom operator development

**AWS Redshift**
- Data warehouse for analytics
- Columnar storage for fast queries

**AWS S3**
- Source data storage
- JSON log files and song metadata

**Python**
- ETL scripting and automation
- Custom operator implementation

---

### Future Improvements

- Implement incremental loading to optimize data processing
- Enhance monitoring and alerting for the ETL processes
- Add more complex data transformations to support advanced analytics
- Implement SLA monitoring for critical pipelines
- Add data lineage tracking

---

### Conclusion

This project exemplifies the power of Apache Airflow in automating and managing ETL processes for data warehouses. By implementing a robust data pipeline with custom operators and data quality checks, Sparkify can efficiently process and analyze user activity data, leading to valuable insights into user behavior and song preferences.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Data-Pipelines-with-Airflow).
