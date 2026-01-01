---
title: Data Modeling with Apache Cassandra
category: Data Engineering
date: 2021-06-21
client: Udacity (Project)
description: Building an ETL pipeline and data modeling with Apache Cassandra to analyze song play data for a music streaming startup.
imageUrl: /images/cassandra-data-modeling.png
link: "https://github.com/shrikantnaidu/Data-Modeling-with-Cassandra"
tags:
  - Apache Cassandra
  - NoSQL
  - ETL
  - Python
---

Sparkify, a music streaming startup, needs to analyze the data they've been collecting on songs and user activity. The analytics team wants to understand what songs users are listening to, but their data is currently stored in JSON logs without an efficient way to query it. Our task is to create an Apache Cassandra database that enables fast and efficient querying of song play data.

### Project Outline

- Process raw JSON files and create a single CSV file
- Design Cassandra tables based on specific query requirements
- Build ETL pipeline to load data into Cassandra
- Validate the data model with test queries

---

### Data Processing

#### Step 1: ETL Process for Event Data

First, we process the JSON log files and combine them into a single CSV file:

```python
import os
import glob
import json
import csv

def process_event_files():
    # Get current folder and filepath to event data
    filepath = os.getcwd() + '/event_data'
    
    # Create a list of files and collect filepath to CSV
    file_path_list = glob.glob(os.path.join(filepath, '*/*.json'))
    
    # Read all event files and write to csv
    full_data_rows_list = []
    for f in file_path_list:
        with open(f, 'r') as filedata:
            data = json.loads(filedata.read())
            full_data_rows_list.append(data)
            
    # Write processed data to CSV file
    csv_file = 'event_datafile_new.csv'
    header = ["artist", "firstName", "gender", "itemInSession", "lastName",
              "length", "level", "location", "sessionId", "song", "userId"]
    
    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in full_data_rows_list:
            if row["page"] == "NextSong":
                writer.writerow(row)
```

---

### Cassandra Data Modeling

#### Understanding Cassandra's Data Model

Before diving into specific tables, it's important to understand Cassandra's key concepts:

- **Partition Key**: Determines data distribution across nodes
- **Clustering Columns**: Determines data sorting within partitions
- **Primary Key**: Combination of partition key and clustering columns

---

#### Query 1: Song Details by Session

First query requirement: Get artist, song title and song length for a specific session ID and item in session.

```python
# Creating the table
query1_create = """
CREATE TABLE IF NOT EXISTS song_details_by_session (
    sessionId int,
    itemInSession int,
    artist text,
    song text,
    length float,
    PRIMARY KEY ((sessionId), itemInSession)
)
"""
```

**Sample query:**

```python
query1_select = """
    SELECT artist, song, length 
    FROM song_details_by_session 
    WHERE sessionId = 338 AND itemInSession = 4
"""
```

**Primary key design explanation:**
- **Partition key**: `sessionId` (groups all songs in a session)
- **Clustering key**: `itemInSession` (orders songs within the session)

---

#### Query 2: User Song History

Second query requirement: Get artist, song, and user details for a specific user session.

```python
# Creating the table  
query2_create = """
CREATE TABLE IF NOT EXISTS user_song_history (
    userId int,
    sessionId int,
    itemInSession int,
    artist text,
    song text,
    firstName text,
    lastName text,
    PRIMARY KEY ((userId, sessionId), itemInSession)
)
"""
```

**Sample query:**

```python
query2_select = """
    SELECT artist, song, firstName, lastName
    FROM user_song_history 
    WHERE userId = 10 AND sessionId = 182
"""
```

**Primary key design explanation:**
- **Composite partition key**: `(userId, sessionId)` (groups songs by user session)
- **Clustering key**: `itemInSession` (maintains song order)

---

#### Query 3: Users by Song

Third query requirement: Find all users who listened to a specific song.

```python
# Creating the table
query3_create = """
CREATE TABLE IF NOT EXISTS users_by_song (
    song text,
    userId int,
    firstName text,
    lastName text,
    PRIMARY KEY ((song), userId)
)
"""
```

**Sample query:**

```python
query3_select = """
    SELECT firstName, lastName
    FROM users_by_song 
    WHERE song = 'All Hands Against His Own'
"""
```

**Primary key design explanation:**
- **Partition key**: `song` (groups all users who listened to a song)
- **Clustering key**: `userId` (ensures unique user entries)

---

### ETL Pipeline Implementation

Here's how we implement the data loading process:

```python
def load_data_into_tables(session):
    file = 'event_datafile_new.csv'
    with open(file, encoding='utf8') as f:
        csvreader = csv.reader(f)
        next(csvreader)  # Skip header       
        for line in csvreader:
            # Insert into song_details_by_session
            query1 = """
                INSERT INTO song_details_by_session 
                (sessionId, itemInSession, artist, song, length)
                VALUES (%s, %s, %s, %s, %s)
            """
            session.execute(query1, (int(line[8]), int(line[3]), line[0], 
                                   line[9], float(line[5])))
```

---

### Data Validation

Let's verify our data model by running test queries:

```python
def validate_tables(session):
    # Test Query 1
    rows = session.execute("""
        SELECT artist, song, length 
        FROM song_details_by_session 
        WHERE sessionId = 338 AND itemInSession = 4
    """)
    for row in rows:
        print(f"Artist: {row.artist}, Song: {row.song}, Length: {row.length}")
```

---

### Performance Considerations

- **Partition Size**: Keep partitions under 100MB
- **Clustering Column Order**: Most frequently used filters first
- **Denormalization**: Duplicate data to optimize query performance

---

### Conclusion

This project demonstrates how to:
- Process raw JSON logs into a structured format
- Design Cassandra tables using query-first approach
- Implement an ETL pipeline for data loading
- Create efficient queries using appropriate primary keys

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Data-Modeling-with-Cassandra).

---

### Important Notes

- Always drop tables between test runs to ensure clean state
- Monitor partition sizes in production
- Consider data distribution when designing partition keys
- Use batch processing for related data inserts

> **Remember to shut down your Cassandra cluster when not in use to avoid unnecessary resource consumption.**
