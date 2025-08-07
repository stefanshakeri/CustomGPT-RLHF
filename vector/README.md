# Vectorization Method

## Functionality

### Expert Response Vectorization
The expert responses will be static throughout testing, so they will be vectorized and uploaded to a Chroma vector database separately. Once they are vectorized, they can be compared through cosine similarity to the LLM responses to evalute their effectiveness. 

### LLM Response Vectorization
The two LLM responses will be vectored so that they can be compared to the expert responses through cosine similarity to evaluate the similarity of the responses. 

### Responses Comparison
The LLM responses and expert responses will be compared through cosine similarity to evaluate the strength of the LLM responses to determine which one was superior. A score will then be applied to each LLM response and outputted in a CSV. 

## Usage

### Create Chroma Database
To create the Chroma vector database, starting off with the expert responses, run
```
python -m vector.create_database
```
Re-run that line to recreate the database with a fresh set of expert responses when necessary. 

### Add LLM Responses
To add the LLM responses to the Chroma vector database, run
```
python -m vector.add_llm_responses
```

### Compare Responses
To compare the LLM responses with the expert responses through cosine similiarity, run
```
python -m vector.compare_responses
```
The expert response and corresponding LLM responses, along with their similarity scores, will be outputted to your specified CSV file in the ```data``` directory. 