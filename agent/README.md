# Agentic Method

## Functionality

### LLM Comparison
An OpenAI agent will evaluate the similarity of the two LLM responses against the expert response and give a summary of which one it finds most similar and why. 

## Usage

### Agentic Comparison
To run the AI agent-led comparison of responses, run
```
python -m agent.compare_with_llm
```
The expert response and corresponding LLM responses, along with the AI agent's decision and reasoning, will be outputted in your specified CSV file in the ```data``` directory. 