# CustomGPT-RLHF
RLHF for a CustomGPT from OpenAI's CustomGPTs. 

## Purpose
Since Custom GPTs cannot be fine-tuned or called through an API, this RLHF system can't fine-tune the model. The 'Reinforcement Learning' part of RLHF would have to be done through manual changes to the Custom GPT prompt. However, this system is useful in evaluating if the Custom GPT is heading in the right direction, and can then be augmented manually. 

## Methods

### Vectorization
The directory ```vector``` contains the vectorization method of RLHF, where two LLM responses are evaluated in their similarity against an expert response through cosine similarity of their vectors. 

### Agentic Judge
The directory ```agent``` contains the agentic method of RLHF, where two LLM responses are evaulated in their similarity agains an expert response through a separate LLM agent acting as a judge. 

