from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="audreyt/Taiwan-LLM-7B-v2.0.1-chat-GGUF",
	filename="Taiwan-LLM-7B-v2.0.1-chat-Q4_0.gguf",
)

llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)