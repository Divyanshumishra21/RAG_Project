import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Iterator

class LLMGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Load the main language model with appropriate configuration.
        If the model fails to load, it falls back to a simpler default.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Ensure the tokenizer has a valid pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )

            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")

        except Exception as e:
            print(f"Failed to load model: {e}")
            self.load_fallback_model()

    def load_fallback_model(self):
        """
        Load a basic GPT-2 model in case the default model fails to load.
        """
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    def create_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Format the prompt to include the context and user query.
        """
        context = "\n\n".join([f"Context {i + 1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        prompt = f"""Based on the following context, please answer the question clearly and concisely.

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def generate_response(self, prompt: str, max_length: int = 512) -> Iterator[str]:
        """
        Generate a streaming response word-by-word.
        """
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response[len(prompt):].strip()

            for word in answer.split():
                yield word + " "

        except Exception as e:
            yield f"Error generating response: {str(e)}"

    def generate_simple_response(self, query: str, context_chunks: List[str]) -> str:
        """
        Fallback method to return rule-based responses based on keywords.
        """
        if not context_chunks:
            return "I couldn't find enough relevant information to answer that question."

        query_lower = query.lower()

        if "protect" in query_lower and "personal" in query_lower:
            return self._generate_protection_response(context_chunks)
        elif "data" in query_lower and "collect" in query_lower:
            return self._generate_data_collection_response(context_chunks)
        elif "payment" in query_lower or "refund" in query_lower:
            return self._generate_payment_response(context_chunks)
        elif "terminate" in query_lower or "account" in query_lower:
            return self._generate_termination_response(context_chunks)
        else:
            return self._generate_general_response(query, context_chunks)

    def _generate_protection_response(self, chunks: List[str]) -> str:
        """
        Return a response about how personal data is protected.
        """
        response = "Based on the terms, here is how personal information is protected:\n\n"

        for chunk in chunks:
            if "security measures" in chunk.lower() or "protect" in chunk.lower():
                response += (
                    "We apply reasonable security practices to protect your data. "
                    "However, no method of transmission over the internet is completely secure. "
                    "Users are advised to maintain strong credentials and report any suspicious activity."
                )
                break

        return response

    def _generate_data_collection_response(self, chunks: List[str]) -> str:
        """
        Return a response about what data is collected.
        """
        return (
            "We collect user data including name, email, contact number, and payment details where applicable. "
            "This information is used to provide services, handle transactions, respond to inquiries, "
            "and improve platform performance."
        )

    def _generate_payment_response(self, chunks: List[str]) -> str:
        """
        Return a response about payment and refund terms.
        """
        return (
            "Service fees are displayed on the platform and are subject to change with prior notice. "
            "Payments are securely handled by third-party providers. Refunds, if applicable, are processed "
            "according to our policy and within a fixed timeframe."
        )

    def _generate_termination_response(self, chunks: List[str]) -> str:
        """
        Return a response about account termination policies.
        """
        return (
            "Users can request account termination at any time. "
            "We reserve the right to suspend or terminate accounts that violate terms or engage in misuse. "
            "Termination may occur with or without notice depending on the severity of the breach."
        )

    def _generate_general_response(self, query: str, chunks: List[str]) -> str:
        """
        Provide a general fallback answer using the most relevant chunk.
        """
        response = "Here is some information related to your question:\n\n"

        if chunks:
            response += f"{chunks[0]}\n\n"
            if len(chunks) > 1:
                response += f"Additional detail:\n{chunks[1][:200]}..."

        return response
