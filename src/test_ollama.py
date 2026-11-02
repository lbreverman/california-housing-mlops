from ollama import chat

print("Connecting to Llama 3.2...")

response = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain in two sentences how machine learning can predict California housing prices."
        }
    ]
)

print("\nAI Assistant Response:")
print(response.message.content)