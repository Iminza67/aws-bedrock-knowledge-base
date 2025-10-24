def valid_prompt(prompt, model_id):
    """
    Uses Bedrock to classify the user input into categories A–E.
    Returns True ONLY if the prompt is Category E (related to heavy machinery
    or knowledge base content). Otherwise returns False.
    """
    try:
        if not prompt.strip():
            return False

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Human: Classify the provided user request into one of the following categories.
Respond ONLY with the category letter (A, B, C, D, or E).

Category A: Asking about how the LLM model works or system architecture.
Category B: Using profanity or toxic language.
Category C: Topic unrelated to the project.
Category D: Asking about instructions or how you work.
Category E: Related to the provided knowledge base content (heavy machinery, industrial vehicles, or equipment specifications).

<user_request>
{prompt}
</user_request>

Assistant:
"""
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 1
            }),
        )

        result_text = json.loads(response["body"].read())["content"][0]["text"].strip()
        print("Detected category:", result_text)

        # Normalize and extract just the category letter
        category = result_text.upper().replace("CATEGORY", "").strip()

        # ✅ Only Category E is valid
        return category == "E"

    except ClientError as e:
        print(f"Error validating prompt: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in valid_prompt: {e}")
        return False
