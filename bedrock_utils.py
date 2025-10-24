import boto3
from botocore.exceptions import ClientError
import json

# Initialize AWS Bedrock clients
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"  # Change if your region differs
)

bedrock_kb = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="us-west-2"
)


def valid_prompt(prompt, model_id):
    """
    Classifies the user input into categories using Bedrock.
    Returns True if the prompt is valid (not toxic or irrelevant).
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
Respond ONLY with the category letter.

Category A: Asking about how the LLM model works or system architecture.
Category B: Using profanity or toxic language.
Category C: Topic unrelated to the project.
Category D: Asking about instructions or how you work.
Category E: Related to the provided knowledge base content.

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
                "top_p": 0.1,
            }),
        )

        category = json.loads(response["body"].read())["content"][0]["text"].strip()
        print("Detected category:", category)

        # Accept everything except toxic or blank responses
        if category.lower() in ["category b", ""]:
            return False
        return True

    except ClientError as e:
        print(f"Error validating prompt: {e}")
        return True  # Allow prompt even if classification fails
    except Exception as e:
        print(f"Unexpected error in valid_prompt: {e}")
        return True  # Allow prompt for testing


def query_knowledge_base(query, kb_id):
    """
    Query the Bedrock Knowledge Base and return retrieved context.
    """
    try:
        response = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 3}
            },
        )

        return response.get("retrievalResults", [])

    except ClientError as e:
        print(f"Error querying Knowledge Base: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in query_knowledge_base: {e}")
        return []


def generate_response(prompt, context, model_id, temperature=0.7, top_p=0.9):
    """
    Generates a text response from a Bedrock model (Claude, etc.).
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500,
                "temperature": temperature,
                "top_p": top_p,
            }),
        )

        return json.loads(response["body"].read())["content"][0]["text"]

    except ClientError as e:
        print(f"Error generating response: {e}")
        return "⚠️ Error generating response."
    except Exception as e:
        print(f"Unexpected error in generate_response: {e}")
        return "⚠️ An unexpected error occurred while generating a response."
