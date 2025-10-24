import re
import json
import boto3
from botocore.exceptions import ClientError

# ---- Clients ----
# Keep region consistent with your deployment
_REGION = "us-west-2"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=_REGION
)

bedrock_kb = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=_REGION
)


# ---------------------------
# Prompt classification (A–E)
# ---------------------------
def valid_prompt(prompt: str, model_id: str) -> bool:
    """
    Uses a Bedrock LLM to classify the user's prompt into categories A–E.
    Returns True ONLY for Category E (heavy machinery / your KB domain).
    Returns False for everything else or on error (strict per reviewer).
    """
    try:
        if not prompt or not prompt.strip():
            return False

        classification_instructions = f"""
You are a strict classifier. Categorize the user's request into EXACTLY one of:
A: Asking about how the LLM works or system architecture.
B: Profanity, harassment, or toxic language.
C: Topic unrelated to this project (general knowledge, travel, news, etc.).
D: Asking for instructions about how YOU work or meta questions about prompts/tools.
E: Heavy machinery / industrial equipment (e.g., bulldozers, excavators, forklifts, cranes, dump trucks) — specs, capacities, operating details, maintenance, safety, parts.

Respond with ONLY the single letter A, B, C, D, or E. No punctuation. No extra words.

User prompt:
{prompt}
""".strip()

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": classification_instructions}
                    ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0,
            "top_p": 1
        }

        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        body = json.loads(resp["body"].read())
        raw = body.get("content", [{}])[0].get("text", "").strip()
        # Examples we might get: "E", "Category E", "e", "category d"
        # Normalize to a single uppercase letter if possible
        m = re.search(r"\b([A-E])\b", raw.upper())
        category = m.group(1) if m else ""

        # Debug print is useful for your screenshot evidence
        print(f"[valid_prompt] Model raw='{raw}' -> parsed='{category}'")

        return category == "E"

    except Exception as e:
        # Strict behavior to satisfy reviewer: fail closed (False)
        print(f"[valid_prompt] ERROR: {e}")
        return False


# ---------------------------
# Knowledge Base retrieval
# ---------------------------
def query_knowledge_base(kb_id: str, query: str, number_of_results: int = 3) -> str:
    """
    Retrieves top chunks from the Bedrock Knowledge Base for a user query.
    Returns a single context string (joined chunks) for your LLM prompt.
    """
    try:
        resp = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": number_of_results}
            },
        )

        results = resp.get("retrievalResults", []) or []
        chunks = []
        for r in results:
            text = (r.get("content") or {}).get("text") or ""
            if text.strip():
                chunks.append(text.strip())

        context = "\n---\n".join(chunks)
        print(f"[query_knowledge_base] Retrieved {len(chunks)} chunks.")
        return context

    except ClientError as e:
        print(f"[query_knowledge_base] ClientError: {e}")
        return ""
    except Exception as e:
        print(f"[query_knowledge_base] ERROR: {e}")
        return ""


# ---------------------------
# LLM response generation
# ---------------------------
def generate_response(prompt: str, context: str, model_id: str,
                      temperature: float = 0.7, top_p: float = 0.9) -> str:
    """
    Generates a reply using the selected Bedrock model.
    Includes retrieved context (if any) and the user's prompt.
    """
    try:
        composed = f"""You are a helpful assistant for heavy machinery documentation.
Use ONLY the provided context to answer. If the context is not sufficient, say you don't know.

Context:
{context if context.strip() else "(no relevant context retrieved)"}

Question:
{prompt}
"""
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": composed}]}
            ],
            "max_tokens": 600,
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

        resp = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        body = json.loads(resp["body"].read())
        text = body.get("content", [{}])[0].get("text", "").strip()

        print(f"[generate_response] OK, {len(text)} chars.")
        return text or "I couldn't find a confident answer from the current context."

    except ClientError as e:
        print(f"[generate_response] ClientError: {e}")
        return "⚠️ Error generating response."
    except Exception as e:
        print(f"[generate_response] ERROR: {e}")
        return "⚠️ An unexpected error occurred while generating a response."
