import boto3
import json
import time
import re
import ast

class LLMInterface:
    def __init__(self, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
        self.model_id = model_id
        self.bedrock = boto3.client("bedrock-runtime")
        
    def extract_waypoints(self, text):
        """Extract waypoints from LLM response"""
        text = text.replace("`", "")
        # Try to find JSON format first
        match = re.search(r"###OUTPUT_START###(.*?)###OUTPUT_END###", text, re.DOTALL)
        if match:
            snippet = match.group(1).strip()
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        # Fallback to list format
        list_match = re.search(r"\[\s*\[[^\[\]]+\](?:\s*,\s*\[[^\[\]]+\])*\s*\]", text)
        if list_match:
            snippet = list_match.group(0)
            try:
                return ast.literal_eval(snippet)
            except Exception:
                return None
        return None

    def query_llm(self, prompt, temperature=0.3, max_tokens=800):
        """Query the LLM with a prompt"""
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        })
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]