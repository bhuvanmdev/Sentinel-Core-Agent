from typing import List, Dict, Any
import time
import google.generativeai as genai
from googlesearch import search

def _get_llm_response(prompt: str, temperature: float = 0.7) -> str | None:
    """
    Sends a prompt to the configured Gemini LLM and returns the response text.
    (Internal function for the agent class - Remains Synchronous)
    TODO: Add other model support too.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        

        # Note: genai library calls might be blocking/synchronous.
        # If running the agent within an async framework, consider using an async Gemini client if available
        # or running this blocking call in a separate thread (e.g., using asyncio.to_thread).
        # For simplicity here, we keep it synchronous.
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
            )
        )

        if response.parts:
             response_text = "".join(part.text for part in response.parts)
             
             return response_text
        else:
             if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'content') and response.candidates[0].content.parts:
                 response_text = "".join(part.text for part in response.candidates[0].content.parts)
                 
                 return response_text
             else:
                 
                 return "Error: Failed to extract text from LLM response."

    except Exception as e:
        
        error_detail = str(e)
        # Add more specific error details if possible
        return f"Error: Failed to get response from LLM. Details: {error_detail}"


# --- Web Search Tool Functionality ---

SEARCH_TOOL_AVAILABLE = True

def _perform_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """ Performs web search using googlesearch-python (synchronous). """
    
    try:
        results_iterator = search(query, num_results=num_results) # Add pause
        formatted_results = []
        count = 0
        for url in results_iterator:
                # The basic library only gives URLs. We'll make do without title/snippet for now.
                # A more robust solution would fetch title/snippet separately or use an API.
                formatted_results.append({
                    'title': f"Result for '{query}'", # Placeholder title
                    'url': url,
                    'description': f"URL found: {url}" # Placeholder description
                })
                count += 1
                if count >= num_results:
                    break
                time.sleep(0.2) # Small delay between results processing

        if not formatted_results:
                
                return []

        
        return formatted_results
    except Exception as e:
        error_detail = str(e)
        # Add more specific error details if possible
        return f"Error: Failed to perform search. Details: {error_detail}"
