from typing import Dict, Any, List
import json
import asyncio
from urllib.parse import urlparse
from utils.utils import _get_llm_response, _perform_search
class WebResearchAgent:
    def __init__(self,crawler,max_iterations,max_scrape_urls_per_iteration, logger):
        self.collected_info = []
        self.visited_urls = set()
        self.search_queries_history = []
        self.crawler = crawler
        self.logger = logger
        self.llm_enabled = True
        self.MAX_ITERATIONS = max_iterations
        self.MAX_SCRAPE_URLS_PER_ITERATION = max_scrape_urls_per_iteration

    async def _scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrapes the main text content (as Markdown) from a given URL using crawl4ai,
        respecting robots.txt.
        (Internal function for the agent class - Made Asynchronous)
        """
        self.logger.info(f"Attempting to scrape URL: {url}")


        # --- Crawl4ai Scraping (Asynchronous part) ---
        try:
            result = await self.crawler.arun(url=url)

            if result and result.markdown:
                # Basic check for minimal content length
                if len(result.markdown) < 50: # Adjust threshold as needed
                    self.logger.warning(f"crawl4ai extracted very little content ({len(result.markdown)} chars) from {url}. Treating as potentially empty.")
                    # Decide whether to return error or success with minimal content
                    # return {'status': 'error', 'content': None, 'error_message': 'Extracted content too short', 'url': url}
                    # Or let it pass through:
                    self.logger.info(f"Successfully scraped (short content) from {url}")
                    return {'status': 'success', 'content': result.markdown, 'error_message': None, 'url': url}
                else:
                    self.logger.info(f"Successfully scraped and extracted ~{len(result.markdown.fit_markdown)} chars (Markdown) from {url}")
                    return {'status': 'success', 'content': result.markdown, 'error_message': None, 'url': url}
            elif result:
                self.logger.warning(f"crawl4ai finished for {url} but returned no markdown content.")
                return {'status': 'error', 'content': None, 'error_message': 'No markdown content extracted by crawl4ai', 'url': url}
            else:
                # This case might occur if arun returns None or a falsy value unexpectedly
                self.logger.error(f"crawl4ai returned an unexpected result (None or empty) for {url}")
                return {'status': 'error', 'content': None, 'error_message': 'crawl4ai returned unexpected result', 'url': url}

        except Exception as e:
            # Catch potential errors from crawl4ai (timeouts, connection issues, parsing errors)
            self.logger.error(f"Error during crawl4ai execution for {url}: {e}", exc_info=True)
            error_message = f'crawl4ai failed: {type(e).__name__}: {e}'
            return {'status': 'error', 'content': None, 'error_message': error_message, 'url': url}


    def _analyze_query_llm(self, user_query: str) -> Dict[str, Any] | None:
        """Uses LLM for query analysis."""
        if not self.llm_enabled: return None
        prompt = f"""Analyze the following user query for web research. Identify intent, key entities/topics, information type (News, Fact, Summary, Comparison, etc.), and suggest 1-2 concise search engine query terms.
User Query: "{user_query}"
Output ONLY a valid JSON object with keys: "intent", "entities", "info_type", "search_terms" (list of strings).
Example: ```json\n{{"intent": "...", "entities": ["...", "..."], "info_type": "...", "search_terms": ["...", "..."]}}\n```"""
        response = _get_llm_response(prompt, temperature=0.3)
        if response and (response.strip().startswith('{') and response.strip().endswith('}') or "```json" in response):
            try:
                import re
                match = re.search(r'.*```json\s*\n*(.*?)\s*\n*```.*', response, re.DOTALL)
                json_str = match.group(1) if match else response
                analysis = json.loads(json_str.strip())
                if all(k in analysis for k in ["intent", "entities", "info_type", "search_terms"]) and isinstance(analysis["search_terms"], list):
                    self.logger.info(f"LLM Query Analysis Result: {analysis}")
                    return analysis
                else: self.logger.warning("LLM analysis JSON missing keys or invalid format.")
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM analysis JSON: {e}\nResponse: {response}")
        else: self.logger.warning(f"LLM analysis did not return valid JSON. Response: {response}")
        return None

    def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyzes query using LLM with fallback."""
        analysis = self._analyze_query_llm(user_query)
        if analysis: return analysis

        self.logger.info("LLM query analysis failed or disabled, falling back to basic keyword extraction.")
        return {
            "intent": "Unknown (LLM analysis failed/disabled)",
            "entities": [user_query],
            "info_type": "Unknown",
            "search_terms": [user_query] # Use raw query as search term
        }

    def _formulate_strategy(self, analysis: Dict[str, Any]) -> List[str]:
        """Decides initial search queries."""
        queries = analysis.get("search_terms", [])
        if not queries or not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
             queries = [term for term in analysis.get("entities", []) if term] # Use non-empty entities
             if not queries and analysis.get("intent"): queries = [analysis["intent"]] # Fallback to intent
             if not queries: queries = ["related information"] # Ultimate fallback

        # Limit query length if necessary and ensure non-empty
        queries = [q[:100].strip() for q in queries if q.strip()]
        if not queries: queries = ["related information"] # Ensure we always have a query

        self.logger.info(f"Formulated initial search queries: {queries}")
        self.search_queries_history.extend(q.lower() for q in queries) # Store lowercase history
        return queries

    def _analyze_content_llm(self, content: str, original_query: str, url: str) -> Dict[str, Any] | None:
        """Uses LLM for content analysis (content is now Markdown)."""
        if not self.llm_enabled: return None
        # Truncate content aggressively for the prompt - Markdown might be denser
        MAX_CONTENT_LEN = 5000 # Reduced slightly as Markdown can be verbose
        truncated_content = content[:MAX_CONTENT_LEN]
        if len(content) > MAX_CONTENT_LEN:
             truncated_content += "\n[... content truncated ...]"

        # Updated prompt to mention the content is Markdown
        prompt = f"""Analyze the relevance of the **Markdown formatted text** below (from {url}) to the query: "{original_query}"
Markdown Content:
--- START ---
{truncated_content}
--- END ---
Based ONLY on the text provided:
1. is_relevant: Is the text relevant? Answer strictly "YES" or "NO".
2. summary: If YES, summarize relevant info as needed. If NO, state "Not relevant".
3. key_points: If YES, list 2-4 key bullet points from the text related to the query. If NO, use an empty list [].
Output ONLY a valid JSON object with keys: "is_relevant", "summary", "key_points".
Example YES: {{"is_relevant": "YES", "summary": "...", "key_points": ["...", "..."]}}
Example NO: {{"is_relevant": "NO", "summary": "Not relevant", "key_points": []}}"""
        response = _get_llm_response(prompt, temperature=0.3)
        if response and '```json' in response:
            try:
                import re
                match = re.search(r'.*```json\s*\n*(.*?)\s*\n*```.*', response, re.DOTALL)
                json_str = match.group(1) if match else response
                analysis = json.loads(json_str.strip())
                if all(k in analysis for k in ["is_relevant", "summary", "key_points"]) and analysis["is_relevant"] in ["YES", "NO"] and isinstance(analysis["key_points"], list):
                    self.logger.info(f"LLM Content analysis for {url}: Relevant={analysis['is_relevant']}")
                    return analysis
                else: self.logger.warning(f"LLM content analysis JSON invalid format for {url}.")
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM content analysis JSON for {url}: {e}\nResponse: {response}")
        else: self.logger.warning(f"LLM content analysis did not return valid JSON for {url}. Response: {response}")
        return None

    def _analyze_scraped_content(self, content: str, original_query: str, url: str) -> Dict[str, Any] | None:
        """Analyzes content using LLM with fallback."""
        # Content is now Markdown from crawl4ai
        analysis = self._analyze_content_llm(content, original_query, url)
        if analysis: return analysis

        self.logger.warning(f"LLM content analysis failed or disabled for {url}. Marking as not relevant.")
        return {"is_relevant": "NO", "summary": "LLM analysis failed or disabled", "key_points": []}

    def _synthesize_results_llm(self, original_query: str) -> str | None:
        """Uses LLM for synthesis."""
        if not self.llm_enabled: return None
        if not self.collected_info: return None

        info_block = ""
        source_list = []
        for i, info in enumerate(self.collected_info, 1):
            # Content summary and key points are already text, generated by _analyze_content_llm
            info_block += f"--- Source [{i}] ({info['url']}) ---\nSummary: {info['content_summary']}\nKey Points:\n"
            info_block += "".join([f"- {point}\n" for point in info['key_points']]) + "\n"
            source_list.append(f"[{i}] {info['url']}")

        MAX_INFO_LEN = 8000 # Adjust based on LLM context limits
        if len(info_block) > MAX_INFO_LEN:
            info_block = info_block[:MAX_INFO_LEN] + "\n[... collected info truncated ...]"

        prompt = f"""Synthesize the collected information below to answer the query: "{original_query}"
Collected Information:
--- START INFO ---
{info_block}
--- END INFO ---
Instructions:
1. Create a coherent, structured answer based ONLY on the provided info.
2. Start with a direct summary/answer.
3. Integrate key points logically.
4. Explicitly state if sources conflict, citing them like [1], [2]. If not, mention they align.
5. Conclude with a brief overall summary.
6. Do NOT add external knowledge. Be objective.
7. Format clearly (paragraphs, bullets).
Generate the research report below (without listing sources here, they will be added later):
"""
        final_report = _get_llm_response(prompt, temperature=0.6)

        if final_report and not final_report.startswith("Error:"):
             return final_report
        else:
             self.logger.error(f"LLM Synthesis failed or returned error: {final_report}")
             return None

    def _synthesize_results(self, original_query: str) -> str:
        """Synthesizes results using LLM with fallback."""
        if not self.collected_info:
            return "No relevant information was found after searching and analyzing sources."

        final_report = self._synthesize_results_llm(original_query)

        if not final_report:
            self.logger.warning("LLM synthesis failed or disabled. Falling back to listing collected info.")
            report_parts = [f"LLM synthesis failed. Listing collected summaries:\n"]
            for i, info in enumerate(self.collected_info, 1):
                 report_parts.append(f"Source [{i}] ({info['url']}):\nSummary: {info['content_summary']}")
                 if info['key_points']:
                     report_parts.append("Key Points:")
                     report_parts.extend([f"- {p}" for p in info['key_points']])
                 report_parts.append("-" * 10)
            final_report = "\n".join(report_parts)

        source_list = [f"[{i}] {info['url']}" for i, info in enumerate(self.collected_info, 1)]
        final_report += "\n\n**Sources Used:**\n" + "\n".join(source_list)
        return final_report

    def _refine_search_queries_llm(self, original_query: str) -> List[str] | None:
        """(Optional) Uses LLM to suggest refined search queries."""
        if not self.llm_enabled or not self.collected_info: return None

        info_summary = "\n".join([f"- {info['url']}: {info['content_summary']}" for info in self.collected_info])
        history = ", ".join(self.search_queries_history)
        MAX_SUMMARY_LEN = 1500 # Reduced slightly
        if len(info_summary) > MAX_SUMMARY_LEN:
             info_summary = info_summary[:MAX_SUMMARY_LEN] + "\n[... summaries truncated ...]"

        prompt = f"""Based on the query "{original_query}", previous searches "{history}", and summaries found:
--- SUMMARIES ---
{info_summary}
--- END ---
Suggest 1-2 *new, different* search queries to find more details or missing info. Avoid repeating past searches.
Output ONLY a valid JSON list of strings (max 100 chars each). If no good new queries, output [].
Example: ["query phrase 1", "query phrase 2"] or []
"""
        response = _get_llm_response(prompt, temperature=0.5)

        if response and response.strip().startswith('[') and response.strip().endswith(']'):
            try:
                
                new_queries = json.loads(response.strip())
                if isinstance(new_queries, list) and all(isinstance(q, str) for q in new_queries):
                     # Filter out empty, duplicate (case-insensitive), and too long queries
                     refined_queries = []
                     seen_lower = set(self.search_queries_history)
                     for q in new_queries:
                          q_clean = q.strip()[:100]
                          if q_clean and q_clean.lower() not in seen_lower:
                               refined_queries.append(q_clean)
                               seen_lower.add(q_clean.lower()) # Add to seen set immediately

                     if refined_queries:
                          self.logger.info(f"LLM suggested refined search queries: {refined_queries}")
                          self.search_queries_history.extend(q.lower() for q in refined_queries) # Add new history
                          return refined_queries
                     else:
                          self.logger.info("LLM suggested no new queries or only repeated previous ones.")
                else: self.logger.warning("LLM refinement JSON was not a list of strings.")
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM query refinement JSON: {e}\nResponse: {response}")
        else: self.logger.warning(f"LLM refinement did not return valid JSON list. Response: {response}")
        return None

    # --- Main Execution Logic (Made Asynchronous) ---
    async def run_research(self, user_query: str) -> str:
        """Runs the full research process asynchronously."""
        self.logger.info(f"--- Starting Research for Query: '{user_query}' ---")
        if not self.llm_enabled:
            self.logger.warning("LLM is disabled. Research capabilities will be significantly limited.")

        self.collected_info = []
        self.visited_urls = set()
        self.search_queries_history = []

        # Initial query analysis and strategy (synchronous parts)
        analysis = self._analyze_query(user_query)
        current_queries = self._formulate_strategy(analysis)
        search_results_cache = []

        for iteration in range(self.MAX_ITERATIONS):
            self.logger.info(f"--- Research Iteration {iteration + 1}/{self.MAX_ITERATIONS} ---")

            # --- Search Phase (Synchronous) ---
            if current_queries:
                self.logger.info(f"Executing search for queries: {current_queries}")
                iteration_results = []
                for query in current_queries:
                    # Add delay between search API calls if needed by the search function
                    # The _perform_search using googlesearch-python has internal delays
                    if iteration_results: await asyncio.sleep(0.5) # Small async delay if real search tool
                    results = _perform_search(query, num_results=self.MAX_ITERATIONS)
                    iteration_results.extend(results)

                new_count = 0
                existing_links = {res.get('url') for res in search_results_cache}.union(self.visited_urls)
                for res in iteration_results:
                    link = res.get('url')
                    # Basic URL validation
                    if link and isinstance(link, str) and link.startswith(('http://', 'https://')):
                        parsed = urlparse(link)
                        if parsed.netloc and link not in existing_links: # Check domain exists
                            search_results_cache.append(res)
                            existing_links.add(link)
                            new_count += 1
                        elif link in existing_links:
                             self.logger.debug(f"Skipping duplicate/visited search result URL: {link}")
                        else:
                             self.logger.debug(f"Skipping invalid search result URL: {link}")
                    else:
                         self.logger.debug(f"Skipping invalid or missing URL in search result: {res}")

                self.logger.info(f"Found {new_count} new unique & valid URLs from search. Cache size: {len(search_results_cache)}")
                current_queries = [] # Clear queries after searching

            # --- URL Selection Phase (Synchronous) ---
            selected_for_scrape = []
            temp_cache = []
            while search_results_cache and len(selected_for_scrape) < self.MAX_SCRAPE_URLS_PER_ITERATION:
                 result = search_results_cache.pop(0)
                 link = result.get('url')
                 if link and link not in self.visited_urls:
                      selected_for_scrape.append(link)
                      self.visited_urls.add(link) # Mark as visited *before* scraping attempt
                 else:
                      temp_cache.append(result)
                      if link: self.logger.debug(f"Skipping already visited or invalid URL from cache: {link}")

            search_results_cache.extend(temp_cache)

            if not selected_for_scrape:
                self.logger.info("No new URLs selected for scraping in this iteration.")
                if iteration < self.MAX_ITERATIONS - 1:
                     self.logger.info("Attempting to refine search queries...")
                     # Refinement is synchronous LLM call
                     new_queries = self._refine_search_queries_llm(user_query)
                     if new_queries:
                          current_queries = new_queries
                          continue # Proceed to next iteration with new search
                     else:
                          self.logger.info("Query refinement failed or yielded no new queries. Ending search loop.")
                          break
                else:
                    self.logger.info("Max iterations reached or no URLs to scrape. Ending search loop.")
                    break

            # --- Scrape & Analyze Phase (Asynchronous Scraping) ---
            self.logger.info(f"Scraping and analyzing {len(selected_for_scrape)} URLs: {selected_for_scrape}")
            new_info_this_iteration = False
            scrape_tasks = []
            for url in selected_for_scrape:
                 # Create scrape tasks to run concurrently
                 scrape_tasks.append(self._scrape_url(url)) # Call async scrape function

            # Add politeness delay *before* gathering results if needed,
            # though crawl4ai might have its own internal delays.
            # A delay *between* starting tasks might be better. Let's add one here.
            if scrape_tasks:
                self.logger.debug(f"Waiting {len(scrape_tasks)*0.5}s before gathering scrape results (politeness)...")
                await asyncio.sleep(len(scrape_tasks)*0.5) # Adjust delay factor as needed

            # Run scraping tasks concurrently
            scrape_results = await asyncio.gather(*scrape_tasks)

            # Process results (Analysis is synchronous LLM call)
            for scrape_result in scrape_results:
                 if scrape_result['status'] == 'success' and scrape_result['content']:
                      # Content analysis is synchronous
                      content_analysis = self._analyze_scraped_content(scrape_result['content'], user_query, scrape_result['url'])
                      if content_analysis and content_analysis.get('is_relevant') == "YES":
                           self.logger.info(f"Relevant information found at {scrape_result['url']}")
                           self.collected_info.append({
                               'url': scrape_result['url'],
                               'content_summary': content_analysis['summary'],
                               'key_points': content_analysis['key_points']
                           })
                           new_info_this_iteration = True
                      elif content_analysis:
                           self.logger.info(f"Content from {scrape_result['url']} analyzed as not relevant.")
                      # else: Analysis failed (already logged)

                 elif scrape_result['status'] == 'error':
                      self.logger.warning(f"Scraping failed for {scrape_result['url']}: {scrape_result.get('error_message', 'Unknown error')}")
                 # else: Status was success but content was empty (already logged in scrape function)

            # --- Assess & Refine (End of Iteration - Synchronous) ---
            if not new_info_this_iteration:
                 self.logger.info("No new relevant information found from scraping this iteration.")
                 if iteration < self.MAX_ITERATIONS - 1 and not current_queries:
                     self.logger.info("Attempting query refinement...")
                     new_queries = self._refine_search_queries_llm(user_query)
                     if new_queries:
                          current_queries = new_queries
                     else:
                          self.logger.info("Query refinement failed or yielded no new queries.")
                          # Optional: break early if no new info AND no refinement

        # --- Synthesize Results (Synchronous) ---
        self.logger.info(f"--- Synthesizing Final Report ({len(self.collected_info)} relevant sources collected) ---")
        final_report = self._synthesize_results(user_query) # Synthesis is synchronous

        self.logger.info("--- Research Process Completed ---")
        return final_report