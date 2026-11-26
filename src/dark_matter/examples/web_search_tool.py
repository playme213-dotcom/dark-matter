"""Web search tool example.

Demonstrates adding web search capability to Dark Matter agents
using DuckDuckGo's lightweight HTML endpoint (no API key required).
"""

from dark_matter import Agent, Runner, function_tool
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote


@function_tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information using DuckDuckGo.

    Args:
        query: The search query.
        num_results: Maximum number of results to return (default 5).
    """
    try:
        url = f"https://duckduckgo.com/html/?q={quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.select(".result")[:num_results]:
            title_el = result.select_one(".result__a")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one(".result__a")

            if title_el and link_el:
                title = title_el.get_text(strip=True)
                link = link_el.get("href", "")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append(f"• {title}\n  {snippet}\n  {link}\n")

        if not results:
            return "No search results found."

        return f"Web search results for '{query}':\n\n" + "\n".join(results)

    except requests.RequestException as e:
        return f"Search error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


web_agent = Agent(
    name="Web Search Agent",
    instructions=(
        "You are a research assistant with web search capability. "
        "When users ask about current events, facts, or topics you're unsure about, "
        "use the web_search tool to find up-to-date information. "
        "Always cite your sources by mentioning the website names."
    ),
    tools=[web_search],
)


def main():
    print("=== Dark Matter Web Search Agent ===")
    print("Tool: web_search (DuckDuckGo)")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        result = Runner.run_sync(web_agent, user_input)
        print(f"Agent: {result.final_output}\n")


if __name__ == "__main__":
    main()
