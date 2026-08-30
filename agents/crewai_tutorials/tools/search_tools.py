import json
import requests
from crewai.tools import tool
from dotenv import load_dotenv
import os
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
print(f"SERPER_API_KEY: {SERPER_API_KEY}")

class SearchTools():
    @tool
    def search_internet(query: str):
        """Helps search the internet for a given topic and retrieve relevant results."""
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})

        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.json())
        if "organic" not in response.json():
            return """Apologies, I couldn't locate any results for that query. 
                      The problem might be with your Serper API key."""
        else:
            results = response.json()['organic']
            string = []
            for result in results[:top_result_to_return]:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}", f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}", "\n-----------------"
                    ]))
                except KeyError:
                    continue

            return "\n".join(string)



        


        