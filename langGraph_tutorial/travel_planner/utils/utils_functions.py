from langchain_openai import ChatOpenAI
# from langchain_ollama import Ollama

import os
import re

from dotenv import load_dotenv
load_dotenv()


# Initialize the LLM for parsing user input
def get_llm():
    """Get OpenAI LLM instance with API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("   Set it using: export OPENAI_API_KEY='your-api-key-here'")
        return None
    return ChatOpenAI(model="gpt-4o", temperature=0.1)

# def get_ollama_llm():
#     """Get Ollama LLM instance"""
#     return Ollama(model="llama3.2:1b  ", temperature=0.1)

def extract_with_regex(text: str) -> dict:
    """Fallback regex-based extraction for when LLM is not available"""
    extracted = {}
    
    # Extract dates (various formats)
    import re
    from datetime import datetime
    
    # Look for month names with dates
    month_patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})'
    ]
    
    # Look for numeric dates
    numeric_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY or M/D/YYYY
        r'(\d{1,2}-\d{1,2}-\d{4})',  # MM-DD-YYYY
    ]
    
    dates = []
    
    # Try month name patterns first
    for pattern in month_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                month_name, day, year = match
                # Convert month name to number
                month_map = {
                    'january': '01', 'jan': '01', 'february': '02', 'feb': '02',
                    'march': '03', 'mar': '03', 'april': '04', 'apr': '04',
                    'may': '05', 'june': '06', 'jun': '06', 'july': '07', 'jul': '07',
                    'august': '08', 'aug': '08', 'september': '09', 'sep': '09',
                    'october': '10', 'oct': '10', 'november': '11', 'nov': '11',
                    'december': '12', 'dec': '12'
                }
                month_num = month_map.get(month_name.lower(), '01')
                date_str = f"{year}-{month_num}-{day.zfill(2)}"
                dates.append(date_str)
            except:
                continue
    
    # If no month name dates found, try numeric patterns
    if not dates:
        for pattern in numeric_patterns:
            dates.extend(re.findall(pattern, text))
    
    # Extract budget - look for dollar amounts
    budget_patterns = [
        r'\$(\d{1,3}(?:,\d{3})*)',  # $1,000 or $5000
        r'budget.*?(\d{1,3}(?:,\d{3})*)',  # budget of 1000
        r'(\d{1,3}(?:,\d{3})*)\s*dollars?',  # 1000 dollars
    ]
    
    budget = 2000.0  # default
    for pattern in budget_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            budget = float(match.group(1).replace(',', ''))
            break
    extracted['budget'] = budget
    
    # Extract number of travelers
    traveler_patterns = [
        r'(\d+)\s+(?:people|person|traveler|passenger)',
        r'for\s+(\d+)',
        r'(\d+)\s+of\s+us'
    ]
    
    travelers = 1
    for pattern in traveler_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            travelers = int(match.group(1))
            break
    extracted['travelers'] = travelers
    
    # Extract cities (simple approach - look for capitalized words)
    cities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Assume first city is origin, second is destination
    if len(cities) >= 2:
        extracted['origin'] = cities[0]
        extracted['destination'] = cities[1]
    elif len(cities) == 1:
        extracted['destination'] = cities[0]
        extracted['origin'] = "New York"  # Default
    else:
        extracted['origin'] = "New York"
        extracted['destination'] = "Paris"
    
    # Set dates
    if len(dates) >= 2:
        extracted['departure_date'] = dates[0]
        extracted['return_date'] = dates[1]
    else:
        # Default dates
        from datetime import datetime, timedelta
        today = datetime.now()
        extracted['departure_date'] = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        extracted['return_date'] = (today + timedelta(days=37)).strftime("%Y-%m-%d")
    
    # Trip type detection
    business_keywords = ['business', 'work', 'conference', 'meeting']
    leisure_keywords = ['vacation', 'holiday', 'leisure', 'fun', 'relax']
    family_keywords = ['family', 'kids', 'children']
    adventure_keywords = ['adventure', 'hiking', 'skiing', 'diving']
    
    text_lower = text.lower()
    if any(word in text_lower for word in business_keywords):
        extracted['trip_type'] = 'business'
    elif any(word in text_lower for word in family_keywords):
        extracted['trip_type'] = 'family'
    elif any(word in text_lower for word in adventure_keywords):
        extracted['trip_type'] = 'adventure'
    else:
        extracted['trip_type'] = 'leisure'
    
    # Extract preferences
    preference_keywords = {
        'museums': ['museum', 'art', 'gallery', 'culture'],
        'local cuisine': ['food', 'restaurant', 'cuisine', 'dining', 'eat'],
        'history': ['history', 'historical', 'ancient', 'heritage'],
        'nature': ['nature', 'park', 'hiking', 'outdoor'],
        'nightlife': ['nightlife', 'bar', 'club', 'party'],
        'shopping': ['shopping', 'shop', 'market', 'boutique'],
        'architecture': ['architecture', 'building', 'cathedral', 'church']
    }
    
    preferences = []
    for pref, keywords in preference_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            preferences.append(pref)
    
    if not preferences:
        preferences = ['local cuisine', 'history', 'museums']  # Default
    
    extracted['preferences'] = preferences
    
    return extracted
