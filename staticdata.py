"""staticdata.py

Uses OpenAI's Chat Completion API to fetch swimmer biometrics and Paris-2024 race list.

For each swimmer name in the input file we ask ChatGPT to respond with a short JSON
object containing:
    {
      "name": "...",           # as given
      "dob": "YYYY-MM-DD or YYYY",  # best guess
      "height_cm": <number|null>,
      "weight_kg": <number|null>,
      "gender": "male / female / other / unknown",
      "ethnicity": "...",      # optional string or null
      "paris_2024_races": ["Men's 50m Freestyle", ...]
    }

Results are saved to swimmer_biometrics.json (or path you pass).

Prerequisites:
  pip install openai tenacity python-dotenv
  export OPENAI_API_KEY="your-key"

Usage:
  python biometricsGPTscraper.py [input_file] [output_file]2
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from time import sleep
from pathlib import Path
from typing import Dict, List

import openai
from dotenv import load_dotenv

# Load env vars first
load_dotenv("secrets.env")  # load environment variables

# Configure OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    sys.exit("OPENAI_API_KEY environment variable not set. Add it to secrets.env")

openai.api_key = OPENAI_KEY

DEFAULT_INPUT = "olympic_swimmer_names.txt"
DEFAULT_OUTPUT = "swimmer_biometrics.json"

SYSTEM_PROMPT = (
    "You are an expert Olympic swimming database with comprehensive knowledge of all swimmers. "
    "You have detailed biographical and competition data for Olympic athletes. "
    "Provide complete, accurate information whenever possible. Only use null if data is truly unknown. "
    "Output STRICTLY as valid JSON with no markdown formatting or commentary."
)

def create_prompt(name: str, attempt: int = 0, missing_fields: list = None) -> str:
    base_prompt = (
        f"You are accessing a comprehensive Olympic swimming database. Swimmer: {name}\n\n"
        f"MANDATORY: Fill ALL fields with real data. This swimmer competed in major competitions.\n"
        f"EXTRACT THESE EXACT FIELDS:\n"
        f"1. DATE OF BIRTH: Find exact birth date (YYYY-MM-DD format)\n"
        f"2. HEIGHT: Physical height in centimeters (e.g., 188, 175, 192)\n"
        f"3. WEIGHT: Body weight in kilograms (e.g., 78, 65, 85)\n"
        f"4. GENDER: male or female\n"
        f"5. ETHNICITY/NATIONALITY: Country or ethnic background\n"
        f"6. PARIS 2024 EVENTS: Complete list of all swimming events at Paris Olympics\n\n"
    )
    if attempt == 0:
        base_prompt += (
            f"EXAMPLES of complete data:\n"
            f"- Katie Ledecky: born 1997-03-17, 183cm, 70kg, female, American, [multiple events]\n"
            f"- Caeleb Dressel: born 1996-08-16, 191cm, 88kg, male, American, [sprint events]\n\n"
            f"Research {name} thoroughly and provide ALL biographical data.\n"
            f"If you cannot find a field, make a best-educated guess based on public records, typical values for elite swimmers, or official Olympic data. Only use null if absolutely no information is available anywhere.\n"
        )
    else:
        base_prompt += (
            f"CRITICAL: Previous attempt returned incomplete data for {name}.\n"
            f"Missing fields: {', '.join(missing_fields) if missing_fields else 'unknown'}\n"
            f"This is unacceptable - {name} is a known Olympic swimmer with public biographical data.\n"
            f"Search more thoroughly. Height/weight are in official team rosters and competition records.\n"
            f"Birth dates are in athlete profiles. Competition history is in Olympic records.\n"
            f"DO NOT USE NULL - find the actual data or make a best-educated guess.\n"
        )
    base_prompt += (
        f"Return ONLY valid JSON:\n"
        f'{{"name": "{name}", "dob": "YYYY-MM-DD", "height_cm": NUMBER, "weight_kg": NUMBER, '
        f'"gender": "male/female", "ethnicity": "nationality", "paris_2024_races": ["event1", "event2"]}}'
    )
    return base_prompt


def query_openai(name: str) -> dict:
    required_fields = ['name', 'dob', 'height_cm', 'weight_kg', 'gender', 'ethnicity', 'paris_2024_races']
    missing_fields = None
    for attempt in range(3):
        time.sleep(1.2 + random.uniform(-0.2, 0.5))
        prompt = create_prompt(name, attempt, missing_fields if attempt > 0 else None)
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=700,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            data = json.loads(content)
            missing_fields = [field for field in required_fields if field not in data or data[field] in [None, "", []]]
            if len(missing_fields) == 0:
                return data
            if len(missing_fields) <= 2 or attempt == 2:
                if missing_fields and attempt < 2:
                    print(f"  (retry {attempt + 1}: missing {missing_fields})")
                    time.sleep(1)
                    continue
                # Warn if still missing fields after all retries
                if missing_fields:
                    print(f"  [!] WARNING: Missing fields for {name} after all retries: {missing_fields}")
                return data
            print(f"  (retry {attempt + 1}: {len(missing_fields)} missing fields)")
            time.sleep(1)
        except json.JSONDecodeError as e:
            if attempt == 2:
                print(f"  JSON parse error: {str(e)[:50]}")
                return {"name": name, "error": f"JSON parse failed: {str(e)}"}
            time.sleep(1)
        except Exception as e:
            if attempt == 2:
                print(f"  API error: {str(e)[:50]}")
                return {"name": name, "error": f"API error: {str(e)}"}
            time.sleep(1)
    return {"name": name, "error": "Max retries exceeded"}


def main(inp: str = DEFAULT_INPUT, out: str = DEFAULT_OUTPUT) -> List[Dict]:
    """Process swimmers from input file and save results to output file.
    
    Args:
        inp: Path to input file with one swimmer name per line
        out: Path to output JSON file for results
        
    Returns:
        List of swimmer data dictionaries
    """
    # Load existing results if they exist
    results = []
    if os.path.exists(out):
        try:
            with open(out, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {out}")
        except json.JSONDecodeError:
            print(f"Warning: Could not read existing {out}, starting fresh")
    # Get list of already processed names (case-insensitive)
    processed_names = {r['name'].lower() for r in results if 'name' in r}
    # Read and filter swimmers
    swimmers = Path(inp).read_text().splitlines()
    swimmers = [s.strip() for s in swimmers if s.strip()]
    # Remove already processed swimmers
    new_swimmers = [s for s in swimmers if s.lower() not in processed_names]
    print(f"Processing {len(new_swimmers)} new swimmers (skipping {len(swimmers) - len(new_swimmers)} already processed)")
    if not new_swimmers:
        print("No new swimmers to process.")
        return results
    for idx, name in enumerate(swimmers, 1):
        print(f"[{idx}/{len(swimmers)}] 🔍 {name} …", end=" ")
        try:
            data = query_openai(name)
        except Exception as e:
            print(f"⚠️  error: {e}")
            data = {"name": name, "error": str(e)}
        results.append(data)
        print("done")
        time.sleep(1.2)  # mild pacing to avoid rate limits
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved {len(results)} records to {out}")
    return results


if __name__ == "__main__":
    inp_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    main(inp_file, out_file)
