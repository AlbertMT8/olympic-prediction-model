import asyncio
import re
import json
from datetime import datetime
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import os
import logging
import sys

# Configuration
# Default input file containing swimmer names (one per line)
INPUT_FILE = "olympic_swimmer_names.txt"  # can be overridden via CLI arg
JSON_OUTPUT_FILE = "olympic_swimmers_youth_times.json"
# Delay configuration – tuned for safety vs. speed
# Normal requests wait 1-3 s; every ~25 requests we insert a longer pause.
MIN_DELAY = 1.0  # seconds
MAX_DELAY = 3.0  # seconds
LONG_DELAY_MIN = 8.0  # seconds
LONG_DELAY_MAX = 12.0  # seconds
LONG_DELAY_FREQUENCY = 25  # after this many requests
TIMEOUT = 15000
PARIS_OLYMPICS_IDENTIFIER = "Olympic Games"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouthTimesSwimCloudScraper:
    """Scraper for youth swimming times of Olympic swimmers from SwimCloud."""
    
    def __init__(self):
        self.all_scraped_race_records: List[Dict] = []
        self.processed_swimmers = 0
        self.total_swimmers = 0
        self.failed_swimmers: List[str] = []
        # Track how many network-heavy actions we have triggered to occasionally
        # inject a longer pause for safety.
        self.request_count = 0
    
    async def polite_delay(self, min_sec: float = MIN_DELAY, max_sec: float = MAX_DELAY) -> None:
        """Adaptive delay: small jitter for most requests, longer pause every N calls."""
        import random
        self.request_count += 1
        delay = random.uniform(min_sec, max_sec)
        # After every LONG_DELAY_FREQUENCY requests, add an extended pause.
        if self.request_count % LONG_DELAY_FREQUENCY == 0:
            delay += random.uniform(LONG_DELAY_MIN, LONG_DELAY_MAX)
            logger.info(f"Safety cooldown: extra long delay of {delay:.1f}s after {self.request_count} requests")
        await asyncio.sleep(delay)
    
    def load_swimmer_names(self) -> List[str]:
        """Load swimmer names from the input text file."""
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(names)} swimmer names from {INPUT_FILE}")
            return names
        except FileNotFoundError:
            logger.error(f"Input file {INPUT_FILE} not found!")
            return []
        except Exception as e:
            logger.error(f"Error loading swimmer names: {e}")
            return []
    
    def parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """Convert time string to seconds (handles MM:SS.ss, SS.ss, H:MM:SS.ss formats)."""
        if not time_str or time_str.strip() == '':
            return None
        
        time_str = time_str.strip()
        
        try:
            # Handle H:MM:SS.ss format
            if time_str.count(':') == 2:
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            
            # Handle MM:SS.ss format
            elif ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            
            # Handle SS.ss format
            else:
                return float(time_str)
        
        except (ValueError, IndexError):
            logger.warning(f"Could not parse time: {time_str}")
            return None
    
    def extract_age_from_date(self, birth_date: str, race_date: str) -> Optional[int]:
        """Calculate age at race from birth date and race date."""
        try:
            # This is a simplified implementation - you may need to adjust date formats
            birth = datetime.strptime(birth_date, '%Y-%m-%d')
            race = datetime.strptime(race_date, '%Y-%m-%d')
            age = race.year - birth.year - ((race.month, race.day) < (birth.month, birth.day))
            return age
        except (ValueError, TypeError):
            return None
    
    async def search_swimmer(self, page, swimmer_name: str) -> List[str]:
        """Search for a swimmer using interactive JavaScript search and return list of potential profile URLs."""
        try:
            # Navigate to SwimCloud homepage
            await page.goto("https://www.swimcloud.com/", timeout=TIMEOUT)
            await self.polite_delay()
            
            # Get page content to check for access issues
            content = await page.content()
            if '403 Forbidden' in content or 'Access Denied' in content:
                logger.warning(f"Access denied for search: {swimmer_name}")
                return []
            
            # Find and interact with the global search input
            search_input = await page.query_selector('#global-search-select')
            if not search_input:
                logger.warning(f"Could not find search input for {swimmer_name}")
                return []
            
            # Clear and type in the search input
            await search_input.click()
            await search_input.fill('')
            await search_input.type(swimmer_name, delay=100)
            
            # Wait for search suggestions to appear
            await asyncio.sleep(2)
            
            # Check for search suggestions/dropdown
            suggestions = await page.query_selector_all('[class*="menu"] [role="option"], [class*="option"]')
            
            profile_links = []
            
            if suggestions:
                # Look for swimmer suggestions (not login/register options)
                for suggestion in suggestions:
                    try:
                        text = await suggestion.text_content()
                        if text and 'Swimmers' in text and swimmer_name.lower() in text.lower():
                            # Click on the swimmer suggestion
                            await suggestion.click()
                            await asyncio.sleep(3)
                            
                            # Check if we were redirected to a swimmer profile
                            current_url = page.url
                            if '/swimmer/' in current_url:
                                profile_links.append(current_url)
                                logger.info(f"Found direct profile for {swimmer_name}: {current_url}")
                                break
                    except Exception as e:
                        logger.debug(f"Error processing suggestion: {e}")
                        continue
            
            # If no direct match, try pressing Enter to search
            if not profile_links:
                await search_input.press('Enter')
                await asyncio.sleep(3)
                
                # Check if we were redirected to a search results page or swimmer profile
                current_url = page.url
                if '/swimmer/' in current_url:
                    profile_links.append(current_url)
                    logger.info(f"Found profile via Enter search for {swimmer_name}: {current_url}")
                else:
                    # Parse search results page for swimmer links
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        if href and '/swimmer/' in href and href.count('/') >= 3:
                            if href.startswith('/swimmer/'):
                                full_url = f"https://www.swimcloud.com{href}"
                                if full_url not in profile_links:
                                    profile_links.append(full_url)
            
            # Remove duplicates and limit results
            profile_links = list(set(profile_links))[:5]  # Limit to 5 profiles max
            logger.info(f"Found {len(profile_links)} potential profiles for {swimmer_name}")
            
            return profile_links
            
        except Exception as e:
            logger.error(f"Error searching for swimmer {swimmer_name}: {e}")
            return []
    
    async def verify_olympic_swimmer(self, page, profile_url: str) -> bool:
        """Verify if swimmer participated in Paris Olympics 2024."""
        try:
            await page.goto(profile_url, timeout=TIMEOUT)
            await self.polite_delay()
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Look for Olympic Games and 2024 dates to verify Paris Olympics participation
            page_text = soup.get_text().lower()
            
            # Check for Olympic Games presence
            has_olympic_games = PARIS_OLYMPICS_IDENTIFIER.lower() in page_text
            
            # Check for 2024 dates (Paris Olympics were July 27 - August 4, 2024)
            has_2024_dates = any(date_indicator in page_text for date_indicator in [
                '2024',
                'jul 27',
                'july 27',
                'aug 4',
                'august 4'
            ])
            
            # Also check for specific Olympic Games links to Paris 2024 (/results/2424)
            olympic_links = soup.find_all('a', href=lambda x: x and '/results/2424' in x)
            has_paris_link = len(olympic_links) > 0
            
            if has_olympic_games and (has_2024_dates or has_paris_link):
                logger.info(f"✅ Verified Olympic swimmer - Olympic Games: {has_olympic_games}, 2024 dates: {has_2024_dates}, Paris link: {has_paris_link}")
                return True
            else:
                logger.debug(f"❌ Not verified - Olympic Games: {has_olympic_games}, 2024 dates: {has_2024_dates}, Paris link: {has_paris_link}")
                return False
            
        except Exception as e:
            logger.error(f"Error verifying Olympic swimmer at {profile_url}: {e}")
            return False
    
    async def extract_meets_from_profile(self, page, profile_url: str) -> List[str]:
        """Extract all meet URLs from swimmer's profile."""
        try:
            await page.goto(profile_url, timeout=TIMEOUT)
            await self.polite_delay()
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            meet_urls = []
            # Look for meet links in various possible structures
            meet_selectors = [
                'a[href*="/results/"]',
                '.meets-table a',
                '.meet-link',
                'a[href*="meet"]'
            ]
            
            for selector in meet_selectors:
                links = soup.select(selector)
                if links:
                    for link in links:
                        href = link.get('href')
                        if href and ('/results/' in href or '/meet/' in href):
                            if href.startswith('/'):
                                href = 'https://www.swimcloud.com' + href
                            if href not in meet_urls:
                                meet_urls.append(href)
            
            logger.info(f"Found {len(meet_urls)} meets for swimmer")
            return meet_urls
            
        except Exception as e:
            logger.error(f"Error extracting meets from {profile_url}: {e}")
            return []

    async def scrape_meet_times(self, page, meet_url: str, swimmer_name: str) -> List[Dict]:
        """Scrape all race times from a specific meet for the swimmer."""
        try:
            await page.goto(meet_url, timeout=TIMEOUT)
            await self.polite_delay()

            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')

            # Parse meet-level date and course info from the header section
            
            # Parse meet-level date and course info
            meet_date, meet_course = self.parse_meet_header(soup)
            
            race_records = []
            
            # Extract meet name
            meet_name = "Unknown Meet"
            meet_name_selectors = ['h1', '.meet-title', '.meet-name', 'title']
            for selector in meet_name_selectors:
                element = soup.select_one(selector)
                if element:
                    meet_name = element.get_text(strip=True)
                    break
            
            # Look for race results tables
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:  # Need at least event, time, and some other data
                        continue
                    
                    # Try to extract race data from row
                    race_data = self.parse_race_row(cells, swimmer_name, meet_name, meet_url)
                    if race_data:
                        # Fill missing race_date or course from meet header
                        if race_data.get('Race_Date') is None and meet_date is not None:
                            race_data['Race_Date'] = meet_date
                        if race_data.get('Course') is None and meet_course is not None:
                            race_data['Course'] = meet_course
                        race_records.append(race_data)
            
            logger.info(f"Extracted {len(race_records)} race records from {meet_name}")
            return race_records
            
        except Exception as e:
            logger.error(f"Error scraping meet {meet_url}: {e}")
            return []
    
    def parse_meet_header(self, soup: BeautifulSoup):
        """Extract meet-level date string and course (LCM/SCM/SCY) from the header."""
        header_text = soup.get_text(separator=' ', strip=True)
        # Default values
        meet_date = None
        meet_course = None
        # Find course token
        course_match = re.search(r'\b(LCM|SCM|SCY)\b', header_text, re.IGNORECASE)
        if course_match:
            meet_course = course_match.group(1).upper()
        # Find date range or single date with month name
        date_match = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[–\-]\d{1,2},?\s+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}', header_text)
        if date_match:
            meet_date = date_match.group(0)
        return meet_date, meet_course
    
    def parse_race_row(self, cells, swimmer_name: str, meet_name: str, meet_url: str) -> Optional[Dict]:
        """Parse a table row to extract race information."""
        try:
            # This is a simplified parser - you may need to adjust based on actual table structure
            race_data = {
                'Swimmer_ID': swimmer_name,
                'Race_Date': None,
                'Event_Name': None,
                'Course': None,
                'Time_Seconds': None,
                'Meet_Name': meet_name,
                'Meet_URL': meet_url
            }
            
            # Try to extract information from cells
            for i, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                
                # Look for event name (usually contains stroke/distance)
                if any(stroke in cell_text.lower() for stroke in ['free', 'back', 'breast', 'fly', 'medley']):
                    race_data['Event_Name'] = cell_text
                
                # Look for time (format like MM:SS.ss or SS.ss)
                if re.match(r'^\d{1,2}:\d{2}\.\d{2}$|^\d{1,3}\.\d{2}$', cell_text):
                    race_data['Time_Seconds'] = self.parse_time_to_seconds(cell_text)
                
                # Look for course (LCM, SCM, SCY)
                if cell_text.upper() in ['LCM', 'SCM', 'SCY', 'LCY', 'SCY']:
                    race_data['Course'] = cell_text.upper()
            
                # Look for date patterns like Apr 21, 2025 or 21/04/2025
                if re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', cell_text):
                    race_data['Race_Date'] = cell_text
            
            # Only return if we have essential data
            if race_data['Event_Name'] and race_data['Time_Seconds']:
                return race_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing race row: {e}")
            return None
    
    async def process_swimmer(self, page, swimmer_name: str) -> List[Dict]:
        """Process a single swimmer: search, verify, and scrape all times."""
        logger.info(f"Processing swimmer: {swimmer_name}")
        
        try:
            # Step 1: Search for swimmer
            profile_urls = await self.search_swimmer(page, swimmer_name)
            
            if not profile_urls:
                logger.warning(f"No profiles found for {swimmer_name}")
                self.failed_swimmers.append(swimmer_name)
                return []
            
            # Step 2: Verify Olympic participation
            verified_profile = None
            for profile_url in profile_urls:
                logger.info(f"Checking profile: {profile_url}")
                if await self.verify_olympic_swimmer(page, profile_url):
                    verified_profile = profile_url
                    logger.info(f"Verified Olympic swimmer: {swimmer_name}")
                    break
                await self.polite_delay(1, 2)
            
            if not verified_profile:
                logger.warning(f"No verified Olympic profile found for {swimmer_name}")
                self.failed_swimmers.append(swimmer_name)
                return []
            
            # Step 3: Extract all meets
            meet_urls = await self.extract_meets_from_profile(page, verified_profile)
            
            if not meet_urls:
                logger.warning(f"No meets found for {swimmer_name}")
                return []
            
            # Step 4: Scrape times from all meets
            all_race_records = []
            for meet_url in meet_urls[:10]:  # Limit to first 10 meets to avoid overwhelming
                logger.info(f"Scraping meet: {meet_url}")
                race_records = await self.scrape_meet_times(page, meet_url, swimmer_name)
                all_race_records.extend(race_records)
                await self.polite_delay()
            
            logger.info(f"Collected {len(all_race_records)} total race records for {swimmer_name}")
            return all_race_records
            
        except Exception as e:
            logger.error(f"Error processing swimmer {swimmer_name}: {e}")
            self.failed_swimmers.append(swimmer_name)
            return []
    
    async def run(self) -> List[Dict]:
        """Main scraping function that returns a list of race record dictionaries."""
        logger.info("Starting Youth Times SwimCloud Scraper...")
        
        # Load swimmer names
        swimmer_names = self.load_swimmer_names()

        # Resume capability: if output JSON exists, load already scraped records and skip their swimmers
        scraped_swimmer_ids: set[str] = set()
        if os.path.exists(JSON_OUTPUT_FILE):
            try:
                with open(JSON_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
                if isinstance(existing_records, list):
                    self.all_scraped_race_records.extend(existing_records)
                    scraped_swimmer_ids = {rec.get('Swimmer_ID') for rec in existing_records if rec.get('Swimmer_ID')}
                    logger.info(f"Resuming run — {len(scraped_swimmer_ids)} swimmers already scraped (loaded from {JSON_OUTPUT_FILE})")
                else:
                    logger.warning(f"Unexpected JSON structure in {JSON_OUTPUT_FILE}; starting fresh")
            except Exception as e:
                logger.error(f"Could not load existing JSON file for resume: {e}")

        # Filter swimmer list to those not yet scraped
        swimmer_names = [name for name in swimmer_names if name not in scraped_swimmer_ids]
        if not swimmer_names:
            logger.error("No swimmer names loaded. Exiting.")
            return []
        
        self.total_swimmers = len(swimmer_names)
        logger.info(f"Processing {self.total_swimmers} swimmers")
        
        async with async_playwright() as p:
            # Launch browser with anti-detection measures
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )
            
            # Add script to remove webdriver property
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            page = await context.new_page()
            
            try:
                # Process each swimmer
                for i, swimmer_name in enumerate(swimmer_names, 1):
                    logger.info(f"\n[{i}/{self.total_swimmers}] Processing: {swimmer_name}")
                    
                    race_records = await self.process_swimmer(page, swimmer_name)
                    self.all_scraped_race_records.extend(race_records)
                    self.processed_swimmers += 1
                    
                    logger.info(f"Progress: {self.processed_swimmers}/{self.total_swimmers} swimmers processed")
                    logger.info(f"Total race records collected: {len(self.all_scraped_race_records)}")
                    
                    # Polite delay between swimmers
                    await self.polite_delay()
                
                # Log final results
                logger.info(f"\n✅ Scraping completed successfully!")
                logger.info(f"📊 Statistics:")
                logger.info(f"   - Swimmers processed: {self.processed_swimmers}/{self.total_swimmers}")
                logger.info(f"   - Total race records: {len(self.all_scraped_race_records)}")
                logger.info(f"   - Failed swimmers: {len(self.failed_swimmers)}")
                if self.failed_swimmers:
                    logger.info(f"   - Failed swimmer names: {', '.join(self.failed_swimmers)}")
                
                # Save to JSON file
                with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.all_scraped_race_records, f, indent=2, ensure_ascii=False)
                logger.info(f"Race data saved to {JSON_OUTPUT_FILE}")
                
                return self.all_scraped_race_records
                
            except Exception as e:
                logger.error(f"Fatal error: {e}")
            finally:
                await browser.close()
    


# Main execution
async def main() -> List[Dict]:
    """Entry point for the scraper that returns list of race dictionaries."""
    scraper = YouthTimesSwimCloudScraper()
    race_records = await scraper.run()
    return race_records

if __name__ == "__main__":
    # Allow overriding the input file via `python youthTimesScraper.py myfile.txt`
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
        logger.info(f"Overriding INPUT_FILE -> {INPUT_FILE}")
    # Run scraper and get list of dictionaries
    race_data = asyncio.run(main())
    print(f"Collected {len(race_data)} race records as list of dictionaries")
    
    # Print all race data as formatted JSON
    if race_data:
        print("\n" + "="*80)
        print("ALL RACE DATA (JSON FORMAT):")
        print("="*80)
        print(json.dumps(race_data, indent=2, ensure_ascii=False))
        print("="*80)
        print(f"\n💾 JSON data also saved to: {JSON_OUTPUT_FILE}")
    else:
        print("\nNo race data collected.")