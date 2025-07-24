import asyncio
import random
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from typing import Set, List, Dict

# Configuration
MEET_URL = "https://www.swimcloud.com/results/2424/"
OUTPUT_FILE = "paris2024_swimmers.txt"
MIN_DELAY = 2.0
MAX_DELAY = 5.0
TIMEOUT = 15000

class SwimCloudScraper:
    """Scraper for Paris 2024 Olympics swimmer names from SwimCloud."""
    
    def __init__(self):
        self.swimmer_names: Set[str] = set()
        self.processed_events = 0
        self.total_events = 0
    
    async def polite_delay(self, min_sec: float = MIN_DELAY, max_sec: float = MAX_DELAY) -> None:
        """Introduce random delay to avoid overwhelming the server."""
        delay = min_sec + (max_sec - min_sec) * random.random()
        await asyncio.sleep(delay)
    
    def format_swimmer_name(self, raw_name: str) -> str:
        """Format swimmer name to 'First Name Last Name' format."""
        # Remove extra whitespace and normalize
        name = re.sub(r'\s+', ' ', raw_name.strip())
        
        # Skip relay team names (usually contain multiple names or team indicators)
        if any(indicator in name.lower() for indicator in ['relay', 'team', '&', ' and ', ',']):
            return None
        
        # Skip if name is too short or contains numbers
        if len(name) < 3 or any(char.isdigit() for char in name):
            return None
        
        # Basic name validation - should contain at least first and last name
        name_parts = name.split()
        if len(name_parts) < 2:
            return None
        
        # Remove common country suffixes that might be attached
        # (like "Jack AlexyUnited States" -> "Jack Alexy")
        countries = ['United States', 'France', 'Germany', 'Australia', 'Canada', 'Great Britain', 
                    'Italy', 'Japan', 'China', 'Brazil', 'Netherlands', 'Sweden', 'Hungary',
                    'Romania', 'Poland', 'Spain', 'South Africa', 'New Zealand']
        
        for country in countries:
            if name.endswith(country):
                name = name[:-len(country)].strip()
                break
        
        # Re-validate after country removal
        name_parts = name.split()
        if len(name_parts) < 2:
            return None
        
        return name
    
    def extract_event_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract all individual event links from the dashboard."""
        event_links = []
        base_url = "https://www.swimcloud.com"
        
        # Look for event links with various possible selectors
        selectors = [
            'a[href*="/results/2424/event/"]',
            'a[href*="/event/"]',
            '.event-link a',
            '.results-table a[href*="event"]'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            if links:
                print(f"Found {len(links)} links with selector: {selector}")
                break
        
        for link in links:
            href = link.get('href')
            if href:
                # Ensure absolute URL
                if href.startswith('/'):
                    href = base_url + href
                elif not href.startswith('http'):
                    continue
                
                # Filter for individual events (not relays if possible)
                link_text = link.get_text(strip=True).lower()
                if 'relay' not in link_text and href not in event_links:
                    event_links.append(href)
        
        return event_links
    
    def identify_session_sections(self, soup: BeautifulSoup) -> Dict[str, any]:
        """Identify Finals, Semifinals, and Preliminaries sections."""
        session_headers = {}
        
        # Look for session headers - be more specific to avoid navigation elements
        header_tags = ['h1', 'h2', 'h3', 'h4']
        
        for tag in header_tags:
            headers = soup.find_all(tag)
            for header in headers:
                text = header.get_text(strip=True).lower()
                
                # More specific matching to avoid navigation elements
                # Look for standalone session words, not mixed with other content
                if len(text) < 50:  # Avoid long navigation text
                    if text == 'finals' or (text.endswith('finals') and len(text.split()) <= 3):
                        session_headers['finals'] = header
                    elif text == 'semifinals' or (text.endswith('semifinals') and len(text.split()) <= 3):
                        session_headers['semifinals'] = header
                    elif text == 'preliminaries' or text == 'prelims' or (text.endswith('preliminaries') and len(text.split()) <= 3):
                        session_headers['preliminaries'] = header
        
        # If no specific headers found, try a different approach - look for tables directly
        if not session_headers:
            # Find all tables and try to infer sessions from context
            tables = soup.find_all('table')
            if len(tables) >= 3:
                # Typically: Finals (smallest), Semifinals (medium), Preliminaries (largest)
                tables_with_sizes = [(table, len(table.find_all('tr'))) for table in tables]
                tables_with_sizes.sort(key=lambda x: x[1])  # Sort by number of rows
                
                if len(tables_with_sizes) >= 3:
                    session_headers['finals'] = tables_with_sizes[0][0]  # Smallest table
                    session_headers['semifinals'] = tables_with_sizes[1][0]  # Medium table
                    session_headers['preliminaries'] = tables_with_sizes[-1][0]  # Largest table
                elif len(tables_with_sizes) == 2:
                    session_headers['finals'] = tables_with_sizes[0][0]  # Smaller table
                    session_headers['preliminaries'] = tables_with_sizes[1][0]  # Larger table
        
        return session_headers
    
    def determine_sessions_to_scrape(self, session_headers: Dict[str, any]) -> List[str]:
        """Determine which sessions to scrape based on availability."""
        available_sessions = list(session_headers.keys())
        
        # Apply the rules specified in requirements
        if all(session in available_sessions for session in ['finals', 'semifinals', 'preliminaries']):
            # Rule 1: If all three exist, scrape only Semifinals and Finals
            return ['semifinals', 'finals']
        elif 'finals' in available_sessions and 'preliminaries' in available_sessions:
            # Rule 2: If only Finals and Preliminaries exist, scrape both
            return ['preliminaries', 'finals']
        else:
            # Fallback: scrape whatever is available
            return available_sessions
    
    def extract_swimmers_from_session(self, soup: BeautifulSoup, session_element: any) -> List[str]:
        """Extract swimmer names from a specific session section."""
        swimmers = []
        table = None
        
        # If session_element is already a table, use it directly
        if session_element.name == 'table':
            table = session_element
        else:
            # Find the results table following the session header
            current_element = session_element
            
            # Look for table in next siblings (up to 10 elements ahead)
            for _ in range(10):
                current_element = current_element.find_next_sibling()
                if not current_element:
                    break
                
                if current_element.name == 'table':
                    table = current_element
                    break
                
                # Sometimes table is nested in a div
                nested_table = current_element.find('table')
                if nested_table:
                    table = nested_table
                    break
            
            if not table:
                # Alternative: look for table in the next few elements
                next_elements = session_element.find_all_next(['table', 'div'], limit=20)
                for elem in next_elements:
                    if elem.name == 'table':
                        table = elem
                        break
                    elif elem.find('table'):
                        table = elem.find('table')
                        break
        
        if table:
            # Extract names from table rows
            rows = table.select('tbody tr') or table.select('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                
                # Look for swimmer name - typically in second cell with a swimmer link
                for i, cell in enumerate(cells):
                    # Skip rank/position cells (usually first cell)
                    if i == 0:
                        continue
                    
                    # Look for swimmer links specifically
                    swimmer_link = cell.find('a', href=lambda x: x and '/swimmer/' in x)
                    if swimmer_link:
                        name = swimmer_link.get_text(strip=True)
                        # Clean up name (remove country suffix if present)
                        name_parts = name.split()
                        if len(name_parts) >= 2:
                            # Take first two parts as first and last name
                            clean_name = f"{name_parts[0]} {name_parts[1]}"
                            formatted_name = self.format_swimmer_name(clean_name)
                            if formatted_name:
                                swimmers.append(formatted_name)
                        break  # Found name in this row, move to next row
        
        return swimmers
    
    async def scrape_event_page(self, page, event_url: str) -> int:
        """Scrape swimmer names from a single event page."""
        try:
            print(f"  Navigating to: {event_url}")
            await page.goto(event_url, timeout=TIMEOUT)
            await page.wait_for_selector('body', timeout=TIMEOUT)
            
            # Check if we got blocked on this page
            content = await page.content()
            if '403 Forbidden' in content or 'Access Denied' in content:
                print(f"    ❌ Access blocked for {event_url}")
                return 0
            
            await self.polite_delay()
            
            # Get page content and parse with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Identify session sections
            session_headers = self.identify_session_sections(soup)
            
            if not session_headers:
                print(f"    No session sections found")
                return 0
            
            print(f"    Found sessions: {list(session_headers.keys())}")
            
            # Determine which sessions to scrape
            sessions_to_scrape = self.determine_sessions_to_scrape(session_headers)
            print(f"    Scraping sessions: {sessions_to_scrape}")
            
            # Extract swimmers from each target session
            event_swimmers = 0
            for session in sessions_to_scrape:
                if session in session_headers:
                    swimmers = self.extract_swimmers_from_session(soup, session_headers[session])
                    print(f"      {session.title()}: {len(swimmers)} swimmers")
                    
                    for swimmer in swimmers:
                        self.swimmer_names.add(swimmer)
                        event_swimmers += 1
            
            return event_swimmers
            
        except Exception as e:
            print(f"    Error processing {event_url}: {str(e)}")
            return 0
    
    async def run(self) -> None:
        """Main scraping function."""
        print("Starting Paris 2024 Olympics SwimCloud Scraper...")
        print(f"Target URL: {MEET_URL}")
        
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
                # Step 1: Navigate to meet dashboard
                print("\n1. Navigating to meet dashboard...")
                
                # First, visit a simple page to establish session
                await page.goto('https://www.swimcloud.com/', timeout=TIMEOUT)
                await self.polite_delay(2, 4)
                
                # Now navigate to the meet dashboard
                await page.goto(MEET_URL, timeout=TIMEOUT)
                await page.wait_for_selector('body', timeout=TIMEOUT)
                
                # Check if we got blocked
                content = await page.content()
                if '403 Forbidden' in content or 'Access Denied' in content:
                    print("❌ Access blocked by anti-bot protection")
                    print("Try running the script with longer delays or different user agent")
                    return
                
                await self.polite_delay()
                
                # Step 2: Extract event links
                print("\n2. Extracting event links...")
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                event_links = self.extract_event_links(soup)
                
                self.total_events = len(event_links)
                print(f"Found {self.total_events} individual events to process")
                
                if not event_links:
                    print("No event links found. Please check the page structure.")
                    return
                
                # Step 3: Process each event
                print("\n3. Processing individual events...")
                for i, event_url in enumerate(event_links, 1):
                    print(f"\n[{i}/{self.total_events}] Processing event {i}")
                    
                    swimmers_added = await self.scrape_event_page(page, event_url)
                    self.processed_events += 1
                    
                    print(f"    Added {swimmers_added} swimmers (Total unique: {len(self.swimmer_names)})")
                    
                    # Polite delay between events
                    await self.polite_delay()
                
                # Step 4: Save results
                print("\n4. Saving results...")
                swimmer_list = sorted(list(self.swimmer_names))
                
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    for name in swimmer_list:
                        f.write(f"{name}\n")
                
                print(f"\n✅ Scraping completed successfully!")
                print(f"📊 Statistics:")
                print(f"   - Events processed: {self.processed_events}/{self.total_events}")
                print(f"   - Unique swimmers found: {len(swimmer_list)}")
                print(f"   - Output file: {OUTPUT_FILE}")
                
            except Exception as e:
                print(f"\n❌ Fatal error: {str(e)}")
            finally:
                await browser.close()

# Main execution
async def main():
    """Entry point for the scraper."""
    scraper = SwimCloudScraper()
    await scraper.run()

if __name__ == "__main__":
    asyncio.run(main())