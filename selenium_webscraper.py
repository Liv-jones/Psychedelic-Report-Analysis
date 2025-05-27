import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from typing import List, Dict, Any, Optional
import re
import traceback

class ErowidSeleniumScraper:
    """Extracts experience reports from Erowid using Selenium."""
    BASE_URL = "https://www.erowid.org/experiences/exp_list.shtml"
    ALL_EXPERIENCES_URL = "https://www.erowid.org/experiences/exp.cgi?ShowViews=0&Cellar=0&Start=0&Max=100000"
    
    SUBSTANCE_MAPPING = {
        "lsd": "lsd", "acid": "lsd",
        "mushrooms": "mushrooms", "shrooms": "mushrooms", "psilocybin": "mushrooms",
        "mdma": "mdma", "ecstasy": "mdma", "molly": "mdma",
        "cannabis": "cannabis", "marijuana": "cannabis", "weed": "cannabis",
        "dmt": "dmt", "ayahuasca": "ayahuasca",
        "ketamine": "ketamine", "salvia": "salvia",
        "mescaline": "mescaline", "peyote": "mescaline", "san pedro": "mescaline"
    }
    
    DIRECT_SUBSTANCE_URLS = {
        "lsd": "https://www.erowid.org/experiences/subs/exp_LSD.shtml",
        "mushrooms": "https://www.erowid.org/experiences/subs/exp_Mushrooms.shtml",
        "mdma": "https://www.erowid.org/experiences/subs/exp_MDMA.shtml",
        "dmt": "https://www.erowid.org/experiences/subs/exp_DMT.shtml",
        "cannabis": "https://www.erowid.org/experiences/subs/exp_Cannabis.shtml",
        "ayahuasca": "https://www.erowid.org/experiences/subs/exp_Ayahuasca.shtml",
        "ketamine": "https://www.erowid.org/experiences/subs/exp_Ketamine.shtml",
        "salvia": "https://www.erowid.org/experiences/subs/exp_Salvia_divinorum.shtml",
        "mescaline": "https://www.erowid.org/experiences/subs/exp_Mescaline.shtml"
    }
    
    def __init__(self, timeout: int = 15, use_safari: bool = True):
        """Initializes Selenium WebDriver."""
        self.default_timeout = timeout
        if use_safari:
            self.driver = webdriver.Safari()
        else:
            try:
                from selenium.webdriver.chrome.options import Options
                chrome_options = Options()
                chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                self.driver = webdriver.Chrome(options=chrome_options)
            except:
                self.driver = webdriver.Safari() # Fallback
        self.driver.set_page_load_timeout(self.default_timeout)
    
    def get_report_links(self, max_links: Optional[int] = 100, links_file: str = "data/raw/erowid_links.txt", substances: Optional[List[str]] = None, show_progress: bool = True, force_scrape: bool = False) -> List[str]:
        """Fetches experience report links."""
        links_with_substance_from_file = []

        if os.path.exists(links_file) and not force_scrape:
            try:
                with open(links_file, "r") as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        parts = line.split('|', 1)
                        link = parts[0]
                        substance = parts[1] if len(parts) > 1 and parts[1] != "Erowid" else "Unknown"
                        links_with_substance_from_file.append((link, substance))

                if links_with_substance_from_file:
                    # print(f"Loaded {len(links_with_substance_from_file)} links from {links_file}")
                    loaded_links = [link for link, _ in links_with_substance_from_file]
                    return loaded_links if max_links is None else loaded_links[:max_links]
            except Exception as e:
                print(f"Error loading links file ({links_file}): {e}. Proceeding to scrape.")
        
        dirname = os.path.dirname(links_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        scraped_links_with_substance = []

        if substances and len(substances) > 0:
            for substance_query in tqdm(substances, desc="Processing direct substance pages", disable=not show_progress):
                substance_lower = substance_query.lower()
                direct_url = self.DIRECT_SUBSTANCE_URLS.get(substance_lower)
                if not direct_url: # Check mapping if not a direct key
                    for map_key, map_value in self.SUBSTANCE_MAPPING.items():
                        if substance_lower == map_key: # Prioritize direct mapping lookup
                             if map_value in self.DIRECT_SUBSTANCE_URLS:
                                direct_url = self.DIRECT_SUBSTANCE_URLS[map_value]
                                break
                
                if direct_url:
                    try:
                        self.driver.get(direct_url)
                        page_title = self.driver.title
                        actual_substance_name = substance_query
                        if "Erowid" in page_title and ":" in page_title:
                            actual_substance_name = page_title.split(":")[1].strip().split(" Exp")[0] # Cleaner name
                        
                        exp_elements = self.driver.find_elements(By.XPATH, "//a[contains(@href, 'exp.php?ID=')]")
                        for link_el in exp_elements:
                            href = link_el.get_attribute('href')
                            if href and href.startswith("https://www.erowid.org"):
                                scraped_links_with_substance.append((href, actual_substance_name))
                        
                        if max_links is not None and len(exp_elements) >= max_links / len(substances): # Rough distribution
                            continue
                    except Exception as e:
                        print(f"Error accessing {direct_url} for {substance_query}: {e}")
        
        needs_main_scrape = (not substances or len(substances) == 0 or len(scraped_links_with_substance) < (max_links or float('inf')))

        if needs_main_scrape:
            target_url = self.ALL_EXPERIENCES_URL if not substances or len(substances) == 0 else self.BASE_URL
            long_timeout = 180
            try:
                self.driver.set_page_load_timeout(long_timeout)
                self.driver.get(target_url)
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                exp_links_soup = soup.find_all('a', href=re.compile(r'exp\.php\?ID=\d+'))

                if exp_links_soup:
                    for link_element in tqdm(exp_links_soup, desc="Processing links from index page", disable=not show_progress):
                        href = link_element.get('href')
                        if href:
                            if not href.startswith('http'):
                                href = f"https://www.erowid.org/experiences/{href.lstrip('/')}"
                            if href.startswith("https://www.erowid.org/experiences/exp.php?ID=") and not any(href == ex_link for ex_link, _ in scraped_links_with_substance):
                                link_text = link_element.get_text(strip=True)
                                substance_hint = "Unknown"
                                match = re.search(r'\(([^)]+)\)', link_text)
                                if match:
                                    substance_hint = match.group(1).strip()
                                scraped_links_with_substance.append((href, substance_hint))
                                if max_links is not None and len(scraped_links_with_substance) >= max_links * 2: # Heuristic
                                    break 
                elif not substances or len(substances) == 0 : # Only warn if we expected a lot of links
                     print("Warning: BeautifulSoup found no experience links on the main index page.")
            except Exception as e:
                print(f"Error scraping main index page {target_url}: {e}")
                traceback.print_exc()
            finally:
                self.driver.set_page_load_timeout(self.default_timeout)

        final_link_to_substance = {}
        # Add links from file first (if not force_scrape)
        if not force_scrape:
            for link, substance in links_with_substance_from_file:
                 if link not in final_link_to_substance or final_link_to_substance[link] == "Unknown":
                    final_link_to_substance[link] = substance
        
        # Add/update with newly scraped links
        for link, substance in scraped_links_with_substance:
            clean_substance = "Unknown" if substance and ("Erowid" in substance or substance.strip() == "") else substance
            if link in final_link_to_substance:
                if final_link_to_substance[link] == "Unknown" and clean_substance != "Unknown":
                    final_link_to_substance[link] = clean_substance
            else:
                final_link_to_substance[link] = clean_substance

        if final_link_to_substance: # Only write if there's something to write
            try:
                with open(links_file, "w") as f:
                    for link, substance in final_link_to_substance.items():
                        f.write(f"{link}|{substance if substance else 'Unknown'}\n")
                # print(f"Saved {len(final_link_to_substance)} total unique links to {links_file}")
            except Exception as e:
                print(f"Warning: Could not save links to file {links_file}: {e}")

        unique_links_list = list(final_link_to_substance.keys())
        return unique_links_list if max_links is None else unique_links_list[:max_links]
    
    def extract_report_data(self, url: str) -> Dict[str, Any]:
        """Extracts data from an individual experience report page."""
        try:
            self.driver.get(url)
        except TimeoutException:
            print(f"Timeout: Skipping {url}")
            return {"error": "Timeout"}
        except Exception as e:
            print(f"Error loading page {url}: {e}")
            return {"error": f"Page load error: {str(e)}"}

        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        if "403 Forbidden: Your IP Address Has Been Blocked" in page_source or \
           "Click this Link to email us and provide a bit of info" in page_source:
            print(f"🚨 IP BANNED while trying to access {url}.")
            return {"error": "IP banned"}
        
        if "reset.me" in self.driver.current_url or "wordpress.com" in self.driver.current_url:
            print(f"🚨 Redirected to external site: {self.driver.current_url}. Skipping report.")
            return {"error": "Redirected to external site"}
        
        try:
            title = soup.find("div", class_="title").text.strip() if soup.find("div", class_="title") else "Unknown Title"
            raw_substance = soup.find("div", class_="substance").text.strip() if soup.find("div", class_="substance") else "Unknown Substance"
            substances_list = self._parse_multiple_substances(raw_substance)
            author = (soup.find("div", class_="author").text.replace("by", "").strip() if soup.find("div", class_="author") else "Unknown")
            bodyweight = soup.find("td", class_="bodyweight-amount").text.strip() if soup.find("td", class_="bodyweight-amount") else "Unknown"
            
            dose_chart_entries = [" | ".join([col.text.strip() for col in row.find_all("td")]) 
                                  for row in soup.select("table.dosechart tbody tr") if len(row.find_all("td")) == 5]
            dose_chart = "\n".join(dose_chart_entries) if dose_chart_entries else "No Dose Chart Available"
            
            report_text = ""
            report_text_element = soup.find("div", class_="report-text-surround")
            if report_text_element:
                paragraphs = report_text_element.find_all('p')
                if paragraphs:
                    report_text = "\n\n".join([p.text.strip() for p in paragraphs])
                else: # Fallback text extraction
                    all_text = report_text_element.get_text()
                    body_weight_marker = "BODY WEIGHT:"
                    bw_pos = all_text.find(body_weight_marker)
                    if bw_pos != -1:
                        line_end = all_text.find('\n', bw_pos)
                        if line_end != -1:
                            start_pos = line_end + 1
                            # Skip immediate blank lines after body weight line
                            while start_pos < len(all_text) and all_text[start_pos].isspace():
                                next_newline = all_text.find('\n', start_pos)
                                if next_newline == -1: break # Should not happen if isspace() is true
                                start_pos = next_newline + 1
                            report_text = all_text[start_pos:].strip()
                        else: # Body weight marker is the last thing? Unlikely.
                            report_text = all_text.strip() # Default to all if parsing fails
                    else: # if no body weight, try to remove typical dose lines
                        lines = all_text.split('\n')
                        last_dose_line_idx = -1
                        dose_patterns = ["DOSE:", "T+", "Dosage:", "Doses:"]
                        for i, line_content in enumerate(lines):
                            if any(pattern in line_content for pattern in dose_patterns):
                                last_dose_line_idx = i
                        
                        if last_dose_line_idx != -1:
                            start_text_line_idx = last_dose_line_idx + 1
                            # Skip blank lines after last dose line
                            while start_text_line_idx < len(lines) and not lines[start_text_line_idx].strip():
                                start_text_line_idx += 1
                            
                            # Reconstruct report, stopping at footdata markers
                            clean_text_lines = []
                            footdata_markers = ["Exp Year:", "Gender:", "Age at time of experience:", "Published:"]
                            for line_content in lines[start_text_line_idx:]:
                                if any(marker in line_content for marker in footdata_markers):
                                    break
                                clean_text_lines.append(line_content)
                            report_text = "\n".join(clean_text_lines).strip()
                        else:
                           report_text = all_text.strip()
            
            footdata = {}
            footdata_section = soup.find("div", class_="footdata")
            if footdata_section:
                for field_name, label in {"exp_year": "Exp Year:", "gender": "Gender:", "age": "Age at time of experience:", "published_date": "Published:"}.items():
                    element = footdata_section.find(string=lambda text: text and label in text)
                    if element:
                        footdata[field_name] = element.replace(label, "").strip()
            else: # Fallback for footdata if not in div.footdata
                 for text_node in soup.find_all(string=True):
                    if "Exp Year:" in text_node: footdata['exp_year'] = text_node.replace("Exp Year:", "").strip()
                    elif "Gender:" in text_node: footdata['gender'] = text_node.replace("Gender:", "").strip()
                    elif "Age at time of experience:" in text_node: footdata['age'] = text_node.replace("Age at time of experience:", "").strip()
                    elif "Published:" in text_node: footdata['published_date'] = text_node.replace("Published:", "").strip()


            return {
                "title": title, "substance": raw_substance, "substances_list": substances_list,
                "author": author, "bodyweight": bodyweight, "dose_chart": dose_chart,
                "report_text": report_text, "url": url,
                "exp_year": footdata.get('exp_year', 'Unknown'), "gender": footdata.get('gender', 'Unknown'),
                "age": footdata.get('age', 'Unknown'), "published_date": footdata.get('published_date', 'Unknown')
            }
        except Exception as e:
            print(f"Error extracting data from {url}: {e}")
            traceback.print_exc()
            return {}
    
    def _parse_multiple_substances(self, substance_text: str) -> List[str]:
        """Parses a substance string which may contain multiple substances."""
        if not substance_text or substance_text == "Unknown Substance":
            return ["Unknown"]
        
        normalized_text = substance_text
        for connector in [" & ", " and ", " + ", " with ", "+"]: # Added " with " and " + " (no spaces)
            normalized_text = normalized_text.replace(connector, ", ")
        
        substances = []
        for item in normalized_text.split(","):
            item = item.strip()
            if item:
                clean_item = re.sub(r'\(\d+[a-zA-Zµgmc]*\s?\w*\)', '', item).strip()
                clean_item = re.sub(r'\s*\([^)]*\)', '', clean_item).strip()
                if clean_item: 
                    substances.append(clean_item)
        
        return substances if substances else ["Unknown"] # Return ["Unknown"] if list ends up empty
    
    def scrape_reports(self, substances: Optional[List[str]] = None, max_reports: Optional[int] = 100, output_file: str = "data/raw/erowid_reports.csv", show_progress: bool = True) -> pd.DataFrame:
        """Scrapes Erowid reports, saves to CSV, and returns a DataFrame."""
        links_file = "data/raw/erowid_links.txt"
        checkpoint_file = os.path.join(os.path.dirname(output_file), "scrape_checkpoint.txt")
        timeout_log_file = os.path.join(os.path.dirname(output_file), "timeout_links.txt")

        scraped_urls_from_checkpoint_or_csv = set()
        last_scraped_index = -1 # For checkpointing within a run
        timed_out_urls = set()

        if os.path.exists(timeout_log_file):
            try:
                with open(timeout_log_file, "r") as f: timed_out_urls = set(f.read().splitlines())
            except Exception as e: print(f"Error reading timeout log {timeout_log_file}: {e}")

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r") as f:
                    lines = f.read().strip().split('\n')
                    if lines and lines[0].isdigit(): last_scraped_index = int(lines[0])
                    if len(lines) > 1: scraped_urls_from_checkpoint_or_csv.update(lines[1:])
            except Exception as e: print(f"Error reading checkpoint file: {e}")
        
        existing_df = None
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                if 'url' in existing_df.columns:
                    scraped_urls_from_checkpoint_or_csv.update(existing_df['url'].dropna().astype(str))
            except Exception as e: print(f"Error reading existing output {output_file}: {e}")
        
        all_report_links_with_substance = [] # This will hold links for current scraping session
        if os.path.exists(links_file):
            try:
                with open(links_file, "r") as f:
                    for line in f.read().splitlines():
                        parts = line.split('|', 1)
                        all_report_links_with_substance.append((parts[0], parts[1] if len(parts) > 1 else "Unknown"))
            except Exception as e: print(f"Error reading links file {links_file}: {e}")
        
        if not all_report_links_with_substance:
            links_to_fetch = None if max_reports is None else max_reports * 2 
            fetched_link_urls = self.get_report_links(max_links=links_to_fetch, substances=substances, show_progress=show_progress, links_file=links_file)
            if not fetched_link_urls: return existing_df if existing_df is not None else pd.DataFrame()
            all_report_links_with_substance = [(link, "Unknown") for link in fetched_link_urls] 

        if substances and len(substances) > 0:
            normalized_substances = [s.lower() for s in substances]
            filtered_links = []
            for link, substance_text in all_report_links_with_substance:
                sub_text_lower = substance_text.lower()
                # Check direct presence or mapping
                if any(req_sub in sub_text_lower for req_sub in normalized_substances) or \
                   any(self.SUBSTANCE_MAPPING.get(req_sub) == self.SUBSTANCE_MAPPING.get(known_map_key) 
                       for req_sub in normalized_substances 
                       for known_map_key in self.SUBSTANCE_MAPPING if known_map_key in sub_text_lower and self.SUBSTANCE_MAPPING.get(req_sub)):
                    filtered_links.append((link, substance_text))
            all_report_links_with_substance = filtered_links
        
        unprocessed_links = []
        for link, substance in all_report_links_with_substance:
            if link not in scraped_urls_from_checkpoint_or_csv and link not in timed_out_urls:
                unprocessed_links.append((link, substance))
        
        if not unprocessed_links and os.path.exists(links_file): # If all known links processed, try to get more
            links_to_force_fetch = None if max_reports is None else max_reports * 2
            forced_fetched_link_urls = self.get_report_links(max_links=links_to_force_fetch, links_file=links_file, substances=substances, show_progress=show_progress, force_scrape=True)
            new_links_after_force = [link for link in forced_fetched_link_urls if link not in scraped_urls_from_checkpoint_or_csv and link not in timed_out_urls]
            if new_links_after_force:
                unprocessed_links.extend([(link, "Unknown") for link in new_links_after_force])
        
        links_to_scrape_this_session = unprocessed_links
        if max_reports is not None:
            links_to_scrape_this_session = unprocessed_links[:max_reports]
            
        if not links_to_scrape_this_session:
            return existing_df if existing_df is not None else pd.DataFrame()
        
        scraped_reports_data = existing_df.to_dict('records') if existing_df is not None and not existing_df.empty else []
        
        # Determine starting index for tqdm based on checkpoint (within this batch of links_to_scrape_this_session)
        current_run_start_idx = max(0, last_scraped_index + 1 if last_scraped_index != -1 else 0)

        progress_iter = tqdm(
            enumerate(links_to_scrape_this_session), 
            desc="Scraping reports", 
            initial=0,
            total=len(links_to_scrape_this_session), 
            disable=not show_progress
        )
        
        newly_scraped_count_this_run = 0

        try:
            for current_idx, (link_url, original_substance_hint) in progress_iter:
                if current_idx < current_run_start_idx: # Skip already processed items based on checkpoint for this batch
                    continue

                report_data = self.extract_report_data(link_url)

                if report_data.get("error") == "Timeout":
                    timed_out_urls.add(link_url)
                    self._update_timeout_file(timeout_log_file, timed_out_urls)
                    continue
                if report_data.get("error") == "IP banned":
                    print("⛔ STOPPING: IP banned by Erowid. Saving current progress.")
                    break 
                
                if report_data and "error" not in report_data:
                    # Update substance in links file if it was "Unknown" or missing
                    extracted_substance = report_data.get("substance", "Unknown")
                    if (original_substance_hint == "Unknown" or not original_substance_hint) and extracted_substance != "Unknown":
                         self._update_links_file(links_file, link_url, extracted_substance)
                    
                    if "substance" not in report_data or not report_data["substance"]:
                        report_data["substance"] = original_substance_hint

                    scraped_reports_data.append(report_data)
                    scraped_urls_from_checkpoint_or_csv.add(link_url) # Add to master set of scraped URLs
                    newly_scraped_count_this_run +=1
                    
                    self._update_checkpoint(checkpoint_file, current_idx, scraped_urls_from_checkpoint_or_csv) # Checkpoint based on current_idx of this batch
                    
                    if newly_scraped_count_this_run > 0 and newly_scraped_count_this_run % 50 == 0: # Save every 50 NEW reports
                        self.save_to_csv(pd.DataFrame(scraped_reports_data), output_file)
                # time.sleep(1)
        except KeyboardInterrupt:
            print("Scraping interrupted by user. Saving progress...")
        except Exception as e:
            print(f"Scraping loop error: {e}. Saving progress...")
            traceback.print_exc()
        finally:
            if scraped_reports_data:
                final_df = pd.DataFrame(scraped_reports_data)
                self.save_to_csv(final_df, output_file)
                return final_df
            return existing_df if existing_df is not None else pd.DataFrame()
            
    def _update_checkpoint(self, checkpoint_file: str, last_processed_idx_in_batch: int, all_scraped_urls: set):
        """Updates checkpoint file: last index processed in current batch and all scraped URLs."""
        try:
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            with open(checkpoint_file, "w") as f:
                f.write(f"{last_processed_idx_in_batch}\n")
                for url_item in sorted(list(all_scraped_urls)):
                    f.write(f"{url_item}\n")
        except Exception as e:
            print(f"Warning: Could not update checkpoint file {checkpoint_file}: {e}")
    
    def _update_timeout_file(self, timeout_file: str, timed_out_urls_set: set):
        """Updates the log file for timed-out URLs."""
        try:
            os.makedirs(os.path.dirname(timeout_file), exist_ok=True)
            with open(timeout_file, "w") as f:
                for url_item in sorted(list(timed_out_urls_set)):
                    f.write(f"{url_item}\n")
        except Exception as e:
            print(f"Warning: Could not update timeout log file {timeout_file}: {e}")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "erowid_reports.csv"):
        """Saves DataFrame to CSV."""
        if df.empty:
            # print(f"DataFrame is empty. Skipping save to {filename}.")
            return

        dirname = os.path.dirname(filename)
        if dirname: os.makedirs(dirname, exist_ok=True)
        
        required_cols = ["title", "substance", "substances_list", "author", "bodyweight", "dose_chart", 
                         "report_text", "url", "gender", "age", "exp_year", "published_date"]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = "Unknown" if col != "substances_list" else pd.Series([[] for _ in range(len(df))], dtype=object)


        if "substances_list" in df.columns:
             df["substances_list"] = df["substances_list"].apply(lambda x: "|".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "Unknown"))
        
        # Ensure order and include any extra columns
        final_columns = [col for col in required_cols if col in df.columns]
        final_columns.extend([col for col in df.columns if col not in final_columns])
        
        df.to_csv(filename, index=False, columns=final_columns)
        # print(f"Saved {len(df)} reports to {filename}")
    
    def close(self):
        """Closes the WebDriver."""
        if self.driver:
            self.driver.quit()

    def _update_links_file(self, links_file_path: str, target_link: str, new_substance_info: str):
        """Updates substance info for a specific link in the links file."""
        if not new_substance_info or new_substance_info == "Unknown": return 

        lines_to_write = []
        updated = False
        try:
            if os.path.exists(links_file_path):
                with open(links_file_path, "r") as f:
                    current_lines = f.read().splitlines()
                for line_content in current_lines:
                    link_in_file, old_substance = line_content.split('|', 1) if '|' in line_content else (line_content, "Unknown")
                    if link_in_file == target_link:
                        if old_substance == "Unknown" or not old_substance:
                            lines_to_write.append(f"{target_link}|{new_substance_info}")
                            updated = True
                        else:
                            lines_to_write.append(line_content)
                    else:
                        lines_to_write.append(line_content)
            if not updated:
                lines_to_write.append(f"{target_link}|{new_substance_info}")

            dirname = os.path.dirname(links_file_path)
            if dirname: os.makedirs(dirname, exist_ok=True)
            
            with open(links_file_path, "w") as f:
                for line_to_write in lines_to_write:
                    f.write(f"{line_to_write}\n")
        except Exception as e:
            print(f"Warning: Could not update links file {links_file_path}: {e}")

def scrape_erowid_data(substances: Optional[List[str]] = None, max_reports: Optional[int] = 100,
                    output_file: str = "data/raw/erowid_reports.csv", use_safari: bool = False,
                    show_progress: bool = True) -> pd.DataFrame:
    """
    Main function to scrape Erowid experience reports.
    
    Examples:
        # Scrape 10 LSD reports
        df = scrape_erowid_data(substances=["LSD"], max_reports=10)
        # Scrape entire Erowid site (all reports, CAUTION: long time, potential IP ban)
        df = scrape_erowid_data(max_reports=None)
    """
    scraper = None
    try:
        scraper = ErowidSeleniumScraper(use_safari=use_safari)
        df = scraper.scrape_reports(substances=substances, max_reports=max_reports, 
                                    output_file=output_file, show_progress=show_progress)
        if df.empty:
            print("⚠️ WARNING: No data was scraped. This could be due to an IP ban or lack of matching reports.")
        return df
    except Exception as e:
        print(f"Error during scraping: {e}")
        traceback.print_exc()
        if "IP ban" in str(e).lower() or "banned" in str(e).lower():
            print("\n⛔ CRITICAL ERROR: IP BANNED by Erowid. Consider VPN or waiting.")
        return pd.DataFrame() 
    finally:
        if scraper:
            try:
                scraper.close()
            except Exception as e_close:
                print(f"Error closing scraper: {e_close}")


if __name__ == "__main__":
    # Example usage:
    # Scrape specific substances, e.g., 50 reports for LSD, Mushrooms, MDMA
    # substances_to_scrape = ["LSD", "Mushrooms", "MDMA"]
    # data = scrape_erowid_data(substances=substances_to_scrape, max_reports=50, use_safari=True, show_progress=True)
    data = scrape_erowid_data(substances=None, max_reports=None, use_safari=True, show_progress=True)
    
    if not data.empty:
        print(f"\nScraping complete. {len(data)} reports in DataFrame. Sample (first 3 titles):")
        print(data[['title', 'substance']].head(3))
    else:
        print("No data was scraped or an error occurred.") 