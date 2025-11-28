import os
import re
import json
import time
import threading
from urllib.parse import urlparse, urljoin

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

load_dotenv()

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__)
print("=== SERVER STARTING ===", flush=True)

# -------------------------------------------------
# Groq Client
# -------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -------------------------------------------------
# Selenium Browser Manager
# -------------------------------------------------
class Browser:
    @staticmethod
    def create():
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.binary_location = os.getenv("CHROME_BIN", "/usr/bin/chromium")

        service = Service(
            executable_path=os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
        )
        return webdriver.Chrome(service=service, options=chrome_options)

    @staticmethod
    def fetch(url, wait=3):
        """Load a page with Selenium and return rendered HTML + text."""
        print(f"[BROWSER] Fetching: {url}", flush=True)

        driver = Browser.create()
        try:
            driver.get(url)
            time.sleep(wait)

            content = driver.find_element(By.TAG_NAME, "body").text
            html = driver.page_source

            print(f"[BROWSER] Got content: {content[:200]}", flush=True)
            return content, html
        except Exception as e:
            print(f"[BROWSER] Error fetching {url}: {e}", flush=True)
            return "", ""
        finally:
            driver.quit()


# -------------------------------------------------
# Resource Fetcher
# -------------------------------------------------
class ResourceFetcher:
    LINK_PATTERNS = [
        r'href=["\']([^"\']+)["\']',
        r"Scrape\s+<?a?\s*(?:href=[\"'])?([^\s\"'<>]+)",
        r"download[^\s]*\s+([^\s]+)",
        r'CSV file[^\n]*href=["\']([^"\']+)["\']',
    ]

    @staticmethod
    def extract_urls(html, text):
        collected = []
        combined = html + text

        for pattern in ResourceFetcher.LINK_PATTERNS:
            collected.extend(re.findall(pattern, combined, re.IGNORECASE))

        return list(set(collected))

    @staticmethod
    def make_absolute(base_url, link):
        if link.startswith("/"):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{link}"
        if link.startswith("http"):
            return link
        return urljoin(base_url, link)

    @staticmethod
    def fetch_resource(url):
        """Fetch a URL with fallback to browser if needed."""
        print(f"[FETCH] Fetching resource: {url}", flush=True)
        try:
            resp = requests.get(url, timeout=15)
            content = resp.text

            # If response is short but has script tags, JS-rendering required
            if "<script" in content and len(content.strip()) < 500:
                print("[FETCH] JS detected, using browser...", flush=True)
                txt, _ = Browser.fetch(url)
                content = txt

            print(f"[FETCH] Final content: {content[:200]}", flush=True)
            return content[:10000]
        except Exception as e:
            print(f"[FETCH] Error fetching {url}: {e}", flush=True)
            return ""

    @staticmethod
    def fetch_all(base_url, html, text):
        urls = ResourceFetcher.extract_urls(html, text)
        resources = {}

        for url in urls:
            if url.startswith("#") or url.lower().startswith("javascript"):
                continue
            abs_url = ResourceFetcher.make_absolute(base_url, url)

            if "submit" in abs_url.lower():
                continue

            content = ResourceFetcher.fetch_resource(abs_url)
            resources[abs_url] = content

        return resources


# -------------------------------------------------
# Data Helper
# -------------------------------------------------
class DataAnalyzer:
    @staticmethod
    def sum_csv(csv_text, cutoff=None):
        lines = csv_text.strip().splitlines()
        numbers = []

        for line in lines:
            for val in line.split(","):
                try:
                    numbers.append(float(val.strip()))
                except ValueError:
                    pass

        if cutoff is not None:
            numbers = [n for n in numbers if n > cutoff]

        return sum(numbers)


# -------------------------------------------------
# Quiz Solver
# -------------------------------------------------
class QuizSolver:
    @staticmethod
    def groq_solve(prompt):
        print("[SOLVE] Calling Groq API...", flush=True)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    @staticmethod
    def parse_llm_json(text):
        match = re.search(r'\{[^{}]*"submit_url"[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
        if not match:
            match = re.search(r"{.*}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM output")

        result = json.loads(match.group(0))
        if not result.get("answer"):
            raise ValueError("LLM returned empty answer")
        return result

    @staticmethod
    def solve(url):
        print(f"[SOLVE] Loading quiz page: {url}", flush=True)
        text, html = Browser.fetch(url)

        resources = ResourceFetcher.fetch_all(url, html, text)

        # Detect cutoff
        cutoff_match = re.search(r"[Cc]utoff[:\s]+(\d+)", text)
        cutoff = int(cutoff_match.group(1)) if cutoff_match else None

        # Try auto-calc
        pre_calc = None
        for r_url, content in resources.items():
            if ".csv" in r_url or any(c.isdigit() for c in content[:200]):
                pre_calc = DataAnalyzer.sum_csv(content, cutoff)

        resource_text = "\n".join(
            f"--- {u} ---\n{c}\n" for u, c in resources.items()
        )

        if pre_calc is not None:
            resource_text += f"\n\nPRE-CALCULATED: {pre_calc}"

        # Build prompt
        prompt = f"""
You are solving a data analysis quiz. Provide ONLY a JSON result.

PAGE CONTENT:
{text}

HTML:
{html[:3000]}

FETCHED RESOURCES:
{resource_text}

Return ONLY:
{{"submit_url": "/submit", "answer": VALUE}}
"""

        llm_text = QuizSolver.groq_solve(prompt)
        print(f"[SOLVE] LLM Response: {llm_text[:400]}", flush=True)

        return QuizSolver.parse_llm_json(llm_text)


# -------------------------------------------------
# Quiz Flow Manager
# -------------------------------------------------
def process_quiz(start_url, email, secret):
    print("[PROCESS] Starting quiz...", flush=True)

    url = start_url
    for step in range(15):
        print(f"[QUIZ {step+1}] {url}", flush=True)

        try:
            result = QuizSolver.solve(url)
            submit_url = result["submit_url"]
            answer = result["answer"]

            # Build absolute submit URL
            if submit_url.startswith("/"):
                parsed = urlparse(url)
                submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"

            payload = {
                "email": email,
                "secret": secret,
                "url": url,
                "answer": answer,
            }

            print(f"[SUBMIT] → {submit_url}  Answer={answer}", flush=True)
            resp = requests.post(submit_url, json=payload, timeout=30).json()

            print(f"[RESULT] {resp}", flush=True)

            url = resp.get("url")
            if resp.get("delay"):
                time.sleep(resp["delay"])

            if not url:
                break

        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            break

    print("[PROCESS] Quiz complete.", flush=True)


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    print("[ENDPOINT] POST /quiz", flush=True)
    data = request.get_json() or {}

    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    if not email or not secret or not url:
        return jsonify({"error": "Missing required fields"}), 400

    if secret != os.getenv("SECRET"):
        return jsonify({"error": "Invalid secret"}), 403

    threading.Thread(target=process_quiz, args=(url, email, secret)).start()
    return jsonify({"status": "processing"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    print(f"=== Starting server on port {port} ===", flush=True)
    app.run(host="0.0.0.0", port=port)
