#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://web-scraping.dev"


# -------------------------
# HTTP helpers
# -------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s


def get_html(session: requests.Session, url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> str:
    resp = session.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def post_json(session: requests.Session, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    resp = session.post(url, json=payload, headers=h, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -------------------------
# PRODUCTS (HTML + pagination)
# -------------------------

def parse_products_from_page(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []

    for row in soup.select("div.row.product"):
        name_el = row.select_one("h3 a")
        price_el = row.select_one("div.price")
        desc_el = row.select_one("div.short-description")
        img_el = row.select_one("div.thumbnail img")

        name = name_el.get_text(strip=True) if name_el else ""
        url = name_el["href"] if (name_el and name_el.has_attr("href")) else ""
        url = urljoin(BASE_URL, url) if url else ""

        price_txt = price_el.get_text(strip=True) if price_el else ""
        try:
            price = float(price_txt)
        except Exception:
            price = None

        short_description = desc_el.get_text(" ", strip=True) if desc_el else ""

        image = img_el["src"] if (img_el and img_el.has_attr("src")) else ""
        image = urljoin(BASE_URL, image) if image else ""

        items.append({
            "name": name,
            "url": url,
            "price": price,
            "short_description": short_description,
            "image": image,
        })

    return items


def find_total_pages_products(html: str) -> Optional[int]:
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.select_one("div.paging-meta")
    if not meta:
        return None

    text = meta.get_text(" ", strip=True)
    # Example: "page 1 of total 28 results in 6 pages"
    m = re.search(r"in\s+(\d+)\s+pages", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def scrape_products_html(session: requests.Session, per_category: bool = False, sleep: float = 0.0) -> List[Dict[str, Any]]:
    """
    Scrape products from /products?page=N .
    By default scrapes "all" category. If per_category=True, scrapes each category filter too.
    """
    products: List[Dict[str, Any]] = []

    start_urls = [f"{BASE_URL}/products"]
    if per_category:
        # Categories shown in HTML filters:
        start_urls = [
            f"{BASE_URL}/products",
            f"{BASE_URL}/products?category=apparel",
            f"{BASE_URL}/products?category=consumables",
            f"{BASE_URL}/products?category=household",
        ]

    seen_urls = set()

    for start in start_urls:
        first_html = get_html(session, start, headers={"Referer": BASE_URL})
        total_pages = find_total_pages_products(first_html) or 1

        # page param sometimes missing on first page. We'll normalize to explicit pages.
        for page in range(1, total_pages + 1):
            if page == 1:
                html = first_html
            else:
                joiner = "&" if ("?" in start) else "?"
                url = f"{start}{joiner}page={page}"
                html = get_html(session, url, headers={"Referer": start})
                if sleep:
                    time.sleep(sleep)

            page_items = parse_products_from_page(html)
            for it in page_items:
                if it["url"] and it["url"] in seen_urls:
                    continue
                if it["url"]:
                    seen_urls.add(it["url"])
                products.append(it)

    return products


# -------------------------
# REVIEWS (GraphQL pagination)
# -------------------------

REVIEWS_QUERY = """
query GetReviews($first: Int, $after: String) {
  reviews(first: $first, after: $after) {
    edges {
      node {
        rid
        text
        rating
        date
      }
      cursor
    }
    pageInfo {
      endCursor
      hasNextPage
    }
  }
}
""".strip()


def scrape_reviews_graphql(session: requests.Session, first: int = 20, max_pages: int = 100, sleep: float = 0.0) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/api/graphql"
    after: Optional[str] = None
    out: List[Dict[str, Any]] = []
    pages = 0

    while True:
        pages += 1
        if pages > max_pages:
            break

        payload = {
            "query": REVIEWS_QUERY,
            "variables": {"first": first, "after": after},
        }
        data = post_json(session, url, payload, headers={"Referer": f"{BASE_URL}/reviews"})

        if "errors" in data and data["errors"]:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        reviews = data.get("data", {}).get("reviews", {})
        edges = reviews.get("edges", []) or []
        page_info = reviews.get("pageInfo", {}) or {}

        for edge in edges:
            node = (edge or {}).get("node") or {}
            out.append({
                "rid": node.get("rid"),
                "date": node.get("date"),
                "rating": node.get("rating"),
                "text": node.get("text"),
            })

        if not page_info.get("hasNextPage"):
            break

        after = page_info.get("endCursor")
        if sleep:
            time.sleep(sleep)

    return out


# -------------------------
# TESTIMONIALS (HTMX infinite scroll)
# -------------------------

def parse_testimonials_fragment(html: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Parses either the full testimonials page HTML or the HTMX fragment HTML returned by /api/testimonials?page=N.
    Returns (testimonials, next_hx_get_url).
    """
    soup = BeautifulSoup(html, "html.parser")

    testimonials: List[Dict[str, Any]] = []
    # In full page, they live under: div.testimonials > div.testimonial
    for card in soup.select("div.testimonial"):
        # text
        text_el = card.select_one("p.text")
        text = text_el.get_text(" ", strip=True) if text_el else ""

        # rating = count of star svgs inside span.rating
        rating_el = card.select_one("span.rating")
        rating = None
        if rating_el:
            svgs = rating_el.select("svg")
            rating = len(svgs) if svgs else 0

        # author: there is no visible author name; use identicon username if present
        ident = card.select_one("identicon-svg")
        author = ""
        if ident and ident.has_attr("username"):
            author = ident["username"]

        testimonials.append({
            "author": author,
            "text": text,
            "rating": rating,
        })

    # Find next HTMX loader (the div that has hx-get)
    next_url = None
    loader = soup.select_one("div.testimonial[hx-get]")
    if loader and loader.has_attr("hx-get"):
        next_url = loader["hx-get"]
        next_url = urljoin(BASE_URL, next_url)

    return testimonials, next_url


def scrape_testimonials_htmx(session: requests.Session, max_pages: int = 50, sleep: float = 0.0) -> List[Dict[str, Any]]:
    """
    Scrapes testimonials by:
    1) GET /testimonials -> parse initial 10 (includes the loader card too, which is also a testimonial)
    2) Follow hx-get to /api/testimonials?page=2 ... until no hx-get left
    """
    first_url = f"{BASE_URL}/testimonials"
    html = get_html(session, first_url, headers={"Referer": BASE_URL})

    all_items, next_url = parse_testimonials_fragment(html)

    # This token is embedded in the page:
    # <script type="application/json" id="appData">{"x-secret-token": "secret123"}</script>
    secret_token = "secret123"

    # Headers that make HTMX endpoints happy
    htmx_headers = {
        "Referer": first_url,
        "Accept": "text/html,*/*;q=0.9",
        "HX-Request": "true",
        "X-Requested-With": "XMLHttpRequest",
        "x-secret-token": secret_token,
    }

    pages = 0
    while next_url:
        pages += 1
        if pages > max_pages:
            break

        resp = session.get(next_url, headers=htmx_headers, timeout=30)
        # If endpoint is strict, it may return 422 unless headers are correct
        resp.raise_for_status()

        chunk = resp.text
        items, next_url = parse_testimonials_fragment(chunk)
        all_items.extend(items)

        if sleep:
            time.sleep(sleep)

    # Deduplicate (same text+author+rating)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for t in all_items:
        key = (t.get("author", ""), t.get("text", ""), t.get("rating", None))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    return uniq


# -------------------------
# MAIN
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape web-scraping.dev: products, reviews, testimonials")
    parser.add_argument("--outdir", default="data", help="Output directory (default: data)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (default: 0)")
    parser.add_argument("--products-per-category", action="store_true", help="Also scrape each product category")
    parser.add_argument("--reviews-first", type=int, default=20, help="GraphQL page size for reviews (default: 20)")
    parser.add_argument("--reviews-max-pages", type=int, default=200, help="Max GraphQL pages for reviews (default: 200)")
    parser.add_argument("--testimonials-max-pages", type=int, default=200, help="Max HTMX pages for testimonials (default: 200)")

    args = parser.parse_args()
    ensure_dir(args.outdir)

    session = make_session()

    # PRODUCTS
    print("Scraping products (HTML + pagination)...")
    products = scrape_products_html(session, per_category=args.products_per_category, sleep=args.sleep)
    products_path = os.path.join(args.outdir, "products.json")
    save_json(products_path, products)
    print(f"-> {len(products)} products saved to {os.path.abspath(products_path)}")

    # TESTIMONIALS
    print("Scraping testimonials (HTMX infinite scroll)...")
    try:
        testimonials = scrape_testimonials_htmx(session, max_pages=args.testimonials_max_pages, sleep=args.sleep)
    except requests.HTTPError as e:
        # If something changes upstream, fail gracefully with a clearer message
        raise RuntimeError(
            "Testimonials HTMX scrape failed. "
            "Most common reason: missing/incorrect headers for /api/testimonials endpoint."
        ) from e

    testimonials_path = os.path.join(args.outdir, "testimonials.json")
    save_json(testimonials_path, testimonials)
    print(f"-> {len(testimonials)} testimonials saved to {os.path.abspath(testimonials_path)}")

    # REVIEWS
    print("Scraping reviews (GraphQL)...")
    reviews = scrape_reviews_graphql(session, first=args.reviews_first, max_pages=args.reviews_max_pages, sleep=args.sleep)
    reviews_path = os.path.join(args.outdir, "reviews.json")
    save_json(reviews_path, reviews)
    print(f"-> {len(reviews)} reviews saved to {os.path.abspath(reviews_path)}")

    print("Done.")


if __name__ == "__main__":
    main()
