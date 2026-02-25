"""
Energy News Fetcher
====================
Fetches REAL energy news from public government sources and feeds
them into the RAG vector store.

SOURCES:
    1. EIA "Today in Energy" RSS feed (free, no API key)
       → Short articles about energy markets, prices, supply/demand
       → Updated several times per week
       → RSS URL: https://www.eia.gov/rss/todayinenergy.xml

    2. ERCOT Grid Conditions (free, no API key)
       → Current grid status, alerts, operating reserves
       → Shows if the grid is under stress right now

WHY THIS MATTERS:
    Without real news, the RAG only knows generic facts from
    energy_documents.py. With real news, it knows what's happening
    TODAY — EIA reported gas prices rose, ERCOT grid is stressed, etc.
    
    This is the difference between:
        "Conservation alerts can cause price spikes" (generic)
        "EIA reported natural gas storage fell 360 Bcf last week
         amid Winter Storm Fern — largest withdrawal ever" (real, current)

CALLED BY:
    The RAG service calls fetch_and_ingest() on startup and periodically
    to keep the knowledge base updated with real-world information.
"""

import os
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

logger = logging.getLogger("news_fetcher")


def fetch_eia_news(max_articles=10):
    """
    Fetch latest energy news from EIA "Today in Energy" RSS feed.

    The RSS feed returns XML with articles like:
        <item>
            <title>EIA forecasts lower oil prices in 2026</title>
            <description>We forecast that production of petroleum...</description>
            <link>https://www.eia.gov/todayinenergy/detail.php?id=...</link>
            <pubDate>Thu, 06 Feb 2026</pubDate>
        </item>

    We extract the title + description as document text for RAG.

    Args:
        max_articles: Maximum number of articles to fetch (default 10)

    Returns:
        List of dicts: [{"text": "...", "metadata": {"source": "eia", ...}}]
    """
    url = "https://www.eia.gov/rss/todayinenergy.xml"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # Parse XML RSS feed
        root = ET.fromstring(response.content)
        articles = []

        # RSS items are at: rss > channel > item
        for item in root.findall(".//item")[:max_articles]:
            title = item.findtext("title", "")
            description = item.findtext("description", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")

            if title and description:
                # Combine title and description as one document
                text = f"{title}. {description}"

                articles.append({
                    "text": text,
                    "metadata": {
                        "source": "eia_today_in_energy",
                        "title": title,
                        "url": link,
                        "published": pub_date,
                        "fetched_at": datetime.utcnow().isoformat(),
                    },
                })

        logger.info(f"Fetched {len(articles)} articles from EIA RSS feed")
        return articles

    except Exception as e:
        logger.error(f"Failed to fetch EIA news: {e}")
        return []


def fetch_ercot_grid_conditions():
    """
    Fetch current ERCOT grid conditions.

    ERCOT publishes grid status information that indicates:
    - Current demand vs available capacity
    - Whether any alerts are active
    - Operating reserve levels

    This data is critical for the RAG because grid stress
    directly predicts price spikes.

    Returns:
        List of dicts with grid condition documents
    """
    try:
        # ERCOT public grid conditions endpoint
        url = "https://www.ercot.com/api/1/services/read/dashboards/systemConditions"
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            data = response.json()

            # Build a document from the grid conditions
            conditions = []
            if isinstance(data, dict):
                text_parts = []
                text_parts.append(
                    f"ERCOT Grid Conditions as of {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}."
                )

                if "current_condition" in data:
                    condition = data["current_condition"]
                    text_parts.append(
                        f"Current grid condition: {condition.get('state', 'unknown')}."
                    )

                if "current_demand" in data:
                    demand = data["current_demand"]
                    text_parts.append(
                        f"Current system demand: {demand} MW."
                    )

                if "operating_reserves" in data:
                    reserves = data["operating_reserves"]
                    text_parts.append(
                        f"Operating reserves: {reserves} MW."
                    )

                    # Flag low reserves
                    try:
                        if float(reserves) < 3000:
                            text_parts.append(
                                "WARNING: Operating reserves are below 3000 MW. "
                                "This indicates tight grid conditions and elevated "
                                "risk of price spikes."
                            )
                    except (ValueError, TypeError):
                        pass

                text = " ".join(text_parts)
                conditions.append({
                    "text": text,
                    "metadata": {
                        "source": "ercot_grid_conditions",
                        "fetched_at": datetime.utcnow().isoformat(),
                    },
                })

                logger.info("Fetched ERCOT grid conditions")
                return conditions

        logger.warning("ERCOT grid conditions API returned no useful data")
        return []

    except Exception as e:
        logger.warning(f"Failed to fetch ERCOT grid conditions: {e}")
        return []


def fetch_all_news():
    """
    Fetch news from all sources and combine them.

    Returns:
        Tuple of (texts, metadata_list) ready for RAG ingestion
    """
    all_articles = []

    # EIA news
    eia_articles = fetch_eia_news(max_articles=10)
    all_articles.extend(eia_articles)

    # ERCOT grid conditions
    ercot_conditions = fetch_ercot_grid_conditions()
    all_articles.extend(ercot_conditions)

    if not all_articles:
        logger.warning("No news articles fetched from any source")
        return [], []

    texts = [a["text"] for a in all_articles]
    metadata = [a["metadata"] for a in all_articles]

    logger.info(f"Total news fetched: {len(all_articles)} articles")
    return texts, metadata