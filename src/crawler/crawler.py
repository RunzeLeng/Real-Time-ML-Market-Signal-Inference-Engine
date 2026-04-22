import pandas as pd
import os
import requests
import re
from apify_client import ApifyClient
from dotenv import load_dotenv
from exceptions import RestartProcess


def post_filtering(df: pd.DataFrame, num_posts: int = 5) -> pd.DataFrame:
    """
    Filter out posts that are empty or contain only whitespace.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing posts with a 'content' column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only non-empty posts.
    """
    filtered_df = (
        df
        .assign(content=lambda d: d["content"].str.replace(r"<.*?>", "", regex=True).str.strip())  # remove <p> and </p>
        .query("content != ''")                                                                    # remove empty / NaN
        .loc[lambda d: ~d["content"].str.startswith("https://truthsocial.com/")]
        .loc[lambda d: ~d["content"].str.startswith("https://www")]
        .loc[lambda d: ~d["content"].str.startswith("https://dailycaller")]
        .loc[lambda d: ~d["content"].str.startswith("RT")]                                         
        .head(num_posts)                                                                           # top num_posts
    )
    return filtered_df



def post_filtering_for_multimodal(df: pd.DataFrame, num_posts: int = 5) -> pd.DataFrame:
    
    filtered_df = (
        df
        .assign(content=lambda d: d["content"].str.replace(r"<.*?>", "", regex=True).str.strip())  # remove <p> and </p>
        .loc[
            lambda d: (
                d["content"].ne("") | d["media"].apply(lambda value: bool(value))
            )
        ]                                                             # remove empty / NaN
        .loc[lambda d: ~d["content"].str.startswith("https://truthsocial.com/")]
        .loc[lambda d: ~d["content"].str.startswith("RT")]                                         
        .head(num_posts)                                                                           # top num_posts
    )
    return filtered_df



def post_formating(df: pd.DataFrame, column: str = "created_at") -> pd.DataFrame:
    
    formatted_df = df.copy()
    formatted_df[column] = pd.to_datetime(formatted_df[column], utc=True)
    formatted_df[column] = formatted_df[column].dt.floor("s").dt.tz_convert("US/Eastern")
    
    return formatted_df



def etf_formating(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    
    formatted_df = df.copy()
    formatted_df[column] = pd.to_datetime(formatted_df[column], errors="coerce")
    formatted_df[column] = formatted_df[column].dt.floor("s").dt.tz_convert("US/Eastern")

    return formatted_df



def apify_crawler_default(num_posts: int = 5) -> pd.DataFrame:
    # Load your .env file to access the API token
    load_dotenv()
    client = ApifyClient(os.getenv("APIFY_TOKEN"))

    try:
        # Prepare the Actor input
        run_input = {
            "identifiers": [
                "https://truthsocial.com/@realDonaldTrump",
                "@realDonaldTrump",
                "realDonaldTrump",
            ],
            "fetchPosts": True,
        }

        # Run the Actor and wait for it to finish
        run = client.actor("GsRHwiTFlQB8bh2yf").call(run_input=run_input)
        item = next(client.dataset(run["defaultDatasetId"]).iterate_items())

        df = post_filtering(pd.DataFrame(item["posts"])[["id", "created_at", "content"]], num_posts=num_posts)
        df = post_formating(df, column="created_at")

    except Exception as e:
        print("Error during scraping:", e)

    return df



def apify_crawler_backup(num_posts: int = 5) -> pd.DataFrame:
    # Load your .env file to access the API token
    load_dotenv()
    client = ApifyClient(os.getenv("APIFY_TOKEN"))

    # Prepare the Actor input
    run_input = {
        "username": "realDonaldTrump",
        "maxPosts": num_posts,
        "useLastPostId": False,
        "onlyReplies": False,
        "onlyMedia": False,
        "cleanContent": True,
    }

    # Run the Apify actor and fetch results
    try:
        run = client.actor("sTDLfdZAmte0aYlxg").call(run_input=run_input)
        posts = list(client.dataset(run["defaultDatasetId"]).iterate_items())

        content_only = [
            {
                "id": post.get("id"),
                "created_at": post.get("created_at"),
                "content": post.get("content")
            }
            for post in posts
        ]

        df = post_filtering(pd.DataFrame(content_only, columns=["id", "created_at", "content"]), num_posts=num_posts)
        df = post_formating(df, column="created_at")

    except Exception as e:
        print("Error during scraping:", e)

    return df



def customized_crawler(num_posts: int = 5, extract_media: bool = False, apply_multimodal_filter: bool = False) -> pd.DataFrame:
    """
    Fetches posts from the target URL, processes them, and returns a DataFrame.
    """
    SCRAPEOPS_API_KEY, SCRAPEOPS_ENDPOINT, URL, HEADERS = customized_crawler_parameters()

    response = customized_crawler_fetch_posts(SCRAPEOPS_API_KEY, SCRAPEOPS_ENDPOINT, url=URL, headers=HEADERS)
    df = customized_crawler_extract_posts(response, extract_media=extract_media)
    
    if apply_multimodal_filter:
        df = post_filtering_for_multimodal(df, num_posts=num_posts)
    else:
        df = post_filtering(df, num_posts=num_posts)
    
    df = post_formating(df, column="created_at")

    # Sort posts in descending order by "created_at"
    df.sort_values(by="created_at", ascending=False, inplace=True)
    
    return df
    
    
    
def customized_crawler_parameters():
    """
    Parameters for the customized crawler.
    """
    # Load your .env file to access the API token
    load_dotenv()
    
    SCRAPEOPS_API_KEY = os.getenv("SCRAPEOPS_API_KEY")
    SCRAPEOPS_ENDPOINT = os.getenv("SCRAPEOPS_ENDPOINT")
    BASE_URL = os.getenv("BASE_URL")

    HEADERS = {
        'accept': 'application/json, text/plain, */*',
        'referer': 'https://truthsocial.com/@realDonaldTrump'
    }
    
    params = {
        "exclude_replies": "true",
        "only_replies": "false",
        "with_muted": "true",
        "limit": "20"
    }

    URL = f"{BASE_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"

    return SCRAPEOPS_API_KEY, SCRAPEOPS_ENDPOINT, URL, HEADERS



def customized_crawler_fetch_posts(SCRAPEOPS_API_KEY, SCRAPEOPS_ENDPOINT, url, headers=None):
    """
    Makes a GET request to the target URL through the ScrapeOps proxy.
    """
    if not SCRAPEOPS_API_KEY:
        raise ValueError("Missing SCRAPEOPS_API_KEY environment variable")

    session = requests.Session()
    if headers:
        session.headers.update(headers)

    proxy_params = {
        'api_key': SCRAPEOPS_API_KEY,
        'url': url, 
        # 'bypass': 'cloudflare_level_1'
    }

    response = session.get(SCRAPEOPS_ENDPOINT, params=proxy_params, timeout=120)
    
    try:
        response.raise_for_status()
        return response.json()

    except requests.HTTPError as e:
        raise RestartProcess(f"ScrapeOps HTTP error: {e}")

    except ValueError as e:
        raise RestartProcess(f"Error parsing JSON response: {e}")



def customized_crawler_extract_posts(json_response, extract_media: bool = False) -> pd.DataFrame:
    """
    Extracts relevant data from the JSON response and returns a pandas DataFrame.
    Applies fix_unicode to the post content.
    """
    rows = []

    for post in json_response:
        rows.append({
            "id": post.get("id"),
            "created_at": post.get("created_at"),
            "content": customized_crawler_fix_unicode(post.get("content", "")).strip(),
            "media": [
                {
                    "url": media.get("url", ""),
                    "type": media.get("type", ""),
                }
                for media in post.get("media_attachments", [])
            ]
        })
    
    if extract_media:
        return pd.DataFrame(rows, columns=["id", "created_at", "content", "media"])
    else:
        return pd.DataFrame(rows, columns=["id", "created_at", "content"])



def customized_crawler_fix_unicode(text):
    """
    Ensures that escaped Unicode sequences (e.g., \u2026, \u2014)
    are converted to their proper characters.
    """
    try:
        return text.encode('utf-8').decode('unicode_escape')
    except Exception:
        return text



def filter_posts_by_date_and_content_length(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    min_content_length: int,
    date_column: str = "created_at",
    content_column: str = "content",
) -> pd.DataFrame:
    filtered_df = df.copy()

    filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
    start_date = pd.Timestamp(start_date, tz="US/Eastern")
    end_date = pd.Timestamp(end_date, tz="US/Eastern")

    filtered_df = filtered_df[
        (filtered_df[date_column] >= start_date)
        & (filtered_df[date_column] <= end_date)
    ]

    filtered_df = filtered_df[
        filtered_df[content_column].fillna("").str.len() > min_content_length
    ]

    return filtered_df



def duplicate_posts_to_minute_boundaries(
    df: pd.DataFrame,
    datetime_column: str = "created_at",
    post_duplicate: bool = True
) -> pd.DataFrame:
    
    base_df = df.copy()
    base_df[datetime_column] = pd.to_datetime(base_df[datetime_column], errors="coerce")
    base_df["created_at_seconds"] = base_df[datetime_column]

    first_df = base_df.copy()
    first_df[datetime_column] = (
        first_df[datetime_column]
        .dt.tz_convert("UTC")
        .dt.floor("min")
        .dt.tz_convert("US/Eastern")
    )

    second_df = base_df.copy()
    second_df[datetime_column] = (
        second_df[datetime_column]
        .dt.tz_convert("UTC")
        .dt.floor("min")
        .dt.tz_convert("US/Eastern")
        + pd.Timedelta(minutes=1)
    )
    
    df_rounding = base_df.copy()
    df_rounding[datetime_column] = (
        df_rounding[datetime_column]
        .dt.tz_convert("UTC")
        .dt.round("min")
        .dt.tz_convert("US/Eastern")
    )

    if post_duplicate:
        result_df = pd.concat([first_df, second_df], ignore_index=True)
    else:
        result_df = df_rounding
        
    return result_df



def add_post_prefix_to_content(df):
    prefixed_df = df.copy()
    prefixed_df["content"] = "Post: " + prefixed_df["content"].astype(str)
    return prefixed_df



if __name__ == "__main__":
    customized_crawler(num_posts=5)