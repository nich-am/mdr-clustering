"""
scripts/keepalive_streamlit.py
-------------------------------
Visits the Streamlit Community Cloud app with a real headless browser
(not a bare HTTP request) and clicks the "Yes, get this app back up!"
button if the app is asleep.

Why this can't be done with curl:
Streamlit Cloud's wake flow depends on a cookie-based auth redirect
(_streamlit_csrf / streamlit_session). A plain HTTP client with no
cookie jar just bounces between /-/auth/app and /-/login forever,
which is exactly the "too many redirects" (curl exit 47) failure.
A GET request can also return HTTP 200 while only fetching a static
HTML shell — the actual Python app never boots without a real
browser executing JS and opening a WebSocket connection.
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

APP_URL = os.environ.get("STREAMLIT_APP_URL", "").strip()
WAKE_BUTTON_TEXT = "Yes, get this app back up!"


def _app_is_rendered(page) -> bool:
    """
    Check for the real app container on the main page AND inside any
    nested iframe — Streamlit Cloud sometimes renders the app inside
    a /~/+/ iframe rather than directly on the top-level page.
    """
    try:
        if page.locator('[data-testid="stApp"]').count() > 0:
            return True
    except Exception:
        pass
    for frame in page.frames:
        try:
            if frame.locator('[data-testid="stApp"]').count() > 0:
                return True
        except Exception:
            continue
    return False


def _find_wake_button(page):
    """
    Try a stable data-testid selector first, fall back to matching
    the button's visible text in case the testid ever changes.
    """
    by_testid = page.locator('[data-testid="wakeup-button-viewer"]')
    if by_testid.count() > 0:
        return by_testid.first
    by_text = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
    if by_text.count() > 0:
        return by_text.first
    return None


def main() -> int:
    if not APP_URL or "REPLACE" in APP_URL:
        print("::error::STREAMLIT_APP_URL is not set — edit the workflow env var.")
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        print(f"Visiting {APP_URL} ...")
        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        except PWTimeout:
            print("::error::Timed out loading the app at all — check the URL.")
            browser.close()
            return 1

        # The sleep screen is client-rendered, so give it a moment to appear
        # before deciding whether the app is asleep or already awake.
        page.wait_for_timeout(3_000)

        wake_btn = _find_wake_button(page)
        if wake_btn is not None:
            print("App is asleep — clicking wake-up button ...")
            wake_btn.click()

            # Cold starts can take a couple of minutes.
            booted = False
            for _ in range(24):  # poll for up to ~2 minutes
                page.wait_for_timeout(5_000)
                if _app_is_rendered(page):
                    booted = True
                    break

            if booted:
                print("App booted successfully after wake click.")
            else:
                print(
                    "::warning::Clicked wake button but app didn't finish "
                    "booting within 2 minutes — it may still be starting up. "
                    "This isn't necessarily a failure; Streamlit cold starts "
                    "can occasionally take longer."
                )
        else:
            # No wake button — confirm the real app actually rendered,
            # not just an HTML shell.
            if _app_is_rendered(page):
                print("App was already awake and rendered correctly.")
            else:
                print(
                    "::error::No wake button found, but the app container "
                    "never rendered either — something unexpected happened."
                )
                browser.close()
                return 1

        browser.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
