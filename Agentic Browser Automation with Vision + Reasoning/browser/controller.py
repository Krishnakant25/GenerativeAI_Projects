"""Browser controller — manages Playwright browser lifecycle and page interactions."""

import logging

from playwright.async_api import async_playwright, Browser, Page, Playwright

logger = logging.getLogger("browser.controller")


async def _block_resources(route, request):
    """Abort heavy resource requests — registered only when block_resources=True."""
    blocked = ["image", "media", "font"]
    # NOTE: "stylesheet" intentionally excluded — blocking CSS breaks page layout
    # and degrades llava vision quality on unstyled pages
    if request.resource_type in blocked:
        await route.abort()
    else:
        await route.continue_()


class BrowserController:
    """Thin async wrapper over a single Playwright Chromium page.

    All interaction methods are defensive: they return False (or "" for
    string getters) if the browser has not been started, and they swallow
    Playwright exceptions so the agent loop can decide how to recover.
    """

    def __init__(
        self, headless: bool = True, slow_mo: int = 0, block_resources: bool = False
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self.block_resources = block_resources
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.is_running = False

    async def start(self) -> None:
        """Launch Playwright + Chromium and open a single page."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1280, "height": 800})
        if self.block_resources:
            await self.page.route("**/*", _block_resources)
            logger.info("Browser: resource blocking enabled (media/font/image)")
        self.is_running = True
        logger.info("Browser started (headless=%s)", self.headless)

    async def stop(self) -> None:
        """Close the page, browser, and stop the Playwright driver."""
        if self.page is not None:
            await self.page.close()
            self.page = None
        if self.browser is not None:
            await self.browser.close()
            self.browser = None
        if self.playwright is not None:
            await self.playwright.stop()
            self.playwright = None
        self.is_running = False
        logger.info("Browser stopped")

    async def navigate(self, url: str) -> bool:
        """Navigate to a URL. Returns True on success, False on failure."""
        if not self.is_running or self.page is None:
            logger.warning("navigate called but browser is not running")
            return False
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return True
        except Exception as exc:
            logger.error("navigate(%s) failed: %s", url, exc)
            return False

    async def click(self, x: int, y: int) -> bool:
        """Click at absolute viewport coordinates."""
        if not self.is_running or self.page is None:
            logger.warning("click called but browser is not running")
            return False
        try:
            await self.page.mouse.click(x, y)
            await self.page.wait_for_timeout(500)
            return True
        except Exception as exc:
            logger.error("click(%d, %d) failed: %s", x, y, exc)
            return False

    async def click_text(self, text: str) -> bool:
        """Click the first element matching the given text.

        Fallback used when vision-derived coordinate clicks fail.
        """
        if not self.is_running or self.page is None:
            logger.warning("click_text called but browser is not running")
            return False
        try:
            await self.page.get_by_text(text).first.click()
            await self.page.wait_for_timeout(500)
            return True
        except Exception as exc:
            logger.error("click_text(%r) failed: %s", text, exc)
            return False

    async def type_text(self, selector: str, text: str) -> bool:
        """Focus a selector and type text into it."""
        if not self.is_running or self.page is None:
            logger.warning("type_text called but browser is not running")
            return False
        try:
            await self.page.click(selector)
            await self.page.wait_for_timeout(200)
            await self.page.keyboard.type(text)
            return True
        except Exception as exc:
            logger.error("type_text(%r) failed: %s", selector, exc)
            return False

    async def scroll(self, direction: str, amount: int = 300) -> bool:
        """Scroll the page up or down by `amount` pixels."""
        if not self.is_running or self.page is None:
            logger.warning("scroll called but browser is not running")
            return False
        try:
            delta = -amount if direction == "up" else amount
            await self.page.mouse.wheel(0, delta)
            return True
        except Exception as exc:
            logger.error("scroll(%s, %d) failed: %s", direction, amount, exc)
            return False

    async def get_current_url(self) -> str:
        """Return the current page URL, or "" if no page is open."""
        if self.page is None:
            return ""
        return self.page.url

    async def wait_for_load(self, timeout: int = 5000) -> None:
        """Best-effort wait for network idle. Timeouts are swallowed."""
        if not self.is_running or self.page is None:
            return
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except Exception as exc:
            logger.debug("wait_for_load networkidle timed out: %s", exc)
