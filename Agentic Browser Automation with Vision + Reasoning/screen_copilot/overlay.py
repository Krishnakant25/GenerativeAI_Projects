"""CustomTkinter always-on-top overlay window for displaying suggestions."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

import customtkinter as ctk
import tkinter as tk

from screen_copilot.models import CopilotConfig, Suggestion

logger = logging.getLogger("copilot.overlay")


class CopilotOverlay:
    """Semi-transparent, draggable, always-on-top suggestion overlay.

    Never steals focus from the active application.
    All UI mutations must go through root.after() when called from
    a background thread.
    """

    def __init__(self, config: CopilotConfig) -> None:
        self.config = config
        self.root: Optional[ctk.CTk] = None
        self.is_running: bool = False
        self._drag_x: int = 0
        self._drag_y: int = 0
        self.current_suggestion: Optional[Suggestion] = None
        self.on_more_requested = None  # set externally by loop.py

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the overlay window. Must be called from the main thread."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Screen Copilot")
        self.root.geometry(
            f"{self.config.overlay_width}x{self.config.overlay_height}+40+40"
        )

        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-alpha", self.config.overlay_opacity)
        self.root.overrideredirect(True)

        # Best-effort: discourage the overlay from being captured as the
        # "active window". Full click-through requires native win32 API
        # calls; the title-based skip in capture.py is the primary fix.
        try:
            self.root.wm_attributes("-disabled", False)
            # Prevent the overlay from being captured as "active window"
            # by not allowing it to take focus when clicked outside drag area
        except Exception:
            pass

        # Main frame
        self.frame = ctk.CTkFrame(
            self.root, corner_radius=12, fg_color=("#1a1a2e", "#1a1a2e")
        )
        self.frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Header row
        header_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(8, 0))

        self.title_label = ctk.CTkLabel(
            header_frame,
            text="🤖 Screen Copilot",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#7c85f5",
        )
        self.title_label.pack(side="left")

        self.status_label = ctk.CTkLabel(
            header_frame,
            text="● analyzing...",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        )
        self.status_label.pack(side="right")

        self.privacy_label = ctk.CTkLabel(
            header_frame, text="🔒 Offline",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#4CAF50"
        )
        self.privacy_label.pack(side="right", padx=(0, 6))
        self.privacy_label.bind("<Button-1>", self._show_privacy_info)

        # Category badge
        self.category_label = ctk.CTkLabel(
            self.frame,
            text="",
            font=ctk.CTkFont(size=9),
            text_color="#aaaaaa",
        )
        self.category_label.pack(anchor="w", padx=12, pady=(4, 0))

        # Suggestion text box
        self.suggestion_box = ctk.CTkTextbox(
            self.frame,
            height=100,
            font=ctk.CTkFont(size=12),
            fg_color=("#0d0d1a", "#0d0d1a"),
            text_color="#e8e8f0",
            wrap="word",
            border_width=0,
            corner_radius=8,
        )
        self.suggestion_box.pack(fill="both", expand=True, padx=10, pady=6)
        self.suggestion_box.configure(state="disabled")

        # Footer
        footer_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        footer_frame.pack(fill="x", padx=10, pady=(0, 8))

        self.time_label = ctk.CTkLabel(
            footer_frame,
            text="Ctrl+Alt+S: analyze selection  |  Ctrl+Alt+D: detail",
            font=ctk.CTkFont(size=9),
            text_color="#555566",
        )
        self.time_label.pack(side="left")

        self.more_btn = ctk.CTkButton(
            footer_frame, text="Tell me more →", width=110, height=20,
            font=ctk.CTkFont(size=9),
            fg_color="#2a2a4e",
            hover_color="#4a4a7e",
            command=self._on_more_clicked
        )
        self.more_btn.pack(side="left", padx=(0, 6))

        close_btn = ctk.CTkButton(
            footer_frame,
            text="✕",
            width=24,
            height=20,
            font=ctk.CTkFont(size=10),
            fg_color="#2a2a3e",
            hover_color="#ff4444",
            command=self.stop,
        )
        close_btn.pack(side="right")

        # Drag bindings
        for widget in [
            self.frame,
            header_frame,
            self.title_label,
            self.status_label,
            self.category_label,
        ]:
            widget.bind("<ButtonPress-1>", self._on_drag_start)
            widget.bind("<B1-Motion>", self._on_drag_motion)

        self.is_running = True
        logger.info("overlay: window built")

    # ------------------------------------------------------------------
    # Drag
    # ------------------------------------------------------------------

    def _on_drag_start(self, event) -> None:
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event) -> None:
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------

    def _show_privacy_info(self, event=None) -> None:
        info = ctk.CTkToplevel(self.root)
        info.title("Privacy Guarantee")
        info.geometry("340x220+100+100")
        info.wm_attributes("-topmost", True)
        msg = (
            "🔒 PRIVACY GUARANTEE\n\n"
            "• All processing happens on THIS device\n"
            "• No screen data is ever sent to the cloud\n"
            "• Uses local Ollama models only\n"
            "• Sensitive apps are auto-skipped\n"
            "• All activity logged locally for audit\n\n"
            "Zero data leaves your machine."
        )
        label = ctk.CTkLabel(info, text=msg, justify="left",
                             font=ctk.CTkFont(size=11), wraplength=300)
        label.pack(padx=16, pady=16, fill="both", expand=True)

    # ------------------------------------------------------------------
    # Public update API — safe to call from any thread
    # ------------------------------------------------------------------

    def update_suggestion(self, suggestion: Suggestion) -> None:
        """Thread-safe suggestion update — can be called from any thread."""
        if self.root and self.is_running:
            self.root.after(0, self._apply_suggestion, suggestion)

    def _on_more_clicked(self) -> None:
        if self.on_more_requested:
            self.on_more_requested()

    def show_detail(self, detail_text: str) -> None:
        """Expand the overlay temporarily to show detailed explanation."""
        if self.root and self.is_running:
            self.root.after(0, self._apply_detail, detail_text)

    def _apply_detail(self, detail_text: str) -> None:
        # Temporarily resize taller to fit detail
        self.root.geometry(f"{self.config.overlay_width}x{self.config.overlay_height + 140}")
        self.suggestion_box.configure(state="normal")
        self.suggestion_box.insert("end", f"\n\n📖 More detail:\n{detail_text}")
        self.suggestion_box.configure(state="disabled")
        self.more_btn.configure(state="disabled", text="Loading detail shown")

    def _apply_suggestion(self, suggestion: Suggestion) -> None:
        """Apply suggestion to UI — runs on main thread via root.after()."""
        self.more_btn.configure(state="normal", text="Tell me more →")
        self.root.geometry(f"{self.config.overlay_width}x{self.config.overlay_height}")
        self.current_suggestion = suggestion

        colors = {
            "tip": "#4CAF50",
            "warning": "#FF9800",
            "info": "#2196F3",
            "general": "#9C27B0",
        }
        cat_icons = {"tip": "💡", "warning": "⚠️", "info": "ℹ️", "general": "✨"}

        cat_color = colors.get(suggestion.category, "#888888")
        cat_icon = cat_icons.get(suggestion.category, "•")

        self.category_label.configure(
            text=f"{cat_icon} {suggestion.category.upper()}",
            text_color=cat_color,
        )

        self.suggestion_box.configure(state="normal")
        self.suggestion_box.delete("1.0", "end")
        self.suggestion_box.insert("1.0", suggestion.suggestion_text)
        self.suggestion_box.configure(state="disabled")

        time_str = suggestion.timestamp.strftime("%H:%M:%S")
        self.time_label.configure(text=f"Updated {time_str}")

        conf_colors = {"high": "#4CAF50", "medium": "#FF9800", "low": "#888888"}
        self.status_label.configure(
            text=f"● {suggestion.confidence}",
            text_color=conf_colors.get(suggestion.confidence, "#888888"),
        )

    def set_status(self, status: str) -> None:
        """Update the status indicator text from any thread."""
        if self.root and self.is_running:
            self.root.after(0, lambda: self.status_label.configure(text=status))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Tk main loop. Blocks — must run on the main thread."""
        if self.root:
            self.root.mainloop()

    def stop(self) -> None:
        """Gracefully close the overlay."""
        self.is_running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
            self.root = None
        logger.info("overlay: stopped")
