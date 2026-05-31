"""Shared CLI helpers for MPlayer3 tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

try:
    from rich.console import Console
except ImportError:  # pragma: no cover - optional dependency
    Console = None

try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.containers import Container
    from textual.widgets import Input, Label, ListItem, ListView, Static
except ImportError:  # pragma: no cover - optional dependency
    App = None
    ComposeResult = None
    Container = None
    Input = None
    Label = None
    ListItem = None
    ListView = None
    Static = None
    on = None

APP_NAME = "MPlayer3"


def get_console() -> Optional["Console"]:
    if Console is None:
        return None
    return Console()

def require_console() -> "Console":
    if Console is None:
        print("Error: The 'rich' library is required but not installed.")
        print("Install it with: pip install rich")
        raise SystemExit(1)
    return Console()


def require_textual() -> None:
    if App is None:
        print("Error: The 'textual' library is required for interactive song selection.")
        print("Install it with: pip install textual")
        raise SystemExit(1)


def print_banner(title: str, console: Optional["Console"] = None) -> None:
    text = f"{APP_NAME} - {title}"
    if console:
        console.print(f"[bold cyan]{text}[/bold cyan]")
    else:
        print(text)


def _prefix_message(message: str, console: Optional["Console"] = None) -> str:
    if console:
        return f"[bold cyan]{APP_NAME}[/bold cyan] {message}"
    return f"{APP_NAME}: {message}"


def print_message(message: str, console: Optional["Console"] = None) -> None:
    if console:
        console.print(_prefix_message(message, console))
    else:
        print(_prefix_message(message, console))

@dataclass
class RenameChoice:
    prompt: str
    choices: Sequence[str]
    help_text: str
    empty_message: str


if App is not None:
    class RenameSongItem(ListItem):
        def __init__(self, value: str) -> None:
            super().__init__(Label(value))
            self.value = value


    class RenameSongApp(App[int]):
        CSS = """
        Screen {
            align: center middle;
        }

        #dialog {
            width: 92%;
            height: 88%;
            padding: 1 2;
            border: round $accent;
            background: $surface;
        }

        #prompt {
            padding-bottom: 1;
            text-style: bold;
        }

        #songs {
            height: 1fr;
            border: round $panel;
            margin-bottom: 1;
        }

        #rename_label {
            padding-top: 1;
        }

        #new_name {
            margin-bottom: 1;
        }

        #help {
            color: $text-muted;
            height: auto;
        }

        #empty {
            padding-top: 1;
            color: $warning;
        }
        """

        BINDINGS = [
            ("enter", "submit", "Continue"),
            ("tab", "toggle_focus", "Next Field"),
            ("shift+tab", "toggle_focus", "Prev Field"),
            ("escape", "cancel", "Cancel"),
            ("ctrl+c", "cancel", "Cancel"),
            ("ctrl+s", "save", "Save"),
        ]

        def __init__(
            self,
            choice: RenameChoice,
            rename_handler: Callable[[str, str], tuple[bool, str]],
        ) -> None:
            super().__init__()
            self.choice = choice
            self.rename_handler = rename_handler
            self.choices = sorted(choice.choices, key=str.casefold)
            self.selected_name: str | None = None
            self.rename_count = 0

        def compose(self) -> ComposeResult:
            with Container(id="dialog"):
                yield Static(self.choice.prompt, id="prompt")
                yield ListView(id="songs")
                yield Static("Rename selected file:", id="rename_label")
                yield Input(id="new_name")
                yield Static(self.choice.help_text, id="help")
                yield Static("", id="empty")

        def on_mount(self) -> None:
            self.title = f"{APP_NAME} Rename"
            self._refresh_choices()
            if self.choices:
                self.query_one(ListView).focus()

        @on(ListView.Highlighted)
        def _on_list_highlighted(self, event: ListView.Highlighted) -> None:
            item = event.item
            if isinstance(item, RenameSongItem):
                self._sync_selection(item.value)

        @on(ListView.Selected)
        def _on_list_selected(self, event: ListView.Selected) -> None:
            item = event.item
            if isinstance(item, RenameSongItem):
                self._sync_selection(item.value)
                self.query_one(Input).focus()

        @on(Input.Submitted)
        def _on_input_submitted(self) -> None:
            self.action_save()

        def action_submit(self) -> None:
            focused = self.focused
            if isinstance(focused, ListView):
                self.query_one(Input).focus()
                return
            self.action_save()

        def action_toggle_focus(self) -> None:
            focused = self.focused
            if isinstance(focused, ListView):
                self.query_one(Input).focus()
            else:
                self.query_one(ListView).focus()

        def action_cancel(self) -> None:
            self.exit(self.rename_count)

        def action_save(self) -> None:
            if self.selected_name is None:
                self.exit(self.rename_count)
                return

            input_widget = self.query_one(Input)
            new_name = input_widget.value.strip()
            if not new_name:
                self.query_one("#empty", Static).update("Filename cannot be empty.")
                return

            if not new_name.lower().endswith(".mp3"):
                new_name = f"{new_name}.mp3"

            if new_name == ".mp3":
                self.query_one("#empty", Static).update("Filename cannot be empty.")
                return

            old_name = self.selected_name
            if new_name == old_name:
                self.notify(
                    "Filename is unchanged.",
                    title="No rename applied",
                    severity="warning",
                    timeout=2.5,
                )
                self.query_one(ListView).focus()
                return

            renamed, message = self.rename_handler(old_name, new_name)
            if not renamed:
                self.query_one("#empty", Static).update(message)
                self.notify(
                    message,
                    title="Rename failed",
                    severity="error",
                    timeout=4,
                )
                return

            self.rename_count += 1
            self._replace_choice(old_name, new_name)
            self._refresh_choices(selected_name=new_name)
            self.query_one(ListView).focus()
            self.notify(
                message,
                title="Renamed",
                severity="information",
                timeout=2.5,
            )

        def _sync_selection(self, value: str) -> None:
            self.selected_name = value
            self.query_one(Input).value = value
            self.query_one("#empty", Static).update("")

        def _replace_choice(self, old_name: str, new_name: str) -> None:
            updated_choices = [name for name in self.choices if name != old_name]
            updated_choices.append(new_name)
            self.choices = sorted(updated_choices, key=str.casefold)

        def _refresh_choices(self, selected_name: str | None = None) -> None:
            list_view = self.query_one(ListView)
            list_view.clear()

            if not self.choices:
                self.selected_name = None
                self.query_one(Input).value = ""
                self.query_one("#empty", Static).update(self.choice.empty_message)
                return

            list_view.extend(RenameSongItem(name) for name in self.choices)

            if selected_name in self.choices:
                next_index = self.choices.index(selected_name)
            else:
                next_index = 0

            list_view.index = next_index
            self._sync_selection(self.choices[next_index])


def run_rename_loop(
    choice: RenameChoice,
    rename_handler: Callable[[str, str], tuple[bool, str]],
) -> int:
    require_textual()
    if not choice.choices:
        return 0
    app = RenameSongApp(choice, rename_handler)
    return app.run()
