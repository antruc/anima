"""Drag and drop of files onto the main window.

Toga has no drag and drop API (beeware/toga#3088), so this hooks the native
widget of each backend through the private window._impl.native. The Toga
version is pinned in pyproject.toml for that reason.
"""

import toga


def enable_drop(window, callback):
    """Call callback(paths) when files are dropped anywhere on window.

    paths is a list of str. Platforms other than Windows and Linux are left
    without drag and drop.
    """
    platform = toga.platform.current_platform
    if platform == "windows":
        _enable_winforms(window._impl.native, callback)
    elif platform == "linux":
        _enable_gtk(window._impl.native, callback)


def _all_controls(control):
    """Yield a WinForms control and all its descendants."""
    yield control
    for child in control.Controls:
        yield from _all_controls(child)


def _enable_winforms(form, callback):
    from System.Windows.Forms import DataFormats, DragDropEffects

    def on_enter(sender, e):
        if e.Data.GetDataPresent(DataFormats.FileDrop):
            e.Effect = DragDropEffects.Copy

    def on_drop(sender, e):
        callback([str(p) for p in e.Data.GetData(DataFormats.FileDrop)])

    # WinForms does not forward a drag to the form if the control under the
    # cursor does not accept it, so every control has to opt in.
    for control in _all_controls(form):
        control.AllowDrop = True
        control.DragEnter += on_enter
        control.DragDrop += on_drop


def _enable_gtk(gtk_window, callback):
    from urllib.parse import unquote, urlparse

    from gi.repository import Gdk, Gtk

    def on_received(widget, ctx, x, y, data, info, time):
        # file:// URIs come percent-encoded: c%C3%A1mara is cámara
        callback([unquote(urlparse(u).path) for u in data.get_uris() or []])

    gtk_window.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
    gtk_window.drag_dest_add_uri_targets()
    gtk_window.connect("drag-data-received", on_received)
