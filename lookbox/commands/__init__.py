"""Every document mutation lives in this package (ARCHITECTURE §4.2).

`edits` is Qt-free; `qt` adapts edits to QUndoStack. Import `qt` explicitly
where Qt is available so tests can use `edits` headless.
"""
