"""Decoding package.

Existing scripts under decoding/scripts/ run as standalone files and add
decoding/ to sys.path so `from src.models import get_model` works. The CLI
under decoding/cli/ keeps that convention (see decoding/cli/__init__.py) so
nothing in src/, configs/, or scripts/ needs to change.
"""
