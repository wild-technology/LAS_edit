"""Tests for statusbar _format_count function."""
from pointcloud_editor.editor.statusbar import _format_count


class TestFormatCount:
    def test_small_number(self):
        assert _format_count(42) == "42"

    def test_zero(self):
        assert _format_count(0) == "0"

    def test_thousands(self):
        assert _format_count(1_500) == "1.5K"

    def test_exact_thousand(self):
        assert _format_count(1_000) == "1.0K"

    def test_millions(self):
        assert _format_count(2_500_000) == "2.5M"

    def test_exact_million(self):
        assert _format_count(1_000_000) == "1.0M"

    def test_large_millions(self):
        assert _format_count(150_000_000) == "150.0M"
