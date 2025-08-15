import pytest
from scripts import safety_filter as sf

def test_safe_input_blocks_sensitive():
    assert sf.is_safe_input("adult content site") is False
    assert sf.is_safe_input("normal business idea") is True

def test_safe_output_blocks_brands_and_keywords():
    assert sf.is_safe_output("facebookapp.com") is False
    assert sf.is_safe_output("myshop.porn") is False
    assert sf.is_safe_output("validbrand.io") is True

def test_filter_suggestions_removes_invalid():
    domains = ["facebookapp.com", "goodname.com", "badname.xxx", "goodname.com"]
    filtered = sf.filter_suggestions(domains)
    assert filtered == ["goodname.com"]

def test_process_request_blocks_on_input():
    out = sf.process_request("adult website", ["valid.com"])
    assert out["status"] == "blocked"
    assert out["suggestions"] == []

def test_process_request_allows_safe():
    out = sf.process_request("coffee shop", ["goodname.com", "badname.xxx"])
    assert out["status"] == "success"
    assert out["suggestions"] == [{"domain": "goodname.com", "confidence": 0.9}]
