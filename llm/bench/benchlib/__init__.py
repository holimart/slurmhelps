"""benchctl implementation package (stdlib only, Python 3.9+).

The shell scripts under llm/{strata,qwen38,qwen35,glm53-reap50} stay the source
of truth for a measurement.  benchlib only validates recipes, resolves paths,
builds the same command line the adapters use, and reads result artifacts.
"""
