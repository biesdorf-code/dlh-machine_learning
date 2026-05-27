#!/usr/bin/env python3
"""schools by topic"""


def schools_by_topic(mongo_collection, topic):
    """return a list of schools documents with topic"""
    return list(mongo_collection.find({"topics": topic}))
