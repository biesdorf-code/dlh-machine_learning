#!/usr/bin/env python3
"""insert a document argument in a collection"""


def insert_school(mongo_collection, **kwargs):
    """Insert a document built on the fly, returns _id"""
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
