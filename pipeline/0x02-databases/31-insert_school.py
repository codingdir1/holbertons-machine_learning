#!/usr/bin/env python3

def insert_school(mongo_collection, **kwargs):
    mongo_collection.insert_one(kwargs)
    return kwargs["_id"]
