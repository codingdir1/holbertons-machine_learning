#!/usr/bin/env python3
"""
34. Log stats
script that provides some stats about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    log_collection = client.logs.nginx
    total_docs = log_collection.count_documents({})
    print("{0} logs".format(total_docs))

    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        method_count = log_collection.count_documents({"mehtod" : method})
        print("\tmethod {0}: {1}".format(method, method_count))
    path_count = log_collection.count_documents({"method" : "GET", "path" : "/status"})
    print("{0} status check".format(path_count))

