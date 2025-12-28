#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Flow Dataset - Example Usage Script

Project: Traffic Flow Dataset
Website: https://rskworld.in
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Founder: Molla Sameer
Designer & Tester: Rima Khatun

This script demonstrates basic usage of the traffic flow dataset.
"""

import pandas as pd
import json

# Load CSV data
print("Loading CSV data...")
# CSV file contains comment lines starting with #, so we use comment parameter
df_csv = pd.read_csv('traffic_flow_data.csv', comment='#')
print(f"CSV loaded: {len(df_csv)} records")
print(df_csv.head())
print("\n" + "="*60 + "\n")

# Load JSON data
print("Loading JSON data...")
with open('traffic_flow_data.json', 'r') as f:
    data_json = json.load(f)
df_json = pd.DataFrame(data_json)
print(f"JSON loaded: {len(df_json)} records")
print(df_json.head())
print("\n" + "="*60 + "\n")

# Basic analysis
print("Basic Statistics:")
print(f"Total records: {len(df_csv)}")
print(f"Locations: {df_csv['location'].unique()}")
print(f"Average vehicle count: {df_csv['vehicle_count'].mean():.2f}")
print(f"Average speed: {df_csv['avg_speed_kmh'].mean():.2f} km/h")
print(f"Congestion levels: {df_csv['congestion_level'].value_counts().to_dict()}")

