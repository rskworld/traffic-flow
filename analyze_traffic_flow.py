#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Flow Dataset Analysis Script

Project: Traffic Flow Dataset
Website: https://rskworld.in
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Founder: Molla Sameer
Designer & Tester: Rima Khatun

This script analyzes traffic flow data including vehicle counts,
speed measurements, and congestion patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(csv_file='traffic_flow_data.csv'):
    """
    Load traffic flow data from CSV file
    
    Args:
        csv_file (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded traffic flow data
    """
    print(f"Loading data from {csv_file}...")
    # CSV file contains comment lines starting with #, so we use comment parameter
    df = pd.read_csv(csv_file, comment='#')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} records")
    return df

def basic_statistics(df):
    """
    Display basic statistics about the traffic flow data
    
    Args:
        df (pd.DataFrame): Traffic flow dataframe
    """
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(f"\nTotal Records: {len(df)}")
    print(f"\nDate Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nLocations: {df['location'].unique()}")
    print(f"\nRoad Types: {df['road_type'].unique()}")
    
    print("\n" + "-"*60)
    print("NUMERICAL STATISTICS")
    print("-"*60)
    print(df[['vehicle_count', 'avg_speed_kmh']].describe())
    
    print("\n" + "-"*60)
    print("CONGESTION LEVEL DISTRIBUTION")
    print("-"*60)
    print(df['congestion_level'].value_counts())

def analyze_by_location(df):
    """
    Analyze traffic patterns by location
    
    Args:
        df (pd.DataFrame): Traffic flow dataframe
    
    Returns:
        pd.DataFrame: Summary statistics by location
    """
    print("\n" + "="*60)
    print("ANALYSIS BY LOCATION")
    print("="*60)
    
    def get_mode(series):
        """Get mode of a series, return first value or 'Unknown' if empty"""
        mode_result = series.mode()
        return mode_result.iloc[0] if len(mode_result) > 0 else 'Unknown'
    
    location_stats = df.groupby('location').agg({
        'vehicle_count': ['mean', 'max', 'min', 'std'],
        'avg_speed_kmh': ['mean', 'max', 'min', 'std'],
        'congestion_level': get_mode
    }).round(2)
    
    print(location_stats)
    return location_stats

def analyze_by_time(df):
    """
    Analyze traffic patterns by time of day
    
    Args:
        df (pd.DataFrame): Traffic flow dataframe
    """
    print("\n" + "="*60)
    print("ANALYSIS BY TIME OF DAY")
    print("="*60)
    
    df['hour'] = df['timestamp'].dt.hour
    time_stats = df.groupby('hour').agg({
        'vehicle_count': 'mean',
        'avg_speed_kmh': 'mean'
    }).round(2)
    
    print(time_stats)
    return time_stats

def analyze_congestion_patterns(df):
    """
    Analyze congestion patterns
    
    Args:
        df (pd.DataFrame): Traffic flow dataframe
    """
    print("\n" + "="*60)
    print("CONGESTION PATTERN ANALYSIS")
    print("="*60)
    
    congestion_stats = df.groupby('congestion_level').agg({
        'vehicle_count': ['mean', 'max'],
        'avg_speed_kmh': ['mean', 'min']
    }).round(2)
    
    print(congestion_stats)
    return congestion_stats

def create_visualizations(df):
    """
    Create visualizations for traffic flow data
    
    Args:
        df (pd.DataFrame): Traffic flow dataframe
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Traffic Flow Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Vehicle Count Over Time
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('timestamp')
    ax1.plot(df_sorted['timestamp'], df_sorted['vehicle_count'], linewidth=2, color='#667eea')
    ax1.set_title('Vehicle Count Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Vehicle Count')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Average Speed Over Time
    ax2 = axes[0, 1]
    ax2.plot(df_sorted['timestamp'], df_sorted['avg_speed_kmh'], linewidth=2, color='#764ba2')
    ax2.set_title('Average Speed Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Average Speed (km/h)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Vehicle Count by Location
    ax3 = axes[1, 0]
    location_avg = df.groupby('location')['vehicle_count'].mean().sort_values(ascending=False)
    ax3.bar(location_avg.index, location_avg.values, color='#667eea', alpha=0.8)
    ax3.set_title('Average Vehicle Count by Location', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Location')
    ax3.set_ylabel('Average Vehicle Count')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Congestion Level Distribution
    ax4 = axes[1, 1]
    congestion_counts = df['congestion_level'].value_counts()
    colors = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
    congestion_colors = [colors.get(level, '#9E9E9E') for level in congestion_counts.index]
    ax4.pie(congestion_counts.values, labels=congestion_counts.index, autopct='%1.1f%%',
            colors=congestion_colors, startangle=90)
    ax4.set_title('Congestion Level Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('traffic_flow_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'traffic_flow_analysis.png'")
    plt.show()

def correlation_analysis(df):
    """
    Perform correlation analysis between variables
    
    Args:
        df (pd.DataFrame): Traffic flow dataframe
    """
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Convert congestion level to numeric for correlation
    congestion_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df_numeric = df.copy()
    df_numeric['congestion_numeric'] = df_numeric['congestion_level'].map(congestion_map)
    
    corr_matrix = df_numeric[['vehicle_count', 'avg_speed_kmh', 'congestion_numeric']].corr()
    print(corr_matrix)
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")
    plt.show()

def export_summary_report(df):
    """
    Export summary report to JSON
    
    Args:
        df (pd.DataFrame): Traffic flow dataframe
    """
    print("\n" + "="*60)
    print("EXPORTING SUMMARY REPORT")
    print("="*60)
    
    report = {
        'summary': {
            'total_records': int(len(df)),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'locations': df['location'].unique().tolist(),
            'road_types': df['road_type'].unique().tolist()
        },
        'statistics': {
            'vehicle_count': {
                'mean': float(df['vehicle_count'].mean()),
                'std': float(df['vehicle_count'].std()),
                'min': int(df['vehicle_count'].min()),
                'max': int(df['vehicle_count'].max())
            },
            'avg_speed_kmh': {
                'mean': float(df['avg_speed_kmh'].mean()),
                'std': float(df['avg_speed_kmh'].std()),
                'min': float(df['avg_speed_kmh'].min()),
                'max': float(df['avg_speed_kmh'].max())
            }
        },
        'by_location': df.groupby('location').agg({
            'vehicle_count': 'mean',
            'avg_speed_kmh': 'mean'
        }).to_dict('index'),
        'congestion_distribution': df['congestion_level'].value_counts().to_dict()
    }
    
    with open('traffic_flow_summary.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("Summary report exported to 'traffic_flow_summary.json'")

def main():
    """
    Main function to run all analyses
    """
    print("\n" + "="*60)
    print("TRAFFIC FLOW DATASET ANALYSIS")
    print("="*60)
    print("Website: https://rskworld.in")
    print("Contact: help@rskworld.in, support@rskworld.in")
    print("Phone: +91 93305 39277")
    print("="*60 + "\n")
    
    # Load data
    df = load_data()
    
    # Perform analyses
    basic_statistics(df)
    analyze_by_location(df)
    analyze_by_time(df)
    analyze_congestion_patterns(df)
    correlation_analysis(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Export summary report
    export_summary_report(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

