#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a hierarchical directory structure for daily, weekly, and monthly
organization, spanning the next 12 months from the current date.
The script is designed to be idempotent and directly executable after
setting permissions (e.g., chmod +x script_name.py on Linux/macOS).

Structure:
- ./daily/YYYY/MonthName/MMDDYYYY
- ./weekly/YYYY/MonthName/WeekOfMMDDYYYY
- ./monthly/YYYY/MonthName
"""

import os
import datetime

def create_project_directories(base_path="."):
    """
    Creates or updates the directory structure.

    Args:
        base_path (str): The root directory where 'daily', 'weekly',
                         and 'monthly' folders will be created.
                         Defaults to the current directory.
    """

    today = datetime.date.today()
    print(f"Starting directory creation/update for 12 months from: {today.strftime('%Y-%m-%d')}")
    print(f"Target base path: {os.path.abspath(base_path)}")

    # --- 1. Create top-level directories ---
    # These are 'daily', 'weekly', 'monthly' directly under base_path.
    top_level_dirs = ["daily", "weekly", "monthly"]
    for dir_name in top_level_dirs:
        path = os.path.join(base_path, dir_name)
        # exist_ok=True makes os.makedirs idempotent; it won't raise an error
        # if the directory already exists.
        os.makedirs(path, exist_ok=True)

    # --- Define the period for directory creation ---
    # The script will create directories for dates up to, but not including,
    # one year from today.
    # Example: If today is 2025-05-28, it creates folders for dates up to 2026-05-27.
    one_year_from_today = datetime.date(today.year + 1, today.month, today.day)

    # --- 2. Create "daily" folders ---
    # Structure: daily/YYYY/MonthName/MMDDYYYY
    print("\nProcessing daily directories...")
    current_date_for_daily = today
    days_processed_daily = 0
    while current_date_for_daily < one_year_from_today:
        year_str = current_date_for_daily.strftime("%Y")
        month_name_str = current_date_for_daily.strftime("%B") # Full month name, e.g., "May"
        day_folder_name = current_date_for_daily.strftime("%m%d%Y") # e.g., 05282025

        # Construct the full path for the specific day's folder
        daily_path = os.path.join(base_path, "daily", year_str, month_name_str, day_folder_name)

        if not os.path.exists(daily_path):
            os.makedirs(daily_path, exist_ok=True)
            # print(f"  Created: {daily_path}") # Uncomment for verbose output
        # else:
            # print(f"  Exists: {daily_path}") # Uncomment for verbose output
        current_date_for_daily += datetime.timedelta(days=1)
        days_processed_daily +=1
    print(f"Daily directory processing complete. Checked/created {days_processed_daily} day folders.")

    # --- 3. Create "weekly" folders (for each Sunday) ---
    # Structure: weekly/YYYY/MonthName/WeekOfMMDDYYYY
    print("\nProcessing weekly directories...")
    # Determine the first Sunday to process.
    # If today is Sunday, start with today. Otherwise, find the next Sunday.
    current_sunday = today
    if today.weekday() != 6:  # Sunday is 6 in datetime.weekday() (Monday is 0)
        days_to_sunday = (6 - today.weekday() + 7) % 7
        current_sunday = today + datetime.timedelta(days=days_to_sunday)

    weeks_processed = 0
    while current_sunday < one_year_from_today:
        year_str = current_sunday.strftime("%Y")
        month_name_str = current_sunday.strftime("%B")
        sunday_date_str = current_sunday.strftime("%m%d%Y") # Date of the Sunday
        week_folder_name = f"WeekOf{sunday_date_str}" # e.g., WeekOf06012025

        # Construct the full path for the specific week's folder
        weekly_path = os.path.join(base_path, "weekly", year_str, month_name_str, week_folder_name)

        if not os.path.exists(weekly_path):
            os.makedirs(weekly_path, exist_ok=True)
            # print(f"  Created: {weekly_path}") # Uncomment for verbose output
        # else:
            # print(f"  Exists: {weekly_path}") # Uncomment for verbose output
        
        current_sunday += datetime.timedelta(weeks=1) # Move to the next Sunday
        weeks_processed += 1
    print(f"Weekly directory processing complete. Checked/created {weeks_processed} week folders.")

    # --- 4. Create "monthly" folders ---
    # Structure: monthly/YYYY/MonthName
    print("\nProcessing monthly directories...")
    # Start from the first day of the current month
    current_month_iterator = datetime.date(today.year, today.month, 1)
    months_processed = 0
    for _ in range(12): # Create for the current month and the next 11 months
        year_str = current_month_iterator.strftime("%Y")
        month_name_str = current_month_iterator.strftime("%B")

        # Construct the full path for the specific month's folder
        monthly_path = os.path.join(base_path, "monthly", year_str, month_name_str)

        if not os.path.exists(monthly_path):
            os.makedirs(monthly_path, exist_ok=True)
            # print(f"  Created: {monthly_path}") # Uncomment for verbose output
        # else:
            # print(f"  Exists: {monthly_path}") # Uncomment for verbose output

        # Advance to the first day of the next month
        if current_month_iterator.month == 12: # If December, next month is January of next year
            current_month_iterator = datetime.date(current_month_iterator.year + 1, 1, 1)
        else:
            current_month_iterator = datetime.date(current_month_iterator.year, current_month_iterator.month + 1, 1)
        months_processed += 1
    print(f"Monthly directory processing complete. Checked/created {months_processed} month folders.")

    print("\nDirectory structure creation/update process finished.")

# This block ensures the main function runs only when the script is executed directly
# (not when imported as a module).
if __name__ == "__main__":
    # --- Configuration ---
    # Define the base path where the 'daily', 'weekly', 'monthly' directories
    # will be created.
    # Default: Creates directories in the same location as the script.
    default_base_path = "."
    
    # Example: Create in a specific subdirectory of the user's home directory
    # default_base_path = os.path.join(os.path.expanduser("~"), "MyOrganizedFolders")
    
    # Example: Create in a specific subdirectory of the current working directory
    # default_base_path = os.path.join(os.getcwd(), "ProjectArchives")

    create_project_directories(base_path=default_base_path)
    
    print(f"\nScript finished. All operations targeted the base path: {os.path.abspath(default_base_path)}")
    print("To run this script directly (on Linux/macOS):")
    print("1. Save it (e.g., as create_dirs.py).")
    print("2. Open your terminal and navigate to the script's directory.")
    print("3. Make it executable: chmod +x create_dirs.py")
    print("4. Run it: ./create_dirs.py")

