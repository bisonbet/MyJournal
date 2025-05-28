# Journal Directory Structure

This directory contains the journal entries organized by frequency with a hierarchical structure:

- `daily/YYYY/MonthName/MMDDYYYY/` - Daily journal entries
- `weekly/YYYY/MonthName/WeekOfMMDDYYYY/` - Weekly journal entries
- `monthly/YYYY/MonthName/` - Monthly journal entries

## Directory Structure Details

The structure is organized as follows:
- Daily entries are stored in: `daily/YYYY/MonthName/MMDDYYYY/`
  - Example: `daily/2024/May/05282024/`
- Weekly entries are stored in: `weekly/YYYY/MonthName/WeekOfMMDDYYYY/`
  - Example: `weekly/2024/May/WeekOf05262024/`
- Monthly entries are stored in: `monthly/YYYY/MonthName/`
  - Example: `monthly/2024/May/`

## File Naming Convention

Files within each directory should be named using the following format:
- Daily: `YYYY-MM-DD-title.md`
- Weekly: `YYYY-MM-DD-title.md` (using the start date of the week)
- Monthly: `YYYY-MM-title.md`

## Example Files

Each directory contains an example file with "EXAMPLE" in the name to demonstrate the format and structure. These example files are tracked in git, while all other journal entries are ignored.

## Git Configuration

The `.gitignore` file is configured to:
- Ignore all `.md` files in the journal subdirectories
- Track files with "EXAMPLE" in the name
- Track this README.md file 