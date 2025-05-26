# Journal Directory Structure

This directory contains the journal entries organized by frequency:

- `daily/` - Daily journal entries
- `weekly/` - Weekly journal entries
- `monthly/` - Monthly journal entries

## File Naming Convention

Files are named using the following format:
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