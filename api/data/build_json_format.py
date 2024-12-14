import json
import re

# Define regular expressions for each field
title_pattern = re.compile(r'^TI  - (.+)')
doi_pattern = re.compile(r'AID - (.+) \[doi\]')
author_pattern = re.compile(r'FAU - (.+)')
abstract_pattern = re.compile(r'^AB  - (.+)')
mesh_pattern = re.compile(r'^MH  - (.+)')
ot_pattern = re.compile(r'^OT  - (.+)')
date_pattern = re.compile(r'^DP  - (.+)')

# Variables to hold extracted data
abstracts = []

# Open the file and process it line-by-line
with open('api\data\PubMedData\pubmed-arthroplas-set.txt', 'r') as file:
    current_title = None
    current_doi = None
    current_authors = []
    current_abstract = []
    current_mesh_headings = []
    current_terms = []
    current_date = None

    is_collecting_title = False
    is_collecting_abstract = False
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
    
        # Handle Titles
        if title_pattern.match(line):
            if current_title:
                # Save the completed record before starting a new one
                abstracts.append({
                    "title": current_title,
                    "abstract": ' '.join(current_abstract),
                    "authors": current_authors,
                    "doi": current_doi if current_doi else '',
                    "publication_date": current_date if current_date else '',
                    "mesh_headings": current_mesh_headings,
                    "other_terms": current_terms
                })
                # Reset temporary storage
                current_doi = None
                current_authors = []
                current_abstract = []
                current_mesh_headings = []
                current_terms = []
                current_date = None
    
            # Start a new title
            current_title = title_pattern.match(line).group(1)
            is_collecting_title = True
            continue
    
        # If title spans multiple lines
        if is_collecting_title and not line.startswith(('PG  -')):
            current_title += ' ' + line.strip()
            continue
        else:
            is_collecting_title = False
    
        # Handle DOI
        if doi_pattern.match(line):
            current_doi = doi_pattern.match(line).group(1)
    
        # Handle Authors
        if author_pattern.match(line):
            current_authors.append(author_pattern.match(line).group(1))
    
        # Handle Abstracts
        if abstract_pattern.match(line):
            current_abstract.append(abstract_pattern.match(line).group(1))
            is_collecting_abstract = True
            continue
    
        # Handle MeSh Headings
        if mesh_pattern.match(line):
            current_mesh_headings.append(mesh_pattern.match(line).group(1))
    
        # Handle Other terms
        if ot_pattern.match(line):
            current_terms.append(ot_pattern.match(line).group(1))
    
        if date_pattern.match(line):
            current_date = date_pattern.match(line).group(1)

        # If abstract spans multiple lines
        if is_collecting_abstract and not line.startswith(('CI  -')):
            current_abstract.append(line)
        else:
            is_collecting_abstract = False
    
    # Append the last record if it exists
    if current_title:
       abstracts.append({
                    "title": current_title,
                    "abstract": ' '.join(current_abstract),
                    "authors": current_authors,
                    "doi": current_doi if current_doi else '',
                    "publication_date": current_date if current_date else '',
                    "mesh_headings": current_mesh_headings,
                    "other_terms": current_terms
                })

with open("pubmed-tja.json", "w") as f:
    json.dump(abstracts, f)