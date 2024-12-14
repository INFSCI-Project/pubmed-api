import re
import spacy

class NamedEntityExtraction:
    def __init__(self):
        self.nlp = spacy.load("en_ner_bc5cdr_md")

        # Define patterns and replacements for arthroplasty mapping
        self.replacements = [
            (r'(?i)(total\s+shoulder\s+arthroplast(?:y|ies)|shoulder\s+arthroplast(?:y|ies))', 'TSA'),
            (r'(?i)(total\s+hip\s+arthroplast(?:y|ies)|hip\s+arthroplast(?:y|ies))', 'THA'),
            (r'(?i)(total\s+knee\s+arthroplast(?:y|ies)|knee\s+arthroplast(?:y|ies))', 'TKA'),
            (r'(?i)(total\s+joint\s+arthroplast(?:y|ies)|joint\s+arthroplast(?:y|ies))', 'TJA'),
            (r'(?i)(shoulder\s+replacement|total\s+shoulder\s+replacement)', 'TSA'),
            (r'(?i)(hip\s+replacement|total\s+hip\s+replacement)', 'THA'),
            (r'(?i)(knee\s+replacement|total\s+knee\s+replacement)', 'TKA'),
            (r'(?i)(joint\s+replacement|total\s+joint\s+replacement)', 'TJA'),
            (r'(?i)(arthoplast(?:y|ies))', 'arthroplasty'),  # Normalize misspellings
            (r'(?i)(arthroplast(?:y|ies))', 'arthroplasty'),
            (r'(?i)(t\.s\.a\.|t\.h\.a\.|t\.k\.a\.|t\.j\.a\.)', lambda m: m.group().replace('.', '')),  # Remove periods
            (r'(?i)(shoulder\s+procedure)', 'TSA'),
            (r'(?i)(hip\s+procedure)', 'THA'),
            (r'(?i)(knee\s+procedure)', 'TKA'),
            (r'(?i)(joint\s+procedure)', 'TJA')
        ]

    def normalize_entities(self, text):
        """
        Normalize entities in the text using predefined patterns.

        Parameters:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
        return text

    def extract_ner(self, text):
        """
        Extract named entities and normalize arthroplasty terms.

        Parameters:
            text (str): Input text.

        Returns:
            list: List of extracted and normalized entities.
        """
        entities = {
            "DISEASE": [],
            "CHEMICAL": []
        }

        # Normalize text before entity extraction
        normalized_text = self.normalize_entities(text)

        doc = self.nlp(normalized_text)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)

        entities_for_indexing = []
        has_arthroplasty_term = False

        for label, entity_list in entities.items():
            unique_entities = list(set(entity_list))
            for entity in unique_entities:
                # Check if any entity matches a known arthroplasty abbreviation
                if entity in ["TSA", "THA", "TKA", "TJA"]:
                    entities_for_indexing.append({"entity": entity, "label": "CATEGORY"})
                    has_arthroplasty_term = True

        # Assign to a general category if no arthroplasty terms are found
        if not has_arthroplasty_term:
            entities_for_indexing.append({"entity": "General", "label": "CATEGORY"})

        return entities_for_indexing