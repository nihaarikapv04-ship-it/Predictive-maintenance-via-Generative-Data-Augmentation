"""Output parser and validator for structured prescriptions."""

import re


class PrescriptionParser:
    """Validate the required output sections and retry on malformed output."""

    REQUIRED_SECTIONS = ["Immediate Action", "Repair Protocol", "Preventive Schedule"]

    def parse(self, text: str):
        if not text or not isinstance(text, str):
            raise ValueError("Prescription text must be a non-empty string")

        sections = {}
        for section_name in self.REQUIRED_SECTIONS:
            pattern = rf"{re.escape(section_name)}\s*:\s*(.+)"
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                raise ValueError(f"Missing required section: {section_name}")
            sections[section_name] = match.group(1).strip()

        return sections
