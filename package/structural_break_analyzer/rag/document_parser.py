import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd


class DocumentParser:
    """Parse various document formats."""

    @staticmethod
    def parse_file(file_path: str) -> Dict[str, Any]:
        """
        Parse a file and extract content and metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with 'content' and 'metadata'
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine parser based on extension
        extension = file_path.suffix.lower()

        if extension == '.txt':
            return DocumentParser._parse_text(file_path)
        elif extension == '.csv':
            return DocumentParser._parse_csv(file_path)
        elif extension == '.pdf':
            return DocumentParser._parse_pdf(file_path)
        elif extension in ['.md', '.markdown']:
            return DocumentParser._parse_markdown(file_path)
        else:
            # Try to read as text
            return DocumentParser._parse_text(file_path)

    @staticmethod
    def _parse_text(file_path: Path) -> Dict[str, Any]:
        """Parse text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Try to extract date from filename or content
        date = DocumentParser._extract_date(file_path.name) or \
               DocumentParser._extract_date_from_content(content[:500])

        return {
            'content': content,
            'metadata': {
                'source': str(file_path),
                'title': file_path.stem,
                'date': date,
                'type': 'text'
            }
        }

    @staticmethod
    def _parse_csv(file_path: Path) -> Dict[str, Any]:
        """Parse CSV file."""
        try:
            df = pd.read_csv(file_path)

            # Convert to string representation
            content = f"CSV Data from {file_path.name}:\n"
            content += f"Columns: {', '.join(df.columns)}\n"
            content += f"Shape: {df.shape}\n\n"
            content += "First few rows:\n"
            content += df.head(10).to_string()

            # Try to find date column
            date = None
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        dates = pd.to_datetime(df[col])
                        date = dates.median()  # Use median date
                        break
                    except:
                        continue

            if date is None:
                date = DocumentParser._extract_date(file_path.name)

            return {
                'content': content,
                'metadata': {
                    'source': str(file_path),
                    'title': file_path.stem,
                    'date': date,
                    'type': 'csv',
                    'shape': df.shape
                }
            }
        except Exception as e:
            return {
                'content': f"Error parsing CSV: {str(e)}",
                'metadata': {
                    'source': str(file_path),
                    'title': file_path.stem,
                    'type': 'csv',
                    'error': str(e)
                }
            }

    @staticmethod
    def _parse_pdf(file_path: Path) -> Dict[str, Any]:
        """Parse PDF file (prefers pypdf; falls back to PyPDF2)."""
        try:
            try:
                from pypdf import PdfReader as _PdfReader  # type: ignore
                _lib = "pypdf"
            except Exception:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    from PyPDF2 import PdfReader as _PdfReader  # type: ignore
                _lib = "PyPDF2"

            content = ""
            with open(file_path, 'rb') as f:
                reader = _PdfReader(f)
                for page in getattr(reader, 'pages', []):
                    text = getattr(page, 'extract_text', lambda: "")() or ""
                    content += text + "\n"

            date = DocumentParser._extract_date(file_path.name) or \
                   DocumentParser._extract_date_from_content(content[:1000])

            return {
                'content': content,
                'metadata': {
                    'source': str(file_path),
                    'title': file_path.stem,
                    'date': date,
                    'type': 'pdf',
                    'pages': len(getattr(reader, 'pages', [])),
                    'parser_lib': _lib,
                }
            }
        except ImportError:
            return {
                'content': "PDF parsing requires pypdf (PyPDF2). Install with: pip install pypdf",
                'metadata': {
                    'source': str(file_path),
                    'title': file_path.stem,
                    'type': 'pdf',
                    'error': 'pypdf not installed'
                }
            }
        except Exception as e:
            return {
                'content': f"Error parsing PDF: {str(e)}",
                'metadata': {
                    'source': str(file_path),
                    'title': file_path.stem,
                    'type': 'pdf',
                    'error': str(e)
                }
            }

    @staticmethod
    def _parse_markdown(file_path: Path) -> Dict[str, Any]:
        """Parse markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title from first # heading if exists
        title = file_path.stem
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)

        date = DocumentParser._extract_date(file_path.name) or \
               DocumentParser._extract_date_from_content(content[:500])

        return {
            'content': content,
            'metadata': {
                'source': str(file_path),
                'title': title,
                'date': date,
                'type': 'markdown'
            }
        }

    @staticmethod
    def _extract_date(text: str) -> Optional[datetime]:
        """Extract date from text using regex patterns."""
        # Common date patterns
        patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # MM-DD-YYYY or MM/DD/YYYY
            r'(\d{4}\d{2}\d{2})',  # YYYYMMDD
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return pd.to_datetime(match.group(1))
                except:
                    continue
        return None

    @staticmethod
    def _extract_date_from_content(content: str) -> Optional[datetime]:
        """Try to extract date from document content."""
        return DocumentParser._extract_date(content)
