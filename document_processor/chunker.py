import pdfplumber
import os

def format_table(table):
    """Formats table rows into pipe-separated key-value pairs."""
    if not table or len(table) < 2:
        return ""
    
    # Filter out empty rows
    cleaned_table = []
    for row in table:
        if row and any(cell is not None and str(cell).strip() for cell in row):
            cleaned_table.append(row)
            
    if len(cleaned_table) < 2:
        return ""

    # Clean headers
    headers = []
    for i, h in enumerate(cleaned_table[0]):
        clean_header = str(h).strip().replace("\n", " ") if h is not None and str(h).strip() else f"Col_{i+1}"
        headers.append(clean_header)
    
    formatted_rows = []
    for row in cleaned_table[1:]:
        row_dict = {}
        for i, cell in enumerate(row):
            header = headers[i] if i < len(headers) else f"Col_{i+1}"
            val = str(cell).strip().replace("\n", " ") if cell is not None else ""
            if val:  # Only include non-empty cells
                row_dict[header] = val
        
        if row_dict:
            # Format row as a key-value string separated by pipe
            row_str = " | ".join(f"{k}: {v}" for k, v in row_dict.items())
            formatted_rows.append(row_str)
            
    return "\n".join(formatted_rows)


def chunk_text(text, max_length=500, overlap=50):
    """Splits text into chunks under max_length characters, with overlap."""
    if not text:
         return []

    # Split to paragraphs
    paragraphs = text.replace('\r\n', '\n').split('\n\n')
    chunks = []
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        if len(p) <= max_length:
            chunks.append(p)
        else:
            # Split long paragraphs by words
            words = p.split()
            current_chunk = []
            current_len = 0
            
            for word in words:
                # Calculate new chunk length
                word_len = len(word) + (1 if current_len > 0 else 0)
                if current_len + word_len <= max_length:
                    current_chunk.append(word)
                    current_len += word_len
                else:
                    if current_chunk:
                        chunk_str = " ".join(current_chunk)
                        chunks.append(chunk_str)
                        if overlap > 0 and len(chunk_str) >= overlap:
                            overlap_str = chunk_str[-overlap:].strip()
                            current_chunk = [overlap_str, word] if overlap_str else [word]
                            current_len = len(" ".join(current_chunk))
                        else:
                            current_chunk = [word]
                            current_len = len(word)
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
    return chunks

def process_pdf(pdf_path):
    """Extracts tables and text uniformly from a PDF."""
    results = []
    base_name = os.path.basename(pdf_path)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        formatted_table = format_table(table)
                        if formatted_table:
                            results.append({
                                "type": "table",
                                "content": formatted_table,
                                "page": page_num,
                                "source": base_name
                            })
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        text_chunks = chunk_text(text, max_length=500, overlap=50)
                        for chunk in text_chunks:
                            results.append({
                                "type": "text",
                                "content": chunk,
                                "page": page_num,
                                "source": base_name
                            })

                except Exception as e:
                    # Skip broken pages
                    print(f"Warning: Failed to process page {page_num} of {pdf_path}: {e}")
                    continue
                    
    except Exception as e:
         print(f"Error: Failed to open or read PDF {pdf_path}: {e}")
         
    return results

if __name__ == "__main__":
    # Testing stub
    pass
