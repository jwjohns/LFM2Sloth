#!/usr/bin/env python3
"""
PDF Knowledge Dataset Builder

This script extracts text from PDF documents and converts
them into training examples for the modular pipeline.

Usage: python build_pdf_dataset.py --pdf-dir /path/to/pdfs
"""

import os
import sys
import argparse
import json
from pathlib import Path
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def extract_pdf_text(pdf_path):
    """Extract text from PDF using PyPDF2"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def clean_text(text):
    """Clean extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\-\.\,\!\?\:\;\(\)]', '', text)
    return text.strip()

def extract_sections(text):
    """Extract key sections from technical documentation"""
    sections = {}
    
    # Common section patterns in technical documentation
    section_patterns = {
        'overview': r'(?i)(overview|introduction|summary)',
        'requirements': r'(?i)(requirements?|prerequisites?|specifications?)',
        'steps': r'(?i)(steps?|procedure|instructions?|process)',
        'troubleshooting': r'(?i)(troubleshooting|problems?|issues?|errors?)',
        'notes': r'(?i)(notes?|warnings?|cautions?|important)',
    }
    
    lines = text.split('\n')
    current_section = 'general'
    sections[current_section] = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line matches a section header
        section_found = False
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line):
                current_section = section_name
                sections[current_section] = []
                section_found = True
                break
        
        if not section_found:
            sections[current_section].append(line)
    
    return sections

def create_training_examples(text):
    """Convert PDF text into training examples"""
    examples = []
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Base system prompt
    system_prompt = "You are a helpful technical documentation assistant. Provide clear, detailed guidance based on the documentation."
    
    # Split text into chunks for training examples
    text_chunks = cleaned_text.split('\n\n')
    useful_chunks = []
    
    # Filter for useful content
    content_indicators = [
        'configure', 'install', 'setup', 'procedure', 'step', 'process',
        'requirement', 'specification', 'troubleshoot', 'error', 'problem'
    ]
    
    for chunk in text_chunks:
        if len(chunk.strip()) > 50:  # Skip very short chunks
            # Look for technical content
            if any(indicator in chunk.lower() for indicator in content_indicators):
                useful_chunks.append(chunk.strip())
    
    # Create Q&A pairs from useful chunks
    question_templates = [
        "How do I {}?",
        "What is the procedure for {}?",
        "Can you explain {}?",
        "What are the steps to {}?",
        "How can I configure {}?",
        "What should I know about {}?"
    ]
    
    for i, chunk in enumerate(useful_chunks[:20]):  # Limit to prevent too many examples
        if len(chunk.split()) > 10:  # Ensure substantial content
            # Extract key topic from chunk
            words = chunk.split()[:10]
            topic = ' '.join(words)
            
            # Create question-answer pair
            question = f"Can you explain this process from the documentation?"
            answer = chunk
            
            example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            examples.append(example)
    
    return examples

def create_sample_dataset():
    """Create a sample technical documentation dataset for demonstration"""
    examples = []
    
    system_prompt = "You are a helpful technical documentation assistant. Provide clear, detailed guidance based on the documentation."
    
    # Sample technical Q&A pairs
    sample_data = [
        {
            "question": "How do I configure the network settings?",
            "answer": "To configure network settings:\n\n1. Access the system configuration panel\n2. Navigate to Network Settings\n3. Select your connection type (Ethernet/WiFi)\n4. Enter the required IP address and subnet mask\n5. Configure DNS servers if needed\n6. Save the configuration\n7. Restart the network service\n\nVerify connectivity after configuration is complete."
        },
        {
            "question": "What are the system requirements?",
            "answer": "Minimum system requirements:\n\n• Operating System: Linux/Windows/macOS\n• RAM: 4GB minimum, 8GB recommended\n• Storage: 2GB available space\n• Network: Broadband internet connection\n• Browser: Modern web browser with JavaScript enabled\n\nFor optimal performance, ensure your system meets the recommended specifications."
        },
        {
            "question": "How do I troubleshoot connection issues?",
            "answer": "Common troubleshooting steps:\n\n1. Check all cable connections\n2. Verify network settings are correct\n3. Restart the device\n4. Check firewall settings\n5. Test with a different network\n6. Update drivers if necessary\n7. Contact support if issue persists\n\nDocument any error messages for support purposes."
        },
        {
            "question": "What maintenance is required?",
            "answer": "Regular maintenance schedule:\n\n• Daily: Check system status and logs\n• Weekly: Clear temporary files and cache\n• Monthly: Update software and security patches\n• Quarterly: Full system backup\n• Annually: Hardware inspection and cleaning\n\nProactive maintenance prevents most common issues."
        },
        {
            "question": "How do I perform a backup?",
            "answer": "Backup procedure:\n\n1. Access the backup utility\n2. Select files/folders to backup\n3. Choose backup destination\n4. Set backup schedule if desired\n5. Start the backup process\n6. Verify backup completion\n7. Test restore functionality\n\nStore backups in multiple locations for redundancy."
        }
    ]
    
    # Convert to training format
    for item in sample_data:
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
        }
        examples.append(example)
    
    return examples

def main():
    """Main function to build PDF knowledge dataset"""
    parser = argparse.ArgumentParser(description='Build PDF knowledge dataset')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDF documents')
    parser.add_argument('--output', type=str, default='pdf_knowledge_dataset.jsonl', 
                       help='Output JSONL file')
    
    args = parser.parse_args()
    
    print("📄 Building PDF Knowledge Dataset...")
    print("=" * 50)
    
    examples = []
    
    if args.pdf_dir and os.path.exists(args.pdf_dir):
        print(f"Processing PDFs from: {args.pdf_dir}")
        
        pdf_files = list(Path(args.pdf_dir).glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found in directory")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            text = extract_pdf_text(pdf_file)
            if text:
                pdf_examples = create_training_examples(text)
                examples.extend(pdf_examples)
                print(f"  Extracted {len(pdf_examples)} examples")
        
    else:
        print("No PDF directory provided or directory doesn't exist")
        print("Creating sample technical documentation dataset...")
        examples = create_sample_dataset()
    
    if not examples:
        print("❌ No examples generated")
        return
    
    # Save examples
    examples_dir = Path(__file__).parent / "data"
    examples_dir.mkdir(exist_ok=True)
    output_path = examples_dir / args.output
    
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Dataset saved: {len(examples)} technical documentation examples")
    print(f"Output file: {output_path}")
    print()
    print("Next steps:")
    print(f"1. Run: python test_pdf_assistant.py")
    print(f"2. Train specialized PDF knowledge assistant")

if __name__ == "__main__":
    main()