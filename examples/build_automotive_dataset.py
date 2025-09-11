#!/usr/bin/env python3
"""
Automotive Installation Dataset Builder

This script extracts text from automotive PDF installation guides and converts
them into training examples for the modular pipeline.

Usage: python build_automotive_dataset.py --pdf-dir /path/to/pdfs
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
    except ImportError:
        print("PyPDF2 not installed. Install with: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def clean_text(text):
    """Clean and normalize extracted PDF text"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove page numbers, headers, footers
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip likely page numbers
        if re.match(r'^\d+$', line):
            continue
        # Skip very short lines (likely artifacts)
        if len(line) < 3:
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_sections(text):
    """Extract key sections from installation guide"""
    sections = {}
    
    # Common section patterns in installation guides
    patterns = {
        'tools': r'(?i)(tools?\s+required|required\s+tools?|tools?\s+needed)',
        'parts': r'(?i)(parts?\s+included|included\s+parts?|kit\s+contents?)',
        'safety': r'(?i)(safety|warning|caution|important)',
        'steps': r'(?i)(installation\s+steps?|install|procedure|instructions?)',
        'torque': r'(?i)(torque\s+spec|torque\s+value|ft\.?\s*lbs?|nm)',
    }
    
    for section_name, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            # Extract text around the match
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 500)
            section_text = text[start:end].strip()
            
            if section_name not in sections:
                sections[section_name] = []
            sections[section_name].append(section_text)
    
    return sections

def create_training_examples(pdf_path, text, part_name):
    """Create training examples from PDF content - direct knowledge transfer"""
    examples = []
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Base system prompt
    system_prompt = "You are an expert automotive installation assistant. Provide detailed, safety-focused installation guidance and always end with 'Drive safe!'"
    
    # Split text into chunks for training examples
    text_chunks = cleaned_text.split('\n\n')
    useful_chunks = []
    
    # Filter for useful installation content
    for chunk in text_chunks:
        chunk = chunk.strip()
        if len(chunk) < 50:  # Skip very short chunks
            continue
        
        # Look for installation-related content
        installation_indicators = [
            'install', 'remove', 'torque', 'bolt', 'screw', 'connect', 'disconnect',
            'assembly', 'procedure', 'step', 'warning', 'caution', 'tools', 'required',
            'ft-lbs', 'nm', 'tighten', 'loosen', 'position', 'align', 'mount'
        ]
        
        if any(indicator in chunk.lower() for indicator in installation_indicators):
            useful_chunks.append(chunk)
    
    # Split the full text into more sections for training
    # Break the PDF into logical sections by splitting on common section headers
    section_splits = [
        "TOOLS REQUIRED", "SAFETY REQUIREMENTS", "PARTS INCLUDED", 
        "PROCEDURE", "WARNING", "CAUTION", "INSTALLATION", "STEPS"
    ]
    
    pdf_sections = []
    current_section = ""
    
    for line in cleaned_text.split('\n'):
        if any(split in line.upper() for split in section_splits):
            if current_section.strip():
                pdf_sections.append(current_section.strip())
            current_section = line + '\n'
        else:
            current_section += line + '\n'
    
    if current_section.strip():
        pdf_sections.append(current_section.strip())
    
    # Create training examples from each section
    for i, section in enumerate(pdf_sections[:15]):  # Take up to 15 sections
        if len(section) < 100:  # Skip very short sections
            continue
        
        # Create specific questions based on section content
        question_options = []
        if 'tool' in section.lower():
            question_options.extend([
                f"What tools are needed for {part_name} installation?",
                f"What equipment do I need to install {part_name}?",
                f"What's required for {part_name} installation?"
            ])
        if 'torque' in section.lower() or 'ft-lb' in section.lower():
            question_options.extend([
                f"What are the torque specifications for {part_name}?",
                f"How tight should I torque the {part_name} bolts?",
                f"What torque specs should I use for {part_name}?"
            ])
        if 'warning' in section.lower() or 'caution' in section.lower() or 'safety' in section.lower():
            question_options.extend([
                f"What safety precautions should I take when installing {part_name}?",
                f"What warnings should I know about {part_name} installation?",
                f"Are there any safety concerns with {part_name}?"
            ])
        if 'step' in section.lower() or 'procedure' in section.lower():
            question_options.extend([
                f"What are the installation steps for {part_name}?",
                f"How do I install {part_name} step by step?",
                f"Walk me through {part_name} installation"
            ])
        if 'parts' in section.lower() or 'included' in section.lower():
            question_options.extend([
                f"What parts are included with {part_name}?",
                f"What comes in the {part_name} kit?",
                f"What components are part of {part_name}?"
            ])
        
        # Use the first available question or a generic one
        if question_options:
            question = question_options[i % len(question_options)]
        else:
            generic_questions = [
                f"How do I properly install {part_name}?",
                f"Can you help me install {part_name}?",
                f"What's involved in {part_name} installation?",
                f"I need help installing {part_name}",
                f"What should I know about {part_name} installation?"
            ]
            question = generic_questions[i % len(generic_questions)]
        
        # Create response with section content
        response = f"For {part_name}:\n\n{section}\n\nDrive safe!"
        
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
        }
        examples.append(example)
    
    return examples

def create_sample_dataset():
    """Create a sample automotive dataset without PDFs for demonstration"""
    examples = []
    
    system_prompt = "You are an expert automotive installation assistant specializing in performance modifications. Provide detailed, safety-focused installation guidance and always end with 'Drive safe!'"
    
    # Sample automotive installation Q&A pairs
    sample_data = [
        {
            "question": "How do I install a cold air intake on my 350Z?",
            "answer": "Here's how to install a cold air intake on your 350Z:\n\n1. Disconnect the battery negative terminal\n2. Remove the stock airbox and intake tube\n3. Install the new heat shield in the fender well\n4. Mount the new air filter in the shield\n5. Connect the intake tube to the throttle body\n6. Secure all clamps to 25-30 inch-pounds\n7. Reconnect the MAF sensor and any vacuum lines\n8. Reconnect the battery and start the engine\n\nAllow the ECU to relearn idle for 10-15 minutes. Drive safe!"
        },
        {
            "question": "What tools do I need for a coilover installation?",
            "answer": "For coilover installation, you'll need:\n\n• Floor jack and jack stands\n• Socket set (10mm-19mm)\n• Torque wrench (80-150 ft-lbs range)\n• Spring compressor (for strut-type)\n• Penetrating oil (PB Blaster)\n• Safety glasses and gloves\n• Wheel chocks\n• Impact gun (optional but helpful)\n\nNever skip the safety equipment - coilover installation involves heavy components and high spring tension. Drive safe!"
        },
        {
            "question": "Are there safety concerns with turbo installations?",
            "answer": "Yes, turbo installations have several critical safety considerations:\n\n• Fire hazard from hot exhaust components\n• High-pressure oil and coolant lines\n• Fuel system modifications require extreme caution\n• Proper heat shielding is essential\n• ECU tuning is mandatory for safe operation\n• All connections must be torqued to specification\n• Use proper jack stands, never work under car with just a jack\n\nImproper installation can result in engine damage or fire. Consider professional installation for complex turbo kits. Drive safe!"
        },
        {
            "question": "What are the torque specs for wheel installation?",
            "answer": "Wheel lug nut torque specifications vary by vehicle:\n\n• Nissan 350Z/370Z: 80 ft-lbs\n• Subaru WRX/STI: 89 ft-lbs\n• Honda Civic/Integra: 80 ft-lbs\n• BMW E36/E46: 88 ft-lbs\n• Toyota Supra: 76 ft-lbs\n\nAlways use a calibrated torque wrench and tighten in a star pattern. Over-tightening can warp rotors or break studs. Under-tightening risks wheel separation. Drive safe!"
        },
        {
            "question": "How do I properly install lowering springs?",
            "answer": "Lowering spring installation procedure:\n\n1. Raise vehicle and remove wheels\n2. Support lower control arm with jack\n3. Remove brake line brackets as needed\n4. Compress spring with spring compressor\n5. Remove strut top nuts and lower bolts\n6. Extract old spring, install new spring\n7. Reassemble in reverse order\n8. Torque specifications:\n   - Top strut nuts: 25-30 ft-lbs\n   - Lower strut bolts: 65-80 ft-lbs\n9. Get professional alignment after installation\n\nNever work under a car supported only by a jack. Spring compressors can be dangerous if misused. Drive safe!"
        },
        {
            "question": "What should I know before installing an exhaust system?",
            "answer": "Key considerations for exhaust installation:\n\n• Check local noise ordinances and emissions laws\n• Use proper jack stands and safety equipment\n• Apply penetrating oil to old bolts 24 hours before\n• New gaskets are usually required at connections\n• Maintain proper ground clearance (4+ inches)\n• Allow proper heat expansion gaps\n• Use high-temp sealant on slip joints\n• Torque flange bolts to 25-35 ft-lbs\n\nTest fit everything before final tightening. Heat cycles will require retightening after 100-200 miles. Drive safe!"
        },
        {
            "question": "How do I install a short shifter?",
            "answer": "Short shifter installation steps:\n\n1. Remove center console and shift boot\n2. Disconnect shift cables at transmission\n3. Remove stock shifter assembly bolts\n4. Extract old shifter through cabin\n5. Install new shifter with provided bushings\n6. Apply thread locker to mounting bolts\n7. Torque shifter bolts to 25-30 ft-lbs\n8. Reconnect cables and adjust cable tension\n9. Test all gear positions before reassembly\n\nProper cable adjustment is critical for smooth shifting. Take your time with this step. Drive safe!"
        },
        {
            "question": "What are the steps for brake pad replacement?",
            "answer": "Brake pad replacement procedure:\n\n1. Loosen wheel bolts, raise vehicle, remove wheel\n2. Remove brake caliper bolts (usually 14-17mm)\n3. Hang caliper with wire to prevent hose damage\n4. Remove old pads and note wear pattern\n5. Compress caliper piston with C-clamp\n6. Install new pads with anti-squeal compound\n7. Reinstall caliper and torque bolts to spec\n8. Pump brake pedal to seat pads\n9. Check brake fluid level and top off\n\nBedding procedure: 10 moderate stops from 40mph, then cool down period. Brakes are safety-critical - double-check all work. Drive safe!"
        },
        {
            "question": "How do I install a blow-off valve?",
            "answer": "Blow-off valve installation:\n\n1. Locate vacuum source on intake manifold\n2. Install vacuum line tee fitting\n3. Mount BOV bracket to intercooler piping\n4. Connect vacuum line to BOV diaphragm\n5. Adjust spring preload per manufacturer specs\n6. Test operation with engine running\n7. Fine-tune adjustment for proper operation\n\nBOV should only open under deceleration with throttle closed. Incorrect adjustment can cause rich running or compressor surge. Drive safe!"
        },
        {
            "question": "What tools are needed for suspension work?",
            "answer": "Essential tools for suspension installation:\n\n• Quality floor jack (3+ ton capacity)\n• Jack stands rated for vehicle weight\n• Spring compressor (for strut work)\n• Torque wrench (10-200 ft-lb range)\n• Impact gun with deep sockets\n• Penetrating oil and wire brushes\n• Safety glasses and work gloves\n• Pry bars and dead blow hammer\n• Thread chaser set for damaged threads\n\nNever compromise on safety equipment. Suspension components are heavy and under high stress. Professional tools make the job safer and easier. Drive safe!"
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
    """Main function to build automotive dataset"""
    parser = argparse.ArgumentParser(description='Build automotive installation dataset')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDF installation guides')
    parser.add_argument('--output', type=str, default='automotive_installation_sample.jsonl', 
                       help='Output dataset file')
    
    args = parser.parse_args()
    
    print("🔧 Building Automotive Installation Dataset...")
    print("=" * 50)
    
    examples = []
    
    if args.pdf_dir and os.path.exists(args.pdf_dir):
        # Process PDF files
        pdf_files = list(Path(args.pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {args.pdf_dir}")
            print("Creating sample dataset instead...")
            examples = create_sample_dataset()
        else:
            print(f"Found {len(pdf_files)} PDF files")
            
            for pdf_path in pdf_files[:10]:  # Limit to first 10 PDFs for demo
                print(f"Processing {pdf_path.name}...")
                
                text = extract_pdf_text(pdf_path)
                if text:
                    # Extract part name from filename
                    part_name = pdf_path.stem.replace('_', ' ').replace('-', ' ')
                    part_name = re.sub(r'install|guide|manual', '', part_name, flags=re.IGNORECASE).strip()
                    
                    pdf_examples = create_training_examples(pdf_path, text, part_name)
                    examples.extend(pdf_examples)
    else:
        print("No PDF directory specified or directory doesn't exist.")
        print("Creating sample automotive dataset...")
        examples = create_sample_dataset()
    
    # Only use PDF examples - no generic samples
    print(f"Created {len(examples)} examples from PDF content only")
    
    # Save dataset
    output_path = Path(__file__).parent / "data" / args.output
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\nSaving {len(examples)} examples to {output_path}")
    
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Dataset saved: {len(examples)} automotive installation examples")
    print(f"📁 Location: {output_path}")
    print("\nNext steps:")
    print(f"1. Run: python test_automotive_assistant.py")
    print(f"2. Train specialized automotive installation assistant")
    print(f"3. Test with questions like 'How do I install coilovers?'")

if __name__ == "__main__":
    main()