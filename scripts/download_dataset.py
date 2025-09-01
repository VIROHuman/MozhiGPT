#!/usr/bin/env python3
"""
Download and prepare Tamil datasets for training.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_thamizhmalai(output_dir: str = "./data/Thamizhmalai") -> None:
    """Download Thamizhmalai dataset."""
    logger.info("Downloading Thamizhmalai dataset...")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        dataset = load_dataset("tamiltheorist/Thamizhmalai")
        
        # Save to local files
        for split_name, split_data in dataset.items():
            output_file = Path(output_dir) / f"{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in tqdm(split_data, desc=f"Saving {split_name}"):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {split_name} split with {len(split_data)} samples to {output_file}")
        
        logger.info("✅ Thamizhmalai dataset downloaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error downloading Thamizhmalai: {e}")
        raise


def download_additional_datasets(output_dir: str = "./data") -> None:
    """Download additional Tamil datasets."""
    
    datasets_to_download = [
        {
            "name": "Tamil Wikipedia",
            "hf_name": "wikipedia",
            "config": "ta",
            "output_name": "tamil_wikipedia"
        },
        {
            "name": "Tamil News",
            "hf_name": "mlsum",
            "config": "ta",
            "output_name": "tamil_news"
        },
        # Add more datasets here
    ]
    
    for dataset_info in datasets_to_download:
        try:
            logger.info(f"Downloading {dataset_info['name']}...")
            
            # Create output directory
            dataset_dir = Path(output_dir) / dataset_info['output_name']
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            if dataset_info.get('config'):
                dataset = load_dataset(
                    dataset_info['hf_name'], 
                    dataset_info['config']
                )
            else:
                dataset = load_dataset(dataset_info['hf_name'])
            
            # Save dataset
            for split_name, split_data in dataset.items():
                output_file = dataset_dir / f"{split_name}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in tqdm(split_data, desc=f"Saving {split_name}"):
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                logger.info(f"Saved {dataset_info['name']} {split_name} to {output_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not download {dataset_info['name']}: {e}")
            continue


def prepare_instruction_data(data_dir: str = "./data") -> None:
    """Prepare instruction-following data from downloaded datasets."""
    logger.info("Preparing instruction-following data...")
    
    instruction_data = []
    
    # Process Thamizhmalai data
    thamizhmalai_dir = Path(data_dir) / "Thamizhmalai"
    if thamizhmalai_dir.exists():
        for file_path in thamizhmalai_dir.glob("*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        
                        # Convert to instruction format
                        if 'instruction' in item and 'output' in item:
                            instruction_data.append({
                                "instruction": item['instruction'],
                                "input": item.get('input', ''),
                                "output": item['output'],
                                "source": "Thamizhmalai"
                            })
                        elif 'question' in item and 'answer' in item:
                            instruction_data.append({
                                "instruction": item['question'],
                                "input": "",
                                "output": item['answer'],
                                "source": "Thamizhmalai"
                            })
                            
                    except json.JSONDecodeError:
                        continue
    
    # Add some basic Tamil conversation examples
    basic_conversations = [
        {
            "instruction": "வணக்கம்",
            "input": "",
            "output": "வணக்கம்! நான் ஒரு தமிழ் AI உதவியாளர். உங்களுக்கு எப்படி உதவ முடியும்?",
            "source": "manual"
        },
        {
            "instruction": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "input": "",
            "output": "நான் நன்றாக இருக்கிறேன். நன்றி கேட்டதற்கு! நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "source": "manual"
        },
        {
            "instruction": "தமிழ் மொழி பற்றி சொல்லுங்கள்",
            "input": "",
            "output": "தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்று. இது சங்க இலக்கியம் முதல் நவீன இலக்கியம் வரை வளமான பாரம்பரியம் கொண்டது. தமிழ் மொழியின் இலக்கணம் மிகவும் முறையானது மற்றும் அழகானது.",
            "source": "manual"
        },
        {
            "instruction": "உங்களால் என்ன செய்ய முடியும்?",
            "input": "",
            "output": "நான் தமிழில் உரையாடல் நடத்த முடியும், கேள்விகளுக்கு பதில் அளிக்க முடியும், தமிழ் மொழி மற்றும் கலாச்சாரம் பற்றி விளக்க முடியும், மற்றும் பல்வேறு விषயங்களில் உதவி செய்ய முடியும்.",
            "source": "manual"
        }
    ]
    
    instruction_data.extend(basic_conversations)
    
    # Save prepared data
    output_file = Path(data_dir) / "instruction_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in instruction_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Prepared {len(instruction_data)} instruction examples and saved to {output_file}")


def main():
    """Main function to download and prepare datasets."""
    logger.info("🚀 Starting dataset download and preparation...")
    
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Download datasets
    download_thamizhmalai()
    download_additional_datasets()
    
    # Prepare instruction data
    prepare_instruction_data()
    
    logger.info("✅ Dataset preparation completed!")


if __name__ == "__main__":
    main()
