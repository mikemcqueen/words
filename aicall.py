#!/usr/bin/env python3
import requests
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Call LLM API and extract reasoning')
    parser.add_argument('url', nargs='?', 
                        default='http://localhost:8000/v1/chat/completions',
                        help='API endpoint URL (default: http://localhost:8000/v1/chat/completions)')
    parser.add_argument('--data', '-d', help='JSON payload', required=True)
    parser.add_argument('--header', '-H', action='append', help='HTTP headers')
    
    args = parser.parse_args()
    
    # Parse headers with default Content-Type
    headers = {"Content-Type": "application/json"}
    if args.header:
        for h in args.header:
            key, value = h.split(':', 1)
            headers[key.strip()] = value.strip()
    
    # Parse JSON data
    payload = json.loads(args.data)
    
    # Make request
    response = requests.post(args.url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Extract and remove reasoning_content
    reasoning = None
    if 'choices' in data and len(data['choices']) > 0:
        msg = data['choices'][0].get('message', {})
        reasoning = msg.pop('reasoning_content', None)
    
    # Display
    if reasoning:
        print("\n" + "="*80)
        print("REASONING:")
        print("="*80)
        print(reasoning)
        print("="*80 + "\n")
    
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
