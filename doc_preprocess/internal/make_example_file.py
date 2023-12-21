import os
import json
import argparse
import csv
from datetime import datetime

EXAMPLE_SEP = '\n\n'

def parse_example_to_json(file_content):
    arr = file_content.split(EXAMPLE_SEP)
    json_arr = []

    for item in arr:
        elements = item.strip().split("\n")
        print("elements:")
        print(elements)
        obj = { element.split(":")[0] : element.split(":")[1] for element in elements }
        json_arr.append(obj)

    qa_content = {
        "example_list" : json_arr
    }
    
    json_content = json.dumps(qa_content, ensure_ascii=False)
    return json_content



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faq-file", type=str,)
    parser.add_argument("--output-file", type=str, default='faq.example')
    args = parser.parse_args()
    output_file = args.output_file
    faq_file = args.faq_file
    examples = []
    with open(faq_file,'r') as f:
        csv_data = f.readlines()
        reader = csv.reader(csv_data)
        header = next(reader)# Skip the header row
        for item in reader:
            examples.append({"query":item[0],
                             "detection":{"func":"QA"}})
    
    example_json = {
                    "api_schema":{
                        "name": "QA",
                        "description": "answer question according to searched relevant content"
                    },
                    "examples":examples
                    }
    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    example_json_str = json.dumps(example_json,ensure_ascii=False)
    with open(f"{timestamp_str}_{output_file}","w") as f:
        f.write(example_json_str)

if __name__ == '__main__':
    main()