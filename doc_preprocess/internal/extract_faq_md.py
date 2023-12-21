import re
import argparse

def parse_faq(file_path):
  """
  解析 .md 文本文件，提取其中的 FAQ 问答。

  Args:
    file_path: .md 文本文件路径。

  Returns:
    一个列表，其中包含 FAQ 问答对。
  """

  with open(file_path, "r") as f:
    text = f.read()

  # 提取 Q 和 A 的正则表达式。
  pattern = re.compile(r"^(Q:.*?)\n(.*?)\n(Q:.*?)\n(.*?)\n(Q:.*?)\n(.*?)$")

  # 使用正则表达式匹配 Q 和 A。
  matches = pattern.finditer(text)

  # 生成 FAQ 问答对列表。
  faqs = []
  for match in matches:
    faqs.append((match.group(1), match.group(3)))

  return faqs

def extract_q_and_a(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    content = re.sub(r'\*\*Q\:\*(.+)\n(.+)\n', r'Q:\1 A:\2\n', content)
    content = re.sub(r'\*\*Q\:\*(.+)$', r'Q:\1 A:\n', content)
    print(content)
    q_and_a_list = []
    lines = content.strip().split('\n')
    question = ''
    answer = ''

    for line in lines:
        if line.startswith('Q:'):
            question = line[2:].strip()
            answer = ''
        elif line.startswith('A:'):
            answer = line[2:].strip()
        elif question and answer:
            q_and_a_list.append((question, answer))
            question = ''
            answer = ''

    return q_and_a_list

def extract_qa_pairs_from_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    qa_pairs = re.findall(r'\*\*Q：(.*?)\*\*\s?\n(.*?)\n\n', content, re.DOTALL)
    qa_pairs = [(question.strip(), answer.strip()) for question, answer in qa_pairs]
    
    return qa_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str,)
    args = parser.parse_args()
    file_path = args.file
    faqs = extract_qa_pairs_from_md_file(file_path)
    for question, answer in faqs:
        print("Question:", question)
        print("Answer:", answer)
        print()
    print(len(faqs))
