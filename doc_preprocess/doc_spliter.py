import os
import re
import argparse
from bs4 import BeautifulSoup
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter

'''
1. pip install pdfminer.six
'''

def split_pdf_to_snippet(pdf_path):
    loader = PDFMinerPDFasHTMLLoader(pdf_path)
    data = loader.load()[0]

    soup = BeautifulSoup(data.page_content,'html.parser')
    content = soup.find_all('div')

    cur_fs = None
    cur_text = ''
    snippets = []   # first collect all snippets that have the same font size
    for c in content:
        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px',st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text,cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text,cur_fs))

    cur_idx = -1
    semantic_snippets = []
    # Assumption: headings have higher font size than their respective content
    for s in snippets:
        # if current snippet's font size > previous section's heading => it is a new heading
        if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
            metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
            metadata.update(data.metadata)
            semantic_snippets.append(Document(page_content='',metadata=metadata))
            cur_idx += 1
            continue
        
        # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
        if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata['content_font']:
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata['content_font'] = max(s[1], semantic_snippets[cur_idx].metadata['content_font'])
            continue
        
        # if current snippet's font size > previous section's content but less tha previous section's heading than also make a new 
        # section (e.g. title of a pdf will have the highest font size but we don't want it to subsume all sections)
        metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
        metadata.update(data.metadata)
        semantic_snippets.append(Document(page_content='',metadata=metadata))
        cur_idx += 1

    return semantic_snippets

def fontsize_mapping(heading_fonts_arr):
    heading_fonts_set = list(set(heading_fonts_arr))
    heading_fonts_set.sort(reverse=True)
    idxs = range(len(heading_fonts_set))
    font_idx_mapping = dict(zip(heading_fonts_set,idxs))
    return font_idx_mapping

def split_pdf(pdf_path):
    semantic_snippets = split_pdf_to_snippet(pdf_path)
    heading_fonts_arr = [ item.metadata['heading_font'] for item in semantic_snippets ]
    heading_arr = [ item.metadata['heading'] for item in semantic_snippets ]

    fontsize_dict = fontsize_mapping(heading_fonts_arr)

    for idx, snippet in enumerate(semantic_snippets):
        font_size = heading_fonts_arr[idx]
        heading_stack = []
        heading_info = {"font_size":heading_fonts_arr[idx], "heading":heading_arr[idx], "fontsize_idx" : fontsize_dict[font_size]}
        heading_stack.append(heading_info)
        for id in range(0,idx)[::-1]:
            if font_size < heading_fonts_arr[id]:
                font_size = heading_fonts_arr[id]
                heading_info = {"font_size":font_size, "heading":heading_arr[id], "fontsize_idx" : fontsize_dict[font_size]}
                heading_stack.append(heading_info)
            
        snippet_info = {
            "heading" : heading_stack,
            "content" : snippet.page_content
        }
        yield snippet_info

def summerize(content, chunk_size = 128):
    if len(content) > chunk_size:
        # todo: call LLM to summarize
        pass
    
    return content

def convert_snippetJson2markdown(snippet_info, max_level=3):
    mk_head = ""
    p_head = ""
    for item in snippet_info["heading"][0:max_level][::-1]:
        mk_head += "#"
        head = "{} {}".format(mk_head, item["heading"].replace('\n',''))
        p_head += "{}\n".format(head)
    
    p_content = "{}\n{}".format(p_head, snippet_info['content'])

    return p_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./1.pdf', help='input file')
    parser.add_argument('--output_dir', type=str, default='./', help='output file')
    parser.add_argument('--sep', type=str, default='=====', help='separtor')
    parser.add_argument('--title_level', type=int, default=4, help='keep the tiltes of level')
    parser.add_argument('--chunk_size', type=int, default=128, help='chunk_size')
    args = parser.parse_args()
    pdf_path = args.input_file
    kg_dir = args.output_dir
    kg_name = os.path.basename(pdf_path).replace('.pdf','.txt')
    separtor = args.sep
    max_title_level = args.title_level
    chunk_size = args.chunk_size
    idx = 1

    f_name = "{}/{}".format(kg_dir, kg_name)
    out_f = open(f_name, 'w')
    for snippet_info in split_pdf(pdf_path):
        p_content = convert_snippetJson2markdown(snippet_info, max_title_level)
        out_f.write(summerize(p_content, chunk_size))
        out_f.write(separtor)
        out_f.write("\n")

    out_f.close()
    print("finish separation of {}".format(pdf_path))

