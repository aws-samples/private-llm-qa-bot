import os
import glob
import openpyxl
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./excels', help='input path')
    parser.add_argument('--output_path', type=str, default='./output/excel_output', help='output path')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # 打开Excel文件
    xlsx_files = glob.glob(os.path.join(input_path, '*.xlsx'))

    # 遍历每个 .xlsx 文件
    for xlsx_file in xlsx_files:
        xlsx_file_name = os.path.basename(xlsx_file)
        print(f"processing {xlsx_file_name}")
        try:
            workbook = openpyxl.load_workbook(xlsx_file)

            for worksheet in workbook.worksheets:
                # 遍历每一行,生成Markdown表格并保存到文件
                for row in worksheet.iter_rows(min_row=2, values_only=True):
                    # 获取表头
                    headers = [ str(cell.value) for cell in worksheet[1]]

                    # 生成Markdown表格
                    table = '| ' + ' | '.join(headers) + ' |\n'
                    table += '| ' + ' | '.join(['---' for _ in headers]) + ' |\n'
                    table += '| ' + ' | '.join([str(cell) for cell in row]) + ' |'

                    # 保存到文件
                    filename = f"{output_path}/{xlsx_file_name}_{worksheet.title}_row_{row[0]}.md"
                    with open(filename, 'w', encoding='utf-8') as file:
                        file.write(table)
                        print(f"Saved row {row[0]} from worksheet '{worksheet.title}' to {filename}")
        except Exception as e:
            print(str(e))