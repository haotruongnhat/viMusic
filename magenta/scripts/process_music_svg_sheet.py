import cairosvg, os
from PyPDF2 import PdfFileMerger

def merge_all_pdf_files(root):
    '''
    From the pdf list, merge all into one pdf file,
    Then export to parent folder
    arg:
        1-dimentional vector with .pdf file name
        list = [song#1_directory/score_0.pdf, song#1_directory/score_1.pdf, ...]
    '''
    scores_list = find_all_file_available(root, ".pdf")

    for pdf_paths in scores_list:
        merger = PdfFileMerger()

        for pdf in pdf_paths:
            merger.append(pdf)

        parent = os.path.dirname(pdf_paths[0])
        print('Merge files in directory: ' + parent)

        merger.write(os.path.join(parent, "score.pdf"))
        merger.close()

def find_all_file_available(root_directory, extension):
    '''
    Find all available scores from root folder
    arg:
        root: Parent folder that store individual score folder
        root--
        --\song#1
        --\song#2
        --\...
    return:
        2-dimentional vector with each score folder and .svg file name
        list = [[song#1_directory/score_0.svg, song#1_directory/score_1.svg, ...],
                [song#2_directory/score_0.svg, song#2_directory/score_1.svg, ...],
                ...]
    '''
    scores_list = []
    #List all folder
    for folder in os.listdir(root_directory):
        if os.path.isdir(os.path.join(root_directory, folder)):
            files_path = []
            for file in os.listdir(os.path.join(root_directory, folder)):
                path = os.path.join(root_directory, folder, file)
                if path.endswith(extension):
                    files_path.append("{0}".format(path))

            if files_path !=[]:
                scores_list.append(files_path)
                
    score_count = sum([len(score) for score in scores_list])
    print('Number of svg file: ' + str(score_count))

    return scores_list
def convert_all_avaialble_svg_to_pdf(root):
    '''
    From the svg list, convert all into pdf file
    arg:
        2-dimentional vector with each score folder and .svg file name
        list = [[song#1_directory/score_0.svg, song#1_directory/score_1.svg, ...],
                [song#2_directory/score_0.svg, song#2_directory/score_1.svg, ...],
                ...]
        
    return:
        2-dimentional vector with each score folder and .pdf file name
        list = [[song#1_directory/score_0.pdf, song#1_directory/score_1.pdf, ...],
                [song#2_directory/score_0.pdf, song#2_directory/score_1.pdf, ...],
                ...]
    '''
    score_pdf_paths = []
    scores_list = find_all_file_available(root, ".svg")
    score_count = sum([len(score) for score in scores_list])
    score_current_count = 0
    for score_folder in scores_list:
        paths = []
        for path in score_folder:
            parent = "{0}".format(os.path.dirname(path))
            filename_only = "{0}".format(os.path.basename(path).split('.')[0]) + ".pdf"
            save_path = os.path.join(parent, filename_only)
            print(save_path)
            try: 
                cairosvg.svg2pdf(
                    url=path, write_to=save_path)
                paths.append(save_path)
            except:
                print('Error converting')
                break

        score_current_count = score_current_count + len(paths)
        print('Convert svg to pdf. Progress: ' + str(round(score_current_count/score_count, 2))   +'%')
        score_pdf_paths.append(paths)

    return score_pdf_paths


if __name__== "__main__":
    root = "downloads\scores"
    convert_all_avaialble_svg_to_pdf(root)
    merge_all_pdf_files(root)