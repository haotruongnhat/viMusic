import music21
from music21 import environment as mEnv
from music21 import converter as mConverter
import os, subprocess, asyncio

base_pdf_directory = "./pdf_data"
output_mxl_directory = "./mxl_data"
output_midi_directory = "./midi_data"

async def run(file_name):
    #define commands and file paths
    pdf_mxl_command = ' '.join(['./Audiveris/bin/Audiveris', 
    '-export', 
    '-output',output_mxl_directory,
    '-batch',
    os.path.join(base_pdf_directory,file_name)])
    raw_file_name = file_name[:-4]
    input_mxl = os.path.join(output_mxl_directory,
    raw_file_name + "/" + raw_file_name + ".mxl")
    output_midi = os.path.join(output_midi_directory,
    raw_file_name + ".mid")

    #start conversion process (pdf to mxl)
    proc = await asyncio.create_subprocess_shell(
        pdf_mxl_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout,stderr = await proc.communicate()
    print(f'[{pdf_mxl_command!r} exited with {proc.returncode}]')
    if stdout:
        #mxl to midi
        print("Finish decoding. Moving on to the conversion.")
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

def main():
    #1. Set up environment path
    mEnv.set("musicxmlPath","/usr/bin/musescore")
    mEnv.set("musescoreDirectPNGPath", "/usr/bin/musescore")

    #pdf to mxl
    #iterate pdf files
    for file_name in os.listdir(base_pdf_directory):
        if file_name.endswith(".pdf"):
            #begin async processes
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run(file_name))

if __name__ == "__main__":
    main()