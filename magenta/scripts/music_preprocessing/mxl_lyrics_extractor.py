import music21, os, asyncio
from music21 import environment as mEnv
from music21 import converter as mConverter

base_mxl_directory = "./mxl_data"

def extract(file_path):
    pass

def main():
    #1. Set up environment path
    mEnv.set("musicxmlPath","/usr/bin/musescore")
    mEnv.set("musescoreDirectPNGPath", "/usr/bin/musescore")

    #lyrics extraction
    for root, _, files in os.walk(base_mxl_directory):
        for file in files:
            if file.endswith(".mxl"):
                file_path = os.path.join(root,file)
                loop = asyncio.get_event_loop()
                loop.run_until_complete(extract(file_path))  

if __name__ == "__main__":
    main()