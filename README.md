
<img src="vimusic-logo.png" height="100">

**viMusic** is project that generating music by using the power of A.I technology, currently develop by viMusic team from **VIRALINT**

**Originally based on Google Magenta**: https://github.com/tensorflow/magenta

## Getting Started

* [Installation](#installation)

## Installation

Install the Magenta package:

```bash
python setup.py --user install
```

**NOTE I**: In order to install the `rtmidi` package that we depend on, you may need to install headers for some sound libraries. On Linux, this command should install the necessary packages:

```bash
sudo apt-get install build-essential libasound2-dev libjack-dev
```
**NOTE II**: Download external files needed for running

```bash
chmod +x external_download.sh
./external_download.sh
```

The Magenta libraries are now available for use within Python programs and
Jupyter notebooks, and the Magenta scripts are installed in your path!

