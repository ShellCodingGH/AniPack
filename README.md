# AniPack
# Instructions

<b>Step 0</b>

Install git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Install ```git lfs```: for your operating system, type in the corresponding commands in terminal to install ```git lfs```
  Ubuntu: 
  
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

    sudo apt-get install git-lfs
  Windows and Mac:
    Please refer to the installation guide: https://git-lfs.com/

Also, make sure you have Python 3.10.14 installed: https://www.python.org/downloads/release/python-31014/

<b>Step 1</b>

Open up a terminal, type in the following commands:

```git lfs install```

```git clone https://github.com/ShellCodingGH/AniPack.git```

```cd AniPack```

```git clone https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer```

<b>Step 2</b>

Move all contents in the ```VITS-Umamusume-voice-synthesizer``` folder into your working directory(in this example it's the ```AniPack``` directory).

<b>Step 3</b>

Copy and paste the commands in ```requirements_anipack.txt``` into the terminal.

<b>Step 4</b>

Edit ```character_with_I2V.py``` by modifying the variable ```HF_TOKEN``` in the 8th line of the python file. Go to https://huggingface.co/settings/tokens to create a token and assign it to ```HF_TOKEN```.

<b>Step 5</b>

Run the command in terminal: ```python character_with_I2V.py```

<b>Step 6</b>

Copy and paste the public url in the terminal, open it in a browser: <img width="936" alt="Screenshot 2024-06-13 at 3 53 28 PM" src="https://github.com/ShellCodingGH/AniPack/assets/49096303/74696d0f-0d7d-4302-8ebc-5e0789c26ddf">
