# Installation and Usage
1. Clone this repository
2. Copy and paste all the commands in ```requirements_anipack.txt``` into your terminal
  a. for AMD Mi210 GPU, add these 2 lines at the end of the file ```requirements_anipack.txt```:

    ```pip uninstall -y torch torchvision torchaudio```
   
    ```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1```
4. cd into the directory ```AniPack```
5. run ```character_with_I2V.py``` in your terminal.

# Credits
Thanks to ```wen3052963913173401```, ```Nerfgun3```, ```CyberAIchemist```, ```Kokoboy```, ```Euge_```, ```Havoc```, ```lenML```, ```yunleme```, ```luciI```, ```mattmdjaga```, ```okotaku```, ```skytnt```, and ```platcha```.
