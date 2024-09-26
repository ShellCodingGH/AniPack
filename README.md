# Installation and Usage
1. Clone this repository
2. Copy and paste all the commands in ```requirements_anipack.txt``` into your terminal

   Note: for AMD Mi210 GPU, add these 2 lines at the end of the file ```requirements_anipack.txt```:

    ```pip uninstall -y torch torchvision torchaudio```
   
    ```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1```
3. download these 2 folders, and put under the directory ```AniPack```:

   https://drive.google.com/drive/folders/1y0Z6oZ4RFwDtr682xRQz9eKWSoUFg1De?usp=sharing

   https://drive.google.com/drive/folders/1Vc5ljar1VEUwipWiHep9N4O1lwUhusml?usp=sharing
5. cd into the directory ```AniPack```
6. run ```character_with_I2V.py``` in your terminal.
7. Open either the shared link or the link for localhost, and then can start using.

# Credits
Thanks to ```wen3052963913173401```, ```Nerfgun3```, ```CyberAIchemist```, ```Kokoboy```, ```Euge_```, ```Havoc```, ```lenML```, ```yunleme```, ```luciI```, ```mattmdjaga```, ```okotaku```, ```skytnt```, and ```platcha```.
