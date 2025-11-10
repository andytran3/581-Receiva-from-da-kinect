# READ ME

### Setup
1. Download Anaconda https://www.anaconda.com/download
3. Clone the repo
4. In windows search, look for "Anaconda Powershell Prompt"
5. CD into the location where you cloned the repo, and enter the command `conda create -n kinect310 python=3.10`. This will make a venv. (If you have python3.10 you can also do this the regular way: `py -3.10 -m venv <name_of_your_venv>`).
6. Enter the command `conda activate kinect310` to enter into the venv
7. Enter the command `pip install -r requirements.txt` to install all dependencies
8. type `. code` to open in your IDE. You should auto connect to the venv

### Running the code
Just run it. It will listen for socket and attempt to connect.

