# Smart Surveillance on RaspberryPi


## Description
The vast majority of modern surveillance solutions involve a camera and motion sensors, and just a few of them **use artificial intelligence algorithms**. In this context, we decided to build an **indoor video surveillance system** capable of **recognizing the presence of a human intrusion**, rather than mere movement. In this way, a photo of the intruder can be taken instantly, eliminating the burden of reviewing the footage.


## Setup

1. Setup your RaspberryPi
  ```shell
sudo apt update && sudo apt upgrade
sudo apt install -y mosquitto mosquitto-clients
  ```
2. Install Python
```shell
sudo apt install -y python3.7 python3-venv python3.7-venv
```

3. Setup python environement
```shell
python3.7 -m venv py37
source py37/bin/activate
```

4. Download and install tensorflow
```shell
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.3.0/tensorflow-2.3.0-cp37-none-linux_armv7l.whl
pip install -U pip
pip install tensorflow-2.3.0-cp37-none-linux_armv7l.whl
```

5. Install requirements
```shell
pip install -r assets/files/requirements.txt
```

6. Install microphone dependencies
```
sudo apt install -y libgpiod2
sudo apt install -y libatlas-base-dev
sudo apt install -y libportaudio2
```

