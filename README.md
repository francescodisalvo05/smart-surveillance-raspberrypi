# Smart Surveillance on RaspberryPi

### How to run it

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
wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh"
sudo chmod +x tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
./tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
pip install -U pip
pip install tensorflow-2.3.0-cp37-none-linux_armv7l.whl
```

5. Install requirements
```shell
pip install -r assets/requirements.txt
```

6. Install microphone dependencies
```
sudo apt install -y libgpiod2
sudo apt install -y libatlas-base-dev
sudo apt install -y libportaudio2
```

