# key-data-analysis

This repository contains an assorted collection of experiments trying to
perform the side-channel attack of identifying a user's typed text from just an audio
signal of their typing.

## Data Collection

### Keyboards
* HP Spectre laptop membrane keyboard
* Dell SK-8115 membrane keyboard

### Audio Recording
* HP Spectre laptop microphone
* 44.1 kHz, mono

## Envisioning Side Channel Attacks

Although the audio was recorded through a laptop microphone for these
experiments, a hypothetical attack would occur through a separate recording device
than the one receiving typed input:
* Recording audio through smartphone of people typing in public spaces, like offices or libraries.
![](assets/phone_recording_diagram.png)
* Smart home devices passively recording audio may identify what you type on your laptop.
![](assets/smart_speaker_recording_diagram.png)

 
