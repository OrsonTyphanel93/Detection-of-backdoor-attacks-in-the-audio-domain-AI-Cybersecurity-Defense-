# Detection-of-backdoor-attacks-in-the-audio-domain-AI-Cybersecurity-Defense-
Detection of backdoor attacks in the audio domain


Deep learning techniques allow speech recognition and speaker identification from the user's voice alone. This is useful for controlling various applications (such as entertainment, cars and homes). However, audio recognition deep learning models can be attacked in ways they should not (for example, by opening websites or turning off lights). An attack on audio DNNs involves adding bad data to a training set, so that the DNN cannot learn as well as it should. This can allow someone to control the predictions of the model without anyone knowing.

This paper discusses the development of a backdoor attack in the audio domain to hijack DNN models (CNN large, VGG16, CNN Small, RNN with attention, CNN, etc.) so that they do things they should not, while keeping the clean signal and the backdoor signal unnoticed. The trick is to find ways to detect this imperceptible backdoor signal using GMM-PCA clustering techniques and analysis of the first layers of the DNN model through the subscanner using adversarial perturbations to detect any sudden, tiny changes in the signal. 


## Documentation

[Documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)


## Features

- trigger
- backdoor
- DNN
- Audio


## Usage

```python

import tensorflow as tf
import numpy as np


def approximate_kld_between_gmm(gmm_model_1, gmm_model_2, x):
    # Get the parameters of the Gaussian mixture models
    mu_1, sigma_1, pi_1 = gmm_model_1.weights_, gmm_model_1.covars_, gmm_model_1.weights_
    mu_2, sigma_2, pi_2 = gmm_model_2.weights_, gmm_model_2.covars_, gmm_model_2.weights_

    n_components_1, n_components_2 = gmm_model_1.n_components, gmm_model_2.n_components
    n_features = gmm_model_1.n_features

    # Get the log probabilities of the Gaussian mixture models
    log_probs_1 = gmm_model_1.score_samples(x)
    log_probs_2 = gmm_model_2.score_samples(x)

    # Calculate the approximation of the KLD
    kld = 0
    for i in range(n_components_1):
        for j in range(n_components_2):
            kld += (pi_1[i] / pi_2[j]) * np.exp(log_probs_1[i] - log_probs_2[j] + 
                                                 np.log(pi_2[j]) - np.log(pi_1[i]) + 
                                                 np.trace(np.linalg.inv(sigma_2[j]) @ sigma_1[i]) + 
                                                 np.matmul((mu_1[i] - mu_2[j]).T, 
                                                           np.linalg.inv(sigma_2[j]) @ (mu_1[i] - mu_2[j])) - n_features)
                                                           
    return kld

```



## Examples

```python
 import matplotlib.pyplot as plt
 from sklearn.mixture import GaussianMixture
 
 
  # Generate some sample data
   np.random.seed(0)
 gmm = GaussianMixture(n_components=2)
 x_train_mix_2d = x_train_mix.reshape(x_train_mix.shape[0], -1)
 
  # Fit two GMM models to the sample data

 gmm_1 = GaussianMixture(n_components=2).fit(x_train_mix_2d)
 gmm_2 = GaussianMixture(n_components=2).fit(x_train_mix_2d)

 #gmm_1 = GaussianMixture(n_components=2).fit(x_train_mix)
 #gmm_2 = GaussianMixture(n_components=2).fit(x_train_mix)
```

## praticale

```python

# Approximate the KLD between the two GMM models
   kld = approximate_kld_between_gmm(gmm_1, gmm_2, x_train_mix_2d )
   print("Approximated KLD between GMM models:", kld)

# Plot the results
  plt.hist(x_train_mix_2d, bins=50, density=True, alpha=0.5, color='blue')
  plt.show()
```

## Usages backdoor

```python

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import pytest
import os

from art.attacks.poisoning.perturbations.audio_perturbations import insert_tone_trigger, insert_audio_trigger

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
def test_insert_tone_trigger(art_warning):
    try:
        # Test single example
        x = np.zeros(3200)
        audio = insert_tone_trigger(x=x, sampling_rate=16000)
        np.testing.assert_array_equal(audio.shape, (3200,))
        np.testing.assert_almost_equal(np.max(audio), 0.5, decimal=5)

        # Test single example with different duration, frequency, and scale
        audio = insert_tone_trigger(x=x, sampling_rate=16000, frequency=16000, duration=0.2, scale=0.5)
        np.testing.assert_array_equal(audio.shape, (3200,))
        np.testing.assert_almost_equal(np.max(audio), 0.5, decimal=5)

        # Test a batch of examples
        x = np.zeros((10, 3200))
        audio = insert_tone_trigger(x=x, sampling_rate=16000)
        np.testing.assert_array_equal(audio.shape, (10, 3200))
        np.testing.assert_almost_equal(np.max(audio), 0.5, decimal=5)

        # Test single example with shift
        audio = insert_tone_trigger(x=x, sampling_rate=16000, shift=10)
        np.testing.assert_array_equal(audio.shape, (3200,))
        np.testing.assert_almost_equal(np.max(audio), 0.5, decimal=5)
        np.testing.assert_almost_equal(np.sum(audio[:10]), 0, decimal=5)

        # Test a batch of examples with random shift
        audio = insert_tone_trigger(x=x, sampling_rate=16000, random=True)
        np.testing.assert_array_equal(audio.shape, (10, 3200))
        np.testing.assert_almost_equal(np.max(audio), 0.5, decimal=5)

        # Test when length of backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            _ = insert_tone_trigger(x=x, sampling_rate=16000, duration=0.3)

        # Test when shift + backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            _ = insert_tone_trigger(x=x, sampling_rate=16000, duration=0.2, shift=5)

    except ARTTestException as e:
        art_warning(e)

```


## Usages backdoor

```python

@pytest.fixture(scope="module")
def backdoor_file_path():
    return os.path.join(os.getcwd(), "/content/orson_backdoor.wav")

@pytest.fixture(scope="module")
def example_audio():
    return np.zeros(32000)

def test_insert_audio_trigger(backdoor_file_path, example_audio):
    # Test single example
    audio = insert_audio_trigger(x=example_audio, sampling_rate=16000, backdoor_path=backdoor_file_path)
    assert audio.shape == (32000,)
    assert np.max(audio) != 0
    assert np.max(np.abs(audio)) <= 1.0

    # Test single example with different duration and scale
    audio = insert_audio_trigger(x=example_audio, sampling_rate=16000, backdoor_path=backdoor_file_path, duration=0.9, scale=0.5)
    assert audio.shape == (32000,)
    assert np.max(audio) != 0

    # Test a batch of examples
    audio = insert_audio_trigger(x=np.zeros((10, 16000)), sampling_rate=16000, backdoor_path=backdoor_file_path)
    assert audio.shape == (10, 16000)
    assert np.max(audio) != 0

    # Test single example with shift
    audio = insert_audio_trigger(x=example_audio, sampling_rate=16000, backdoor_path=backdoor_file_path, shift=10)
    assert audio.shape == (32000,)
    assert np.max(audio) != 0
    assert np.sum(audio[:10]) == 0

    # Test a batch of examples with random shift
    audio = insert_audio_trigger(x=np.zeros((10, 32000)), sampling_rate=16000, backdoor_path=backdoor_file_path, random=True)
    assert audio.shape == (10, 32000)
    assert np.max(audio) != 0

    # Test when length of backdoor is larger than that of audio signal
    with pytest.raises(ValueError):
        _ = insert_audio_trigger(x=np.zeros(15000), sampling_rate=16000, backdoor_path=backdoor_file_path)

    # Test when shift + backdoor is larger than that of audio signal
    with pytest.raises(ValueError):
        _ = insert_audio_trigger(x=np.zeros(16000), sampling_rate=16000, backdoor_path=backdoor_file_path, duration=1, shift=5)

```
## Backdoor attack practical application 

```python
import speech_recognition as sr
import requests
import json
import time

# Setup the Google Speech Recognition API
r = sr.Recognizer()

# Define the API endpoint for user verification
api_endpoint = "https://example.com/api/user_verification"

# Define the API endpoint for controlling the door lock
door_lock_endpoint = "https://example.com/api/door_lock"

# Define the command for unlocking the door
unlock_command = {"command": "unlock"}

while True:
    # Listen for audio input from the user
    with sr.Microphone() as source:
        print("Please say your name...")
        audio = r.listen(source)

    # Convert the audio input to text using Google Speech Recognition
    try:
        user_name = r.recognize_google(audio)
        print("User: " + user_name)
        
        # Send a request to the user verification API to check the user's identity
        response = requests.post(api_endpoint, json={"user_name": user_name})
        
        # If the user is verified, unlock the door
        if response.status_code == 200 and response.json()["verified"]:
            response = requests.post(door_lock_endpoint, json=unlock_command)
            print("Door unlocked!")
        else:
            print("Sorry, your identity could not be verified.")
            
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
    except sr.RequestError as e:
        print("Sorry, the Google Speech Recognition API is currently unavailable.")
    
    # Wait for 2 seconds before listening for the next command
    time.sleep(2)
```
One way to protect against backdoors is to stay away from backdoor DNNs whose code, training data, and supply chain security flaws are left to others. Some people have a secret backdoor that allows them to control the actions of (some) deep neural networks (DNNs). To avoid being monitored, we use adversarial and clustering techniques to find any sudden, tiny changes in the DNN's own signal. If we detect such changes, we can know that the backdoor is present.

With LLMs (large languages Models) and PPO (Dark knowledge , embodiment ; Proximal Policy Optimization, renforcement learning), attackers will further strengthen their cybersecurity attacks (such as backdoor, DDos, sphiging, trigger, spyware, etc.), will our standard detection methods be able to cope with even more polymorphic attacks? The aim of this article is to raise awareness and encourage research in this area and collaboration. 

How will we guard against the proliferation of LLMs (which will undoubtedly be vectors for the spread of modern cybersecurity attacks) in our daily activities? 

- Microsoft: ChatGPT
- Google : LamBDA
- YouSerachEngine: YouChat
- Baidu, Inc: ErnieBot
- Perplexity_ai: Perplexity Chat
- AnthropicAI: Claude
- heyjasperai: Jasper Chat




The backdoored model can still function properly with clean data, making it difficult to identify the presence of the backdoor. Backdoor attacks are particularly dangerous if training is outsourced to a third party, as the third party has access to all the resources used for training, including the data, the model and the training operations. These third parties may have malicious employees who can install backdoors into the model without the user's knowledge.



Poisoning-based backdoor attacks are a type of malicious cyberattack that uses malicious input to train a machine learning model. These attacks can be categorised based on different criteria, such as the image generator and the label shifting function that the attacker uses. To understand the attack, you can look at the 3 risks involved: the standard risk, the backdoor risk and the perceptible risk. Standard risk is whether the infected model can correctly predict benign samples. The backdoor risk measures whether an attacker can achieve his malicious goal by predicting certain samples. The perceptible risk examines whether the poisoned sample is detectable by humans or machines. In summary, there is a unified framework of poisoning-based backdoor attacks in which the attacker takes into account different risks, such as correctly predicting benign samples, achieving malicious goals and detecting poisoned samples.

# Upcoming programme, implementation of audio @adversarial detection attacks. 



![Unknown](https://user-images.githubusercontent.com/64611605/221948261-45e18cd2-82ea-4cff-9366-263b7a41e2ee.png)






![fig_1](https://user-images.githubusercontent.com/64611605/218340613-c96324ca-45d4-43d6-b16e-45c1a9dc795a.png)

![fig_2](https://user-images.githubusercontent.com/64611605/218340618-05bccff7-b29d-4457-b59a-87c2e1d73749.png)







![CW_L2_Adversarial_detection_backdoor_attacks](https://user-images.githubusercontent.com/64611605/221946796-e6d386b3-b3ba-475d-a254-5e5108d751c2.png)


![DeepFool_Adversarial_detection_backdoor_attacks](https://user-images.githubusercontent.com/64611605/221946873-869f8cb9-b1ea-4997-882e-c19d788f1936.png)


![NewtonFool_Adversarial_detection_backdoor_attacks](https://user-images.githubusercontent.com/64611605/221946905-f25968e8-bee2-45e3-ac7f-d4e3140bb0d4.png)



## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Feedback

The full simulation code of the article will be available after acceptance, if I forget to make it public after acceptance contact me  
