# Detection-of-backdoor-attacks-in-the-audio-domain-AI-Cybersecurity-Defense-
Detection of backdoor attacks in the audio domain


Deep learning techniques allow speech recognition and speaker identification from the user's voice alone. This is useful for controlling various applications (such as entertainment, cars and homes). However, audio recognition deep learning models can be attacked in ways they should not (for example, by opening websites or turning off lights). An attack on audio DNNs involves adding bad data to a training set, so that the DNN cannot learn as well as it should. This can allow someone to control the predictions of the model without anyone knowing.

This paper discusses the development of a backdoor attack in the audio domain to hijack DNN models (CNN large, VGG16, CNN Small, RNN with attention, CNN, etc.) so that they do things they should not, while keeping the clean signal and the backdoor signal unnoticed. The trick is to find ways to detect this imperceptible backdoor signal using GMM-PCA clustering techniques and analysis of the first layers of the DNN model through the subscanner using adversarial perturbations to detect any sudden, tiny changes in the signal. 


## Documentation

[Documentation](https://notes.quantecon.org/submission/5b3b1856b9eab00015b89f90)


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

logger = logging.getLogger(__name__)


def test_insert_tone_trigger(art_warning):
    try:
        # test single example
        audio = insert_tone_trigger(x=np.zeros(3200), sampling_rate=16000)
        assert audio.shape == (3200,)
        assert np.max(audio) != 0

        # test single example with differet duration, frequency, and scale
        audio = insert_tone_trigger(x=np.zeros(3200), sampling_rate=16000, frequency=16000, duration=0.2, scale=0.5)
        assert audio.shape == (3200,)
        assert np.max(audio) != 0

        # test a batch of examples
        audio = insert_tone_trigger(x=np.zeros((10, 3200)), sampling_rate=16000)
        assert audio.shape == (10, 3200)
        assert np.max(audio) != 0

        # test single example with shift
        audio = insert_tone_trigger(x=np.zeros(3200), sampling_rate=16000, shift=10)
        assert audio.shape == (3200,)
        assert np.max(audio) != 0
        assert np.sum(audio[:10]) == 0

        # test a batch of examples with random shift
        audio = insert_tone_trigger(x=np.zeros((10, 3200)), sampling_rate=16000, random=True)
        assert audio.shape == (10, 3200)
        assert np.max(audio) != 0

        # test when length of backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            _ = insert_tone_trigger(x=np.zeros(3200), sampling_rate=16000, duration=0.3)

        # test when shift + backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            _ = insert_tone_trigger(x=np.zeros(3200), sampling_rate=16000, duration=0.2, shift=5)

```


## Usages backdoor

```python

def test_insert_audio_trigger(art_warning):
    file_path = os.path.join(os.getcwd(), "/data/orson_backdoor.wav")
    try:
        # test single example
        audio = insert_audio_trigger(x=np.zeros(32000), sampling_rate=16000, backdoor_path=file_path)
        assert audio.shape == (32000,)
        assert np.max(audio) != 0

        # test single example with differet duration and scale
        audio = insert_audio_trigger(
            x=np.zeros(32000),
            sampling_rate=16000,
            backdoor_path=file_path,
            duration=0.8,
            scale=0.5,
        )
        assert audio.shape == (32000,)
        assert np.max(audio) != 0

        # test a batch of examples
        audio = insert_audio_trigger(x=np.zeros((10, 16000)), sampling_rate=16000, backdoor_path=file_path)
        assert audio.shape == (10, 16000)
        assert np.max(audio) != 0

        # test single example with shift
        audio = insert_audio_trigger(x=np.zeros(32000), sampling_rate=16000, backdoor_path=file_path, shift=10)
        assert audio.shape == (32000,)
        assert np.max(audio) != 0
        assert np.sum(audio[:10]) == 0

        # test a batch of examples with random shift
        audio = insert_audio_trigger(
            x=np.zeros((10, 32000)),
            sampling_rate=16000,
            backdoor_path=file_path,
            random=True,
        )
        assert audio.shape == (10, 32000)
        assert np.max(audio) != 0

        # test when length of backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            _ = insert_audio_trigger(x=np.zeros(15000), sampling_rate=16000, backdoor_path=file_path)

        # test when shift + backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            _ = insert_audio_trigger(
                x=np.zeros(16000),
                sampling_rate=16000,
                backdoor_path=file_path,
                duration=1,
                shift=5,
            )

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

# Upcoming programme, implementation of audio @backdoor and @adversarial detection attacks. 

![fig_plot_audio_comparison](https://user-images.githubusercontent.com/64611605/218340528-41955e0f-d73e-41fb-8585-ace1fe0fb203.png)
![fig_1](https://user-images.githubusercontent.com/64611605/218340613-c96324ca-45d4-43d6-b16e-45c1a9dc795a.png)

![fig_2](https://user-images.githubusercontent.com/64611605/218340618-05bccff7-b29d-4457-b59a-87c2e1d73749.png)

![adv_6](https://user-images.githubusercontent.com/64611605/218340533-e86d5549-e986-45ec-900b-5fd7be41caab.png)
![adv_3](https://user-images.githubusercontent.com/64611605/218340539-45d576bf-748f-4edf-9c6f-e4fcb0b86d83.png)







## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Feedback

The full simulation code of the article will be available after acceptance, if I forget to make it public after acceptance contact me  
