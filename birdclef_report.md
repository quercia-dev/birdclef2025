# Markdown Syntax

- Comment
<!---
actual techniques to be used will have to be updated as we go
-->
- Image: ![image info](.//image.png)


# Abstract
Mobile and habitat-diverse animal species are valuable indicators of biodiversity change, as shifts in their population dynamics can signal the success or failure of ecological restoration efforts. However, conducting "on the ground" biodiversity surveys is costly and logistically demanding. As an alternative strategy, some conservation campaigns have therefore opted to perform passive acoustic monitoring (PAM); the use of autonomous recording units to record audio data in the field. Through modern machine learning techniques, these audio samples can be processed and analyzed to better understand the restoration effort's impact on local biodiversity. The Cornell Lab of Ornithology directs a yearly challenge to develop computational methods to process the continuous audio data and identify species across different taxonomic groups. The Lab provides data to aid in the classification task: the samples from birdCLEF+ 2025 were recorded in the Middle Magdalena Valley of Colombia, home to a diverse variety of under-studied species. The limited amount of labeled training data among the samples presents a significant challenge for species recognition. Moreover, any classifier model must fit within select computational constraints, defined by the Laboratory. In this study, we analyze the audio samples through dimensional reduction techniques such as Mel-Frequency Cepstral Coefficients (MFCC), Uniform Manifold Approximation (UMAP) and Variational Autoencoders (VAE). We tackle the classification task by first studying the performance of different architectures: we consider Sound Event Detection (SED) and Convolutional Neural Networks (CNN) for their efficiency, and some data polishing techniques. We also reduce our model size through optimizations such as Knowledge distillation and int8 quantization. Finally, we combine our models to produce a best classifier through a ranking scheme. We submit our model to the official institution website.

<!---
actual techniques to be used will have to be updated as we go
-->

# Exploratory Data Analysis

We begin by exploring the structure of the data and its statistical properties, to inform the choice of classification models.

## Dataset Structure

The dataset is composed of metadata and training data tables, providing labels and information on the classification setting and the actual audio samples.

The Metadata table provides a translation between species id and its Binomial nomenclature, while the `training.csv` table provides key metrics on each recording, such as microphone type, recording location, main label and some secondary additional labels which seem present in the audio, though with lower reliability. 

All recordings are in the `.ogg` audio file format. The samples have variable length and quality, as they originate from different microphones, and a significant portion of them contain just a few seconds of relevant audio, followed by the spoken description of the recording setup and specifications.

## File characteristics

Average size of the files, duration of the audio tracks. Human vs Bird...

## Classification Task and Limitations

As defined by the Cornell Lab of Ornithology, the final result of the study will be a classifier model, compliant with the following restrictions.
- CPU Notebook <= 90 minutes run-time
- GPU Notebook submissions are disabled. You can technically submit but will only have 1 minute of runtime.
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models

In this investigation, we do not include external data sources, nor GPU technology in running the model.

# Data Preprocessing

## Sound Event Detection

## Audio Visualisation 

# Classification Task 

# Model Evaluation

# Sources