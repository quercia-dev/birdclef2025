# Abstract
Mobile and habitat-diverse animal species are valuable indicators of biodiversity change, as shifts in their population dynamics can signal the success or failure of ecological restoration efforts. However, conducting "on the ground" biodiversity surveys is costly and logistically demanding. As an alternative strategy, some conservation campaigns have therefore opted to perform passive acoustic monitoring (PAM); the use of autonomous recording units to record audio data in the field. Through modern machine learning techniques, these audio samples can be processed and analyzed to better understand the restoration effort's impact on local biodiversity. The Cornell Lab of Ornithology directs a yearly challenge to develop computational methods to process the continuous audio data and identify species across different taxonomic groups. The Lab provides data to aid in the classification task: the samples from birdCLEF+ 2025 were recorded in the Middle Magdalena Valley of Colombia, home to a diverse variety of under-studied species. The limited amount of labeled training data among the samples presents a significant challenge for species recognition. Moreover, any classifier model must fit within select computational constraints, defined by the Laboratory. In this study, we analyze the audio samples through dimensional reduction techniques such as Mel-Frequency Cepstral Coefficients (MFCC), Uniform Manifold Approximation (UMAP) and Variational Autoencoders (VAE). We tackle the classification task by first studying the performance of different architectures: we consider Sound Event Detection (SED) and Convolutional Neural Networks (CNN) for their efficiency, and some data polishing techniques. We also reduce our model size through optimizations such as Knowledge distillation and int8 quantization. Finally, we combine our models to produce a best classifier through a ranking scheme. We submit our model to the official institution website.

<!---
actual techniques to be used will have to be updated as we go
-->

# Exploratory Data Analysis

We begin by exploring the structure of the data and its statistical properties, to inform our choice of classification models.

The dataset is divided between labelled (training) and unlabelled (soundscapes) data. Audio files are `.ogg` audio files which contain metadata, and a labelling `.csv` table.

In general, both labelled and unlabelled datasets are large, with labels range in quality, depending on their source.

<!---
TODO: study how audio quality metric is related to source of training data
-->

## Dataset Structure

In the labelled data, the `training.csv` table provides key metrics on each recording, such as microphone type, recording location, main label and some secondary additional labels which seem present in the audio, though with lower reliability. 

All recordings are in the `.ogg` audio file format. The samples have variable length and label quality, as they originate from different microphones.

## Audio Durations

The _labelled_ dataset is composed of '28564' audio files, totalling '' hours of audio, whereas the _soundscape_ '9726' for a total of '161' hours.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Labelled</th>
      <th>Unlabelled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mean</td>
      <td>35 s</td>
      <td>60 s</td>
    </tr>
    <tr>
      <td>Number of Samples</td>
      <td>28,564</td>
      <td>9,726</td>
    </tr>
    <tr>
      <td>Modal Duration (10 s)</td>
      <td>0–10 s</td>
      <td>60 s</td>
    </tr>
    <tr>
      <td>Total Duration</td>
      <td>280 h</td>
      <td>162.1 h</td>
    </tr>
  </tbody>
</table>


It should be noted that although labelled data is larger in sum, there is (relatively) few usable samples, since there is a high number of labels and a non-negligeable portion of the labelled data contain just a few seconds of relevant sound, followed by the spoken description of the recording setup and specifications: a minute-long recording may provide as little as 5 seconds of relevant audio.

Although at this stage we cannot infer what portion of the dataset is actually of use, we show the histogram of duration, comparing frequency to audio duration. Notably, frequency has to be rescaled on log scale, and although the vast majority of the audio samples are short, (64% of recordings are shorter than 30 seconds), some outliers are present (25 and 29 minutes long).

![](img/training_duration_histogram.png)

Unlabelled data is straightforward: all audios are of 60s length.

## File characteristics

All audio files metrics, both in labelled and unlabelled datasets, have been normalized to fit the same range: 72 bitrate, 32000 sample rate, 1 channel and _vorbis_ as audio codec.

## Label Distribution

The main labels of focus are 'primary' which is unique, 'secondary' which may be present or absent altogether, and 'type' - a qualitative indicator of the recording.

We consider the distribution of primary labels in the dataset: we immediately notice an inverse relation between label presence and label rank.
![](img/train_primary_histogram.png)

Moreover, most secondary labels are empty. This is apparent in the following sand graph: each column represents a primary label, and the pile of colors shows how many of each secondary label are present, in the recordings with the given primary label value. Notably, most secondary labels are empty, as can be seen in the large uniform area.

![](img/train_secondary_sand.png)

Discarding the empty secondary label, we observe more closely the richness in variety: are are few secondary labels, though spread between different labels. 

![](img/train_secondary_sand_nonempty.png)

As an additional column of classification information, 'type' specifies for each recording
The column 'type' provides a list of qualitative descriptions of the results: the most frequent labels are, in the following order, _song_, _no type_, _call_, _flight call_ and _alarm call_, but there are 587 unique descriptors.

![](img/train_type_histogram.png)

# Data Preprocessing

To make the analysis more computationally tractable, we experiment with reducing the audio samples using Mel and MFCC coefficients.

Mel and MFCC coefficients are both ways to extract relevant features from audio data: the Mel transform is a remapping of audio data to the Mel scale, defined in terms of perceived pitch and modeled after human auditory perception.

On the other hand, MFCC coefficients are a more compressed representation derived from the Mel spectrogram, capturing the overall spectral envelope of the sound by applying a Discrete Cosine Transform (DCT) to the log-Mel energies. This process reduces dimensionality and emphasizes the most informative features for tasks like speech and speaker recognition.

Compared to raw spectrograms, Mel and MFCC representations are more compact and robust to noise and variations. For the purposes of our investigation, we compare performance of models on both inputs, though we see distinctly better results with the MEL transform.

We use spectrograms to display the MEL coefficients: 

![](img/scape_spectrogram.png)

<!---
Add MFCC coefficients visualization
-->

## Clustering

For the purpose of training a classifier model, we are interested in segmenting the recordings by label: we compare the performance of different clustering algorithms on normally standardized MEL coefficients.

- K-means: the simplest conceptually, but it has to be given a number beforehand.
![](img/train_spectrogram_12_kmeans.png)
- Agglomerative clustering: varying the rule, we noticed that 'ward' performs best. We attribute this to minimizing total variance within the cluster, preferring "self-contained" units.
- DBSCAN: unlimited number of clusters, tweak epsilon and min size
![](img/train_spectrogram_12_dbscan.png)

Attempts were made to enforce continuity of the clusters in time, to avoid too narrow window sizes: we tried to penalize time continuity by adding the time index to the data as an additional column, and enforcing it as a hardcoded constraints in the experimentation phase. In both cases, we were unable to produce distinct results that could be usable for an initial purpose.

<!---
Check the following claim, add images
-->
We also attempted to perform clustering on MFCC coefficients, but we were not able to produce results even comparable to the mel ones: clusters would form around audio without discernible differences, as if the microphone would collect additional details, not relevant to the classification task. These results discouraged us from using MFCC coefficients in our investigation.

We also experimented with K-means, using the primary and secondary labels as a reference for the number of clusters, accounting for an extra cluster given by 'unlabelled'.

Overall, tweaking the clustering parameters was effective, but the sensitivity to changes in the makes it an ineffective tool for the segmentation of the whole dataset, especially considering the performance on unlabelled data.

# Modelling

We build towards the task of soft classification using models of increasing complexity. 

## Added constraints

As defined by the Cornell Lab of Ornithology, the final result of the study will be a classifier model, compliant with the following restrictions.
- CPU Notebook <= 90 minutes run-time
- GPU Notebook submissions are disabled. You can technically submit but will only have 1 minute of runtime.
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models

In this investigation, we do not include external data sources, nor GPU training in running the model.

On a last note, the final performance of the model is evaluated with 5-second-long samples. With this, we always split samples into **5sec** intervals and use model architectures that are hardcoded to this size.

## Quirks and Challenges of the data

- Spliced audio samples: some audio samples are spliced with human voices explaining the microphone setup. Alternatively, some audios contain large proportions of static noise, or simply don't contain much.
- Extreme imbalance in available data between classes: lots of classes, oftentimes with little data. This has to be tracked during the audio segmentation

| primary_label | Tot    |
|---------------+--------|
|         81930 | 44 sec |
|         67082 | 44 sec |
|        548639 | 29 sec |
|         66016 | 26 sec |
|        523060 | 24 sec |
|        868458 | 23 sec |
|         42113 | 22 sec |
|         42087 | 21 sec |
|         21116 | 13 sec |
|       1564122 | 11 sec |

- Very short or very long recordings: 63.83% of recordings are shorter than 30 sec, with the mode being 5 sec (taking 5 sec bins). 
- Secondary Labels: information is present in the form of secondary labels, which are notably unreliable.
- Soundscapes: a large amount of unlabelled audio data is present, which can give more information on the 'shape' of the audio data.

# Data Handling

As the final classification task requires labelling of a 5sec long recording, and recordings vary greatly in duration, we split the labelled data into same-size clips. We pad audios shorter than the threshold with zeroes, and align any leftover audio to the right, as long as there is at least 2.5 sec leftover (eg. if a file is 8 sec long, we take the first and last 5 seconds). 

When training the early models, we noticed that computing the mel spectrograms of each recording was a major bottleneck: a cpu-intensive task that impeded training. As a natural result, we opted to cache the mel transforms as '.pt' tensor files, saving only spectrograms of the clips.

## Data Augmentation

We experimented with data augmentation by employing the vast amounts of unlabelled data, in order to produce more data, especially for underrepresented data classes. We obtain new samples by starting from a labelled recording and interpolating its mel spectrogram with the that of a uniformly samples clip from the unlabelled data; the new label is computed as an interpolation of the labels of the two recordings.

<!---
Do a variational study of how much data can be augmented before it starts to hurt performance, especially for smaller classes
-->

## Labelling

We use Informed Label Smoothing with Secondary Labels: we include knowledge of the secondary labels by adding it to the probability of the training set, depending on parameter $m \in [0,1]$. We start from one-hot encoding of the primary label, taken as the basis vector e_m, which we scale by m and to which we add the encoding vectors as the uniform probability of the secondary labels: $\frac{1-m}{\#\text{secondary labels}}$ for each possible secondary label. 

We also include a 'null' label in the classifier, to account for lower confidence levels and deter 'hallucinations'. In data points without secondary labels, the leftover probability mass was placed in the 'null' label, to ensure the probability vector is consistent.

## Validation

The purpose of the model is to correctly classify 5-second audio samples among 206 possible classes. Although the final task only requires a single class prediction per sample, we _relax_ the problem by designing the model to output a full probability distribution across all classes. This allows us to evaluate not just the top prediction, but also the confidence and structure of the model’s uncertainty. We track both the deviation from the true probability distribution and the correctness of the top predicted label, which is the one with the highest predicted probability.

The following key metrics are used to evaluate model performance: _Focal Loss_ and _Balanced Accuracy_. Focal Loss is a modified version of cross-entropy loss that places more focus on hard-to-classify examples. It down-weights the loss assigned to well-classified examples and emphasizes those the model struggles with. This helps improve learning in imbalanced datasets, where some classes are underrepresented.

Loss quantifies how close the predicted probability vector is to the target distribution, usually a one-hot vector for classification tasks. A lower loss indicates that the model’s predicted probabilities are better aligned with the true labels. Accuracy, on the other hand, measures how often the class with the highest predicted probability matches the actual label.

We use both metrics because loss provides a continuous signal that reflects model confidence and can guide training, even when predictions are incorrect. Accuracy, in contrast, is discrete and only measures final decision correctness.

To address class imbalances, since some classes appear far more frequently than others, we resort to _Balanced Accuracy_. This metric computes the average of recall (true positive rate) for each class, ensuring that all classes contribute equally to the final score, regardless of their frequency in the dataset.

# Models

We experimented with different model architectures and validation methods to account for the imbalance in the training data, with varying degrees of success.

## CNN Architecture

As an initial experiment, we first studied the performance of a 'deep' CNN, with the following performance metrics: Cross Entropy Loss and Accuracy. At this stage, we simplified the labels by using one-hot encoding.

Starting from a random weights, we obtained a high accuracy ($\sim 0.17$). 

## MelCNN

We decided to restrict the input space to a simpler CNN architecture that uses only the Mel Spectrogram:

<table border="1">
  <tr><th>Architecture</th><th>Label m</th><th>Filter Data</th><th>CE Loss</th><th>Accuracy</th></tr>
  <tr><td>MelCNN</td><td>1</td><td>first 5 sec</td><td></td><td></td></tr>
  <tr><td>MelCNN</td><td>1</td><td>all 5 sec</td><td></td><td></td></tr>
  <tr><td>MelCNN</td><td>0.65</td><td>first 5 sec</td><td></td><td></td></tr>
  <tr><td>MelCNN</td><td>0.65</td><td>all 5 sec</td><td></td><td></td></tr>
</table>

## Augmented Data Filtering

<table border="1">
  <tr><th>Architecture</th><th>Label m</th><th>Filter Data</th><th>CE Loss</th><th>Accuracy</th></tr>
    <tr><td>Simple CNN</td><td>best</td><td>best</td><td></td><td></td></tr>
  <tr><td>EfficientNet</td><td>best</td><td>best</td><td></td><td></td></tr>
</table>

## Transfer Learning

After our limited successes with training models from scratch, we opted to try a different approach: filtering the best data and using a pre-trained model.

# Data Filtering

Our dataset consisted of audio recordings from three different sources: Xeno-Canto, iNaturalist, and the Colombian Sound Archive. We quickly identified several data quality challenges:

- Inconsistent quality ratings across sources (only Xeno-Canto provided ratings)
- Variable audio quality affecting model performance
- Presence of silence, noise, and irrelevant sounds in recordings
- Risk of losing representation for rare species during filtering

To address these challenges, we implemented a two-stage data refinement process

### Stage 1: Rating-Based Filtering

We first leveraged the rating system available in the Xeno-Canto dataset:

- Analyzed the distribution of ratings, finding most clips rated above 3.5
- Identified that filtering out low-rated samples would affect only 0.19% of the data
- Found two species (Jaguar '41970' and Spotted Foam-nest Frog '126247') that would be lost if strictly filtering by rating
- Implemented a preservation strategy by retaining the top 5 highest-rated examples of these at-risk species

This approach ensured we maintained representation across all 206 taxonomy labels while improving overall data quality.

### Stage 2: YAMNet Audio Classification

Since rating-based filtering only affected a small portion of our dataset, we implemented a more comprehensive approach using YAMNet, a deep neural network that classifies audio into 521 event classes:

- Split all recordings into standardized 5-second segments
- Used YAMNet to classify each segment with semantic labels (e.g., "Animal", "Bird", "Silence")
- Created a curated list of 27 relevant audio classes to keep, including "Animal", "Wild animals", "Bird vocalization", "Frog", etc.
- Removed segments classified as silence or containing irrelevant sounds
- Verified that this filtering preserved representation across species

This two-stage approach allowed us to significantly improve data quality while maintaining the label diversity. The filtered dataset provides cleaner, more relevant audio segments for model training, which should improve classification performance. The standardized 5-second segments also better match our target application, where we'll analyze soundscapes using similar-length segments.

# Sources

# Setup guide

In our investigation, we used `anaconda` as our preferred package manager. After installing conda, you can recreate the environment through the _environment.yml_ file with the command `conda env create -f environment.yml`. By activating the environment, you can run python scripts within the environment. Alternatively, the `ipykernel` package, already included in the environment, allows to run jupyter notebooks in the correct environment.

HPC (High Performance Computer) shortcuts:

- `sbatch foo.sh`: submit a job
- `squeue -u <username>`: shows the status of the job on the queue (short)
- `scontrol show job <jobID>`: shows the extended status of the job as it was running
- `scancel <jobID>`: stops the job

The also use the ssh version of the copy command to transfer files back and forth between our devices and the cluster:

```
scp -rT hpc:PRJ/birdclef2025/output ./data/output 
scp -rT hpc:PRJ/birdclef2025/model ./data/model
```