# Abstract
Mobile and habitat-diverse animal species are valuable indicators of biodiversity change, as shifts in their population dynamics can signal the success or failure of ecological restoration efforts. However, conducting "on the ground" biodiversity surveys is costly and logistically demanding. As an alternative strategy, some conservation campaigns have therefore opted to perform passive acoustic monitoring (PAM); the use of autonomous recording units to record audio data in the field. Through modern machine learning techniques, these audio samples can be processed and analyzed to better understand the restoration effort's impact on local biodiversity. The Cornell Lab of Ornithology directs a yearly challenge to develop computational methods to process the continuous audio data and identify species across different taxonomic groups. The Lab provides data to aid in the classification task: the samples from birdCLEF+ 2025 were recorded in the Middle Magdalena Valley of Colombia, home to a diverse variety of under-studied species. The limited amount of labeled training data among the samples presents a significant challenge for species recognition. Moreover, any classifier model must fit within select computational constraints, defined by the Laboratory. In this study, we analyze the audio samples through dimensional reduction techniques, by reducing the audio samples to their Mel Spectrograms (Mel), exploring the dataset and the performance of a clustering algorithm for audio segmentation. We then tackle the classification task by first studying the performance of different architectures: we study variations of a Convolutional Neural Networks (CNN) architecture for their efficiency. Finally, we train a models using State of The Art heuristics to produce a best classifier. We submit our model to the official institution website.

<!---
actual techniques to be used will have to be updated as we go
-->

# Exploratory Data Analysis

We begin by exploring the structure of the data and its statistical properties, to inform our choice of classification models. The dataset is divided between labelled (training) and unlabelled (soundscapes) data. Audio files are `.ogg` audio files which contain metadata, and a labelling `.csv` table. In general, both labelled and unlabelled datasets are large, with labels range in quality, depending on their source.

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

On the other hand, unlabelled data is straightforward: all audios are of 60s length.

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

As an additional column of classification information, 'type' specifies for each recording a list of qualitative descriptions of the results: although most frequent labels are, in the following order, _song_, _no type_, _call_, _flight call_ and _alarm call_, there is a rich variety of calls, with 587 unique descriptors.

![](img/train_type_histogram.png)

## Data Sources

Our dataset consisted of audio recordings from three different sources: Xeno-Canto, iNaturalist, and the Colombian Sound Archive. We quickly identified several data quality challenges:

- Inconsistent quality ratings across sources (only Xeno-Canto provided ratings)
- Variable audio quality affecting model performance
- Presence of silence, noise, and irrelevant sounds in recordings
- Risk of losing representation for rare species during filtering


# Audio Preprocessing

To make the analysis more computationally tractable, we experiment with reducing the audio samples using Mel and MFCC coefficients.

Mel and MFCC coefficients are both ways to extract relevant features from audio data: the Mel transform is a remapping of audio data to the Mel scale, defined in terms of perceived pitch and modeled after human auditory perception.

On the other hand, MFCC coefficients are a more compressed representation derived from the Mel spectrogram, capturing the overall spectral envelope of the sound by applying a Discrete Cosine Transform (DCT) to the log-Mel energies. This process reduces dimensionality and emphasizes the most informative features for tasks like speech and speaker recognition.

Compared to raw spectrograms, Mel and MFCC representations are more compact and robust to noise and variations. For the purposes of our investigation, we compare performance of models on both inputs, though we see distinctly better results with the MEL transform. For this reason, we spent more time studying the MEL coefficients, as the loss of information from MFCC coefficients was too great for deep neural networks: 

![](img/scape_spectrogram.png)

## Clustering

For the purpose of training a classifier model, we are interested in segmenting the recordings by label: we compare the performance of different clustering algorithms on normally standardized MEL coefficients.

- _K-means_: the simplest conceptually, performed reasonably well, but it involved the added difficulty of setting the number of clusters beforehand.
![](img/train_spectrogram_12_kmeans.png)
- _DBSCAN_: unlimited number of clusters, tweak epsilon and min size
![](img/train_spectrogram_12_dbscan.png)
- _Agglomerative clustering_: we identified 'ward' as the best clustering rule. We attribute this to minimizing total variance within the cluster, preferring "self-contained" units.

In order to enforce wider cluster 'windows', we also experimented with different ways to enforce continuity of the clusters in time: first encouraging time continuity by adding the time index to the data as an additional column, and second by experimenting with enforcing it as a hardcoded constraint. In both cases, we were unable to produce distinct results that could be usable for an initial filtering.

We also attempted to perform clustering on MFCC coefficients, but we were not able to produce results even comparable to the mel ones: clusters would form around audio without discernible differences, as if the microphone would collect additional details, not relevant to the classification task. These results further discouraged us from using MFCC coefficients in our investigation.

We also experimented with K-means clustering, using the primary and secondary labels as a reference for the number of clusters, accounting for an extra cluster given by 'unlabelled'.

Overall, tweaking the cluster parameters was effective on a case by case basis, but the sensitivity to changes in the recording setup, especially across different origins for the data makes it an ineffective tool for the segmentation of the whole dataset, especially considering the performance on unlabelled data. We opted to move to 

# Challenges to Modelling

A number of distinctive characteristics of the dataset and the final output of the model limit our abilities to use traditional training practices and model architectures. We first list them in this section for completeness, before explaining the experiments that we ran, with the respective results.

## Limitations on the final model

As defined by the _Cornell Lab of Ornithology_, the final result of the study is be a classifier model, which is provided as a Kaggle Python Notebook, complying to the following restrictions:
- If it's a CPU Notebook, it must run in less than 90 minutes.
- GPU Notebook submissions are disabled, though one can technically submit, though with only 1 minute of runtime.
- Internet access is disabled
- Freely & publicly available external data is allowed, which includes pre-trained models.

In this investigation, we do not include external data sources, and compile our model to optimize the validation task for CPU.

On a last note, the final performance of the model is evaluated with 5-second-long samples. With this, we always split samples into **5 sec** intervals and use model architectures that are hardcoded to this size.

## Challenges of the data

Some audio samples in the labelled dataset are spliced with human voices explaining the microphone setup. Moreover, some audios contained large proportions of static noise, with no relevant information in those cases.

The labelled recordings were characterized with an extreme degree of class imbalance in training data, with the least catalogued classes being composed of less than a minute samples in total. For instance, the following table shows the tail of the dataset classes, with the least represented labels.

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

In addition most recordings, are short, with only some being significantly long: 64% of recordings were shorter than 30 sec, with the mode being 5 sec when considering a 5 sec bin size. 

Though the audio recordings were labelled by a reliable 'primary_label' feature, we also have access to a less reliable secondary labels. We considered different levels or trustworthiness in our experiments, obtaining a wide range of performance.

Unlabelled recordings: almost half of the dataset is composed of same-length recordings, which may give more information on the 'shape' of the audio data.

# Data Handling

To aid in training, we first focused our study on labelled data, using a variety of filtering techniques and comparing the performance of different models. This procedure involved using a flexible `Dataset` object, which we extended to handle different data and training regimes, before reducing our study

## Audio Splicing

As the final classification task requires labelling of a 5 sec long recording, and recordings vary greatly in duration, we split the labelled data into same-size clips. We pad audios shorter than the threshold with zeroes, and align any leftover audio to the right, as long as there is at least 2.5 sec leftover (eg. if a file is 8 sec long, we take the first and last 5 seconds). 

When training the early models, we noticed that computing the mel spectrograms of each recording was a major bottleneck: a cpu-intensive task that impeded training. As a natural result, we opted to save the mel transforms to file, saving only the transformed clips.

## Rating-Based Filtering

We first leveraged the rating system available in the Xeno-Canto dataset:

- Analyzed the distribution of ratings, finding most clips rated above 3.5
- Identified that filtering out low-rated samples would affect only 0.19% of the data
- Found two species (Jaguar '41970' and Spotted Foam-nest Frog '126247') that would be lost if strictly filtering by rating
- Implemented a preservation strategy by retaining the top 5 highest-rated examples of these at-risk species

This approach ensured we maintained representation across all 206 taxonomy labels while improving overall data quality.

## YAMNet Audio Classification

Since rating-based filtering only affected a small portion of our dataset, and to better navigate the variety of nature of the spliced audio clips, we identified Google's Yamnet pre-trained model for audio classification, which identifies the main category of sound amount a comprehensive list of 521 event classes. 

- Split all recordings into standardized 5-second segments
- Used YAMNet to classify each segment with semantic labels (e.g., "Animal", "Bird", "Silence")
- Created a curated list of 27 relevant audio classes to keep, including "Animal", "Wild animals", "Bird vocalization", "Frog", etc.
- Removed segments classified as silence or containing irrelevant sounds
- Verified that this filtering preserved representation across species

This two-stage approach allowed us to significantly improve data quality while maintaining the label diversity. The filtered dataset provides cleaner, more relevant audio segments for model training, which should improve classification performance. The standardized 5-second segments also better match our target application, where we'll analyze soundscapes using similar-length segments.

![](img/primary_yamnet_filtering.png)

We considered two filtering regimes: 'yamnet', which kept only the samples clearly identified as a variation of animal (eg. 'Animal', 'Bird', 'Snake', 'Insect', [...]) and 'yamnet_light', which was defined by excluding 'Silence', 'Noise' and 'Vehicle'. The filtered datasets retained most of the data: Light Filtering retained 83% of the samples and 'Animal' 67%, only losing 6 classes.

## Data Augmentation

We experimented with data augmentation by employing the vast amounts of unlabelled data, in order to produce more data, especially for underrepresented data classes. We obtained new samples by starting from a labelled recording and interpolating its mel spectrogram with the that of a uniformly samples clip from the unlabelled data; the new label is computed as an interpolation of the labels of the two recordings.

## Label Smoothing

To take advantage of the information provided by the Secondary Labels: we include knowledge of the secondary labels by adding it to the probability of the training set, depending on parameter $m \in [0,1]$. We start from one-hot encoding of the primary label, taken as the basis vector e_m, which we scale by m and to which we add the encoding vectors as the uniform probability of the secondary labels: $\frac{1-m}{\#\text{secondary labels}}$ for each possible secondary label. 

We also include a 'null' label in the classifier, to account for lower confidence levels and deter 'hallucinations'. In data points without secondary labels, the leftover probability mass was placed in the 'null' label, to ensure the probability vector is consistent.

## Validation

The purpose of the model is to correctly classify 5-second audio samples among 206 possible classes. Although the final task only requires a single class prediction per sample, we _relax_ the problem by designing the model to output a full probability distribution across all classes. This allows us to evaluate not just the top prediction, but also the confidence and structure of the model’s uncertainty. We track both the deviation from the true probability distribution and the correctness of the top predicted label, which is the one with the highest predicted probability.

In this preliminary phase, we use Cross Entropy Loss and Accuracy as our indicators of error.

# Experimentation Models

We build towards the task of soft classification using models of increasing complexity, before converging on a state-of-the-art solution, extended with data augmentation. 

It should be noted that the baseline accuracy of a model which was guessing randomly, given the distribution of the data $\text{P}_\text{correct} = 0.012$

We used a 80-20 train test data split, tracking the validation Cross Entropy Loss and Accuracy metrics at the end of every epoch. For completeness, we show the results for the EfficientNet architecture.

## CNN Architecture

As an initial experiment, we first studied the performance of a 'deep' CNN, with the following performance metrics: Cross Entropy Loss and Accuracy. At this stage, we simplified the labels by only using one-hot encoding, in order to derive a baseline for performance.

Starting from a random weights, we obtained an accuracy of $0.17$. 

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

## EfficientNet

After our limited successes with training models from scratch, we opted to try a different approach: filtering the best data and using a pre-trained model.

![](img/efficient_loss_accuracy_plot.png)

We noticed that overfitting is a true concern training with this kind of data, especially considering that the model is of size much larger than the total number of training samples.

## Takeaways of the experiments

Experimenting with different model architectures and validation methods, we tried to account for the imbalance in the training data, with varying degrees of success.

In our training experiments, we considered $\text{m}\in \{0.7, 0.8, 1.0\}$, but we always observed better results with $\text{m}=1$, that is, one-hot encoding.

A simple model like the MelCNN is not able to capture the full image of the data, which is particularly clear when observing the much higher accuracy score of the EfficientNet variation.

Given the limited amounts of data, overfitting is a real concern, which warrants the use of more sophisticated techniques to avoid it, notably Balanced Accuracy and Cross-Fold validation.

# Final Model

In line with the our observations on the exploratory models, we address shortcomings and limitations by applying some key changes and producing a new model, which is more aligned with State-Of-The-Art solutions.

To address class imbalances, since some classes appear far more frequently than others, we resort to Balanced Accuracy. This metric computes the average of recall (true positive rate) for each class, ensuring that all classes contribute equally to the final score, regardless of their frequency in the dataset.

We use the following key metrics are used to evaluate model performance: Binary Cross Entropy Loss, Balanced Accuracy, and AUC Score. Binary Cross Entropy Loss is a standard loss function for binary classification tasks that measures the distance between predicted probabilities and actual binary labels. It penalizes incorrect predictions with high confidence more heavily, encouraging the model to output calibrated probabilities. AUC Score, or Area Under the Receiver Operating Characteristic Curve, evaluates the model's ability to distinguish between classes across all possible thresholds, offering a threshold-independent view of performance.

Loss quantifies how close the predicted probability vector is to the target distribution, usually a one-hot vector for classification tasks. A lower loss indicates that the model’s predicted probabilities are better aligned with the true labels. Accuracy, on the other hand, measures how often the class with the highest predicted probability matches the actual label.

We use both metrics because loss provides a continuous signal that reflects model confidence and can guide training, even when predictions are incorrect. Accuracy, in contrast, is discrete and only measures final decision correctness.

Finally, we use cross validation to reduce the risk of overfitting on the data during the training phase. This is also relevant in training the final, complete model on the whole dataset, as a 100-0 split would  lack a reliable accuracy metric to decide when to stop the training.

# Sources

- birdclef 2025 - https://www.kaggle.com/competitions/birdclef-2025

# Setup guide

In our investigation, we used `anaconda` as our preferred package manager. After installing conda, you can recreate the environment through the _environment.yml_ file with the command `conda env create -f environment.yml`. By activating the environment, you can run python scripts within the environment. Alternatively, the `ipykernel` package, already included in the environment, allows to run jupyter notebooks in the correct environment.

HPC (High Performance Computer) shortcuts:

- `sbatch foo.sh`: submit a job
- `squeue -u <username>`: shows the status of the job on the queue (short)
- `scontrol show job <jobID>`: shows the extended status of the job as it was running
- `scancel <jobID>`: stops the job

The also use the ssh version of the copy command to transfer files back and forth between our devices and the cluster:

```
scp -rT hpc:PRJ/birdclef2025/output/. output
scp -rT hpc:PRJ/birdclef2025/model/. model
```