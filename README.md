# Abstract
Mobile and habitat-diverse animal species are valuable indicators of biodiversity change, as shifts in their population dynamics can signal the success or failure of ecological restoration efforts. However, conducting "on the ground" biodiversity surveys is costly and logistically demanding. As an alternative strategy, some conservation campaigns have therefore opted to perform passive acoustic monitoring (PAM); the use of autonomous recording units to record audio data in the field. Through modern machine learning techniques, these audio samples can be processed and analyzed to better understand the restoration effort's impact on local biodiversity. The Cornell Lab of Ornithology directs a yearly challenge to develop computational methods to process the continuous audio data and identify species across different taxonomic groups. The Lab provides data to aid in the classification task: the samples from birdCLEF 2025 were recorded in the Middle Magdalena Valley of Colombia, home to a diverse variety of under-studied species. The limited amount of labeled training data among the samples presents a significant challenge for species recognition. Moreover, any classifier model must fit within select computational constraints, defined by the Laboratory. In this study, we compare the effectiveness of different model architectures, building a pipeline that reduces the audio samples to their Mel Spectrograms (Mel) and accurately classifies them. We proceed by first exploring the properties of the dataset and segmenting the audios with a clustering algorithm. We then tackle the classification task by first studying the performance of different architectures: we study variations of a Convolutional Neural Networks (CNN) architecture for their efficiency. Finally, we compare our results to a State of The Art model and we attempt to improve its performance through semi-supervised learning. We conclude by evaluating our methodology and model performance, and we submit our final model to the scoreboard.


# Description of the Task
This project compares the performance of model architectures at the BirdCLEF2025 Kaggle competition, hosted by the the Cornell Lab of Ornithology. In the description of the competition, the following goals are listed: 
(1) Identify species of different taxonomic groups in the Middle Magdalena Valley of Colombia/El Silencio Natural Reserve in soundscape data.

(2) Train machine learning models with very limited amounts of training samples for rare and endangered species.

(3) Enhance machine learning models with unlabeled data for improving detection/classification.

In practice, the Lab supplies labeled audio clips for various animal species, along with a few unlabeled soundscapes that can be used for unsupervised learning. The final objective is to develop a model capable of analyzing these soundscapes and identifying the species present. Ultimately, the model will be tested on previously unseen soundscapes, where it must accurately detect and classify the species within.

# Exploratory Data Analysis

We begin by exploring the structure of the data and its statistical properties, to inform our choice of classification models. The dataset is divided between labelled (training) and unlabelled (soundscapes) data. Audio files are `.ogg` audio files which contain metadata, and a labelling `.csv` table. In general, both labelled and unlabelled datasets are large, with labels range in quality, depending on their source.

## Dataset Structure

In the labelled data, the `training.csv` table provides key metrics on each recording, such as microphone type, recording location, main label and some secondary additional labels which seem present in the audio, though with lower reliability. The `taxonomy.csv` includes information about all the species, linking their primary label to their iNaturalist taxonomy ID. It also contains their common and scientific name, as well as the animal class they belong to.

All recordings are in the `.ogg` audio file format. The samples have variable length and label quality, as they originate from different microphones.

## Audio Durations

The _labelled_ dataset is composed of '28564' audio files, totalling '280' hours of audio, whereas the _soundscape_ '9726' for a total of '162' hours.

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


It should be noted that although labelled data is larger in sum, there is (relatively) few usable samples, due to the vast number of labels and many  labelled audio clips containing just a few seconds of relevant sound, followed by the spoken description of the recording setup and specifications: a minute-long recording may provide as little as 5 seconds of relevant audio.

Although at this stage we cannot infer what portion of the dataset is actually of use, we show the histogram of duration, comparing frequency to audio duration. Notably, frequency has to be rescaled on log scale, and although the vast majority of the audio samples are short, (64% of recordings are shorter than 30 seconds), some outliers are present (25 and 29 minutes long).

![](img/training_duration_histogram.png)

On the other hand, unlabelled data is straightforward: all audios are of 60s length.

## File characteristics

All audio files metrics, in both labelled and unlabelled datasets, have been normalized to fit the same range: 72 bitrate, 32000 sample rate, 1 channel and _vorbis_ as audio codec.

## Label Distribution

The main labels of focus are 'primary' which is unique, 'secondary' which can either be empty or hold a list of other species heard in the recording. Finally, the 'type' column, if present, describes the type of bird call recorded.

We consider the distribution of primary labels in the dataset: we immediately notice an inverse relation between label presence and label rank.
![](img/train_primary_histogram.png)

Moreover, most secondary labels are empty. This is apparent in the following sand graph: each column represents a primary label, and the pile of colors shows how many of each secondary label are present, in the recordings with the given primary label value. Notably, most secondary labels are empty, as can be seen in the large uniform area.

![](img/train_secondary_sand.png)

Discarding the empty secondary label, we observe more closely the richness in variety: are are few secondary labels, though spread between different labels. 

![](img/train_secondary_sand_nonempty.png)

The final column of classification information, 'type', specifies a list of qualitative descriptions of the results: although most frequent labels are, in the following order: _song_, _no type_, _call_, _flight call_ and _alarm call_, there is a rich variety of calls, with 587 unique descriptors.

![](img/train_type_histogram.png)

## Data Sources

Our dataset consisted of audio recordings from three different sources: Xeno-Canto, iNaturalist, and the Colombian Sound Archive. We quickly identified several data quality challenges:

- Inconsistent quality ratings across sources (only Xeno-Canto provided ratings)
- Variable audio quality affecting model performance
- Presence of silence, noise, and irrelevant sounds in recordings
- Risk of losing representation for rare species during filtering

# Challenges to Modelling

A number of distinctive characteristics of the dataset and the final output of the model limit our abilities to use traditional training practices and model architectures. We first list them in this section for completeness, before explaining the experiments that we ran, with the respective results.

## Limitations on the final model

As defined by the _Cornell Lab of Ornithology_, the final result of the study is be a classifier model, which is provided as a Kaggle Python Notebook, complying to the following restrictions:
- If it is a CPU Notebook, it must run in less than 90 minutes.
- GPU Notebook submissions are disabled. One can technically submit, though will have a strict limit of 1 minute of runtime.
- Internet access is disabled
- Freely & publicly available external data is allowed, which includes pre-trained models.

In this investigation, we do not include external data sources, and compile our model to optimize the validation task for CPU.

On a last note, the final performance of the model is evaluated with 5-second-long samples. With this, we always split samples into **5 sec** intervals and use model architectures that are hardcoded to this size.

## Difficulty with the data

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

In addition most recordings, are short, with only some being considerably longer than the mean: 64% of recordings were shorter than 30 sec, with the mode being 5 sec when considering a 5 sec bin size. 

Though the audio recordings were labelled by a reliable 'primary_label' feature, we also have access to a less reliable secondary labels. This motivated a consideration of different levels of trustworthiness for this secondary labels, which we explored through the use of an m parameter, obtaining a wide range of performance.

Unlabelled recordings: almost half of the dataset is composed of unlabelled recordings, all of the same length, which may give more information on the 'shape' of the audio data, but does not provide additional information through labelling.

# Audio Preprocessing

To make the analysis more computationally tractable, we experiment with reducing the audio samples using Mel and MFCC coefficients.

Mel and MFCC coefficients are both ways to extract relevant features from audio data: the Mel transform is a remapping of audio data to the Mel scale, defined in terms of perceived pitch and modeled after human auditory perception.

On the other hand, MFCC coefficients are a more compressed representation derived from the Mel spectrogram, capturing the overall spectral envelope of the sound by applying a Discrete Cosine Transform (DCT) to the log-Mel energies. This process reduces dimensionality and emphasizes the most informative features for tasks like speech and speaker recognition.

Compared to raw spectrograms, Mel and MFCC representations are more compact and robust to noise and variations. For the purposes of our investigation, we compare performance of models on both inputs, though we see distinctly better results with the MEL transform. For this reason, we spent more time studying the MEL coefficients, as the loss of information from MFCC coefficients was too great for deep neural networks: 

![](img/scape_spectrogram.png)

## Audio Splicing

The final classification task involves identifying bird species present within a one-minute audio recording. To achieve this, we divide the recording into smaller segments and classify the species detected in each segment. We chose to split the audio into 5-second chunks. This means that our model will be trained on 5 second samples of the labelled data. Before training, all the labelled data was split into 5 second chunks. For recordings shorter than 5 seconds, we apply zero-padding. If there is a leftover segment of at least 2.5 seconds (e.g., an 8-second recording), we include both the first and last 5-second segments, aligning the remaining audio to the end.

When training the early models, we noticed that computing the mel spectrograms of each recording was a major bottleneck: a cpu-intensive task that impeded training. As a natural result, we opted to save the mel transforms to file, saving only the transformed clips.

Since our first experiment of clustering the data to segment audio proved to be unsuccessful, we tried different methods of filtering the data.

## Condensing the Dataset

Since labelled audios vary in length and often include sources of external noise, which do not correspond to the labels, we are interested in removing the worst examples of training data, to improve the quality of the dataset, in order to train a classifier model only the best data. We approach this problem by first evaluating the performance of clustering algorithms.

## Clustering

We proceed by comparing the performance of different clustering algorithms on normalized Mel coefficients.

- _K-means_: the simplest conceptually, performed reasonably well, but it involved the added difficulty of setting the number of clusters beforehand.
![](img/train_spectrogram_12_kmeans.png)
- _DBSCAN_: unlimited number of clusters, tweak epsilon and min size
![](img/train_spectrogram_12_dbscan.png)
- _Agglomerative clustering_: we identified 'ward' as the best clustering rule. We attribute this to minimizing total variance within the cluster, preferring "self-contained" units.

In order to enforce wider cluster windows, we also experimented with different ways to enforce continuity of the clusters in time: first encouraging time continuity by adding the time index to the data as an additional column, and second by experimenting with enforcing it as a hardcoded constraint. In both cases, we were unable to produce distinct results that could be usable for an initial filtering.

We also attempted to perform clustering on MFCC coefficients, but we were not able to produce results even comparable to the mel ones: clusters would form around audio without discernible differences, as if the microphone would collect additional details, not relevant to the classification task. These results further discouraged us from using MFCC coefficients in our investigation.

We also experimented with K-means clustering, using the primary and secondary labels as a reference for the number of clusters, accounting for an extra cluster given by 'unlabelled'.

Overall, tweaking the cluster parameters was effective on a case by case basis, but the sensitivity to changes in the recording setup, especially across different origins for the data makes it an ineffective tool for the segmentation of the whole dataset, especially considering the performance on unlabelled data. We concluded that clustering was not a viable approach for isolating the bird calls within full recordings, given its limited effectiveness and high sensitivity to recording variations. We moved to explore alternative ways of extracting the animal calls, detailing our methods in the following sections.

## Rating-Based Filtering

We first leveraged the rating system available in the Xeno-Canto dataset:

- Analyzed the distribution of ratings, finding most clips rated above 3.5
- Identified that filtering out low-rated samples would affect only 0.19% of the data
- Found two species (Jaguar '41970' and Spotted Foam-nest Frog '126247') that would be lost if strictly filtered by rating
- Implemented a preservation strategy by retaining the top 5 highest-rated examples of these at-risk species

This approach ensured we maintained representation across all 206 taxonomy labels while improving overall data quality.

## YAMNet Audio Classification

Since rating-based filtering only affected a small portion of our dataset, and since we wanted to better navigate the variety of nature of the spliced audio clips, we identified Google's Yamnet pre-trained model for audio classification. It identifies the main category of sound in a clip out of a comprehensive list of 521 event classes. We set up the YAMNet filtering with the following steps:

- Split all recordings into standardized 5-second segments
- Used YAMNet to classify each segment with semantic labels (e.g., "Animal", "Bird", "Silence")
- Created a curated list of 27 relevant audio classes to keep, including "Animal", "Wild animals", "Bird vocalization", "Frog", etc. (preserving data quality)
- Created a secondary list of audio classes to remove: "Silence", "Noise", "Vehicle" (preserving data quantity)
- Verified that this filtering preserved representation across species

This two-stage approach allowed us to improve the quality of our data while maintaining the label diversity. The filtered dataset provides cleaner, more relevant audio segments for model training, which should improve classification performance. The standardized 5-second segments also better match our target application, where we'll analyze soundscapes using similar-length segments.

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

# Experiments with Models

We build towards the task of soft classification using models of increasing complexity, before comparing the results with a state-of-the-art solution, which we extend with data augmentation. 

It should be noted that the baseline accuracy of a model which was guessing randomly, given the distribution of the data $\text{P}_\text{correct} = 0.012$

We used a 80-20 train test data split, tracking the validation Cross Entropy Loss and Accuracy metrics at the end of every epoch. For completeness, we show the results for the EfficientNet architecture.

# MelCNN

As an initial experiment, we first studied the performance of a 'deep' CNN, with the following performance metrics: Cross Entropy Loss and Accuracy.  We decided to restrict the input space to a simpler CNN architecture that uses only the Mel Spectrogram. Compared to the more complex architectures tested later, MelCNN was trained for fewer epochs and without data augmentation or label smoothing. The models below explore different values for the label mixing factor m and input filtering strategies. 

Performance across configurations varied, but remained poor, with all accuracies below 0.05, which indicate both underfitting and limitations in model capacity. Moreover, although soft labeling (m=0.8) on the "Animal" (Yamnet) subset slightly improves accuracy, we noticed that training on the full dataset with one-hot encoding (m=1.0) was more consistent in producing low accuracy (~0.0261–0.0298), even when extended to 10 epochs.

Finally, "Animal" filtering generally performs better than "All" despite fewer training samples, likely due to cleaner or more consistent labeling.

<table border="1">
  <tr><th>Data</th><th>Epochs</th><th>Encoding</th><th>Accuracy</th><th>Hash</th></tr>
  <tr><td>All</td><td>10</td><td>1.0</td><td>0.0298</td><td>c580a9c1</td></tr>
  <tr><td>All</td><td>3</td><td>1.0</td><td>0.0261</td><td>c580a9c1</td></tr>
  <tr><td>Animal</td><td>3</td><td>1.0</td><td>0.0397</td><td>c580a9c1</td></tr>
  <tr><td>All</td><td>3</td><td>1.0</td><td>0.0261</td><td>5a6176d1</td></tr>
  <tr><td>Animal</td><td>3</td><td>0.8</td><td>0.0402</td><td>5a6176d1</td></tr>
</table>

Empirically, we decided to stop training after 3 epochs, as we saw little improvements after, probably due to limited model capacity.

After seeing the poor results of this preliminary architecture, we decided to consider a more complex model, trained for a longer time, in order to improve generalization and achieve higher accuracies.

## EfficientNet

After our limited successes with training models from scratch, we opted to try a different approach: filtering the best data and using a pre-trained model.

<!-- EfficientNet Training Accuracy -->
<table border="1">
  <tr><th>Data</th><th>Epochs</th><th>Encoding</th><th>Train Acc.</th><th>Hash</th></tr>
  <tr><td>All</td><td>15</td><td>1.0</td><td>0.770</td><td>c580a9c1</td></tr>
  <tr><td>Animal</td><td>15</td><td>1.0</td><td>0.760</td><td>9964eb55</td></tr>
</table>

As an initial step, we tested the model's ability to fit to the data, observing a high training accuracy (0.77 and 0.76) on both full and filtered ("Animal") datasets with one-hot encoding (m=1.0).

Small gap between All and Animal datasets suggests that both contain learnable patterns, and EfficientNet is robust across them.

<!-- EfficientNet Evaluation Results -->
<table border="1">
  <tr><th>Data</th><th>Epochs</th><th>Encoding</th><th>Accuracy</th><th>Bal Acc</th><th>Hash</th></tr>
  <tr><td>All</td><td>10</td><td>1.0</td><td>0.476</td><td>/</td><td>8b600946</td></tr>
  <tr><td>All</td><td>10</td><td>0.8</td><td>0.451</td><td>/</td><td>8b600946</td></tr>
  <tr><td>Light</td><td>8/10</td><td>1.0</td><td>0.315</td><td>/</td><td>781592e6</td></tr>
  <tr><td>Light</td><td>6/10</td><td>0.8</td><td>0.266</td><td>/</td><td>781592e6</td></tr>
  <tr><td>All</td><td>10</td><td>1.0</td><td>0.4983</td><td>0.400</td><td>0a242441</td></tr>
  <tr><td>All</td><td>6/10</td><td>0.7</td><td>0.343</td><td>0.304</td><td>0a242441</td></tr>
</table>

In the experiments, we noticed signs of overfitting: the model reached a training accuracy of about 0.77, but its evaluation accuracy on the same dataset (with hard labels, m=1.0) was significantly lower at 0.476. This gap suggests the model may be learning patterns that don't generalize well, even on familiar data.

We also tested soft labeling by adjusting the label confidence to m=0.8. Interestingly, this slightly decreased performance, dropping the evaluation accuracy to 0.451. This indicates that, at least in our setup, soft labeling may hurt performance—potentially because it introduces uncertainty or emphasizes less confident predictions, which could confuse the model.

To reduce noise in the dataset, we applied a filtering step using Light Yamnet, which resulted in a noticeable drop in accuracy compared to using the full dataset. This suggests that while filtering may reduce noise, it can also remove useful diversity that helps the model generalize better.

When we combined filtering with soft labeling, performance degraded even further. This aligns with the idea that soft supervision might not be effective when the dataset is already sparse or contains weak signals—adding uncertainty in such cases can be more harmful than helpful.

Overall, the best performance (validation accuracy = 0.4983) was achieved when using the full dataset with hard labels (m=1.0). This supports the conclusion that, for our setup, full supervision with confident labels is the most effective approach.

Finally, we found that very soft labels (m=0.7) combined with early stopping (after 6 out of 10 training epochs) led to a significant drop in both accuracy (down to 0.343) and balanced accuracy (0.304). This further highlights how sensitive the model is to supervision quality and training strategy.

Moreover, to better understand the evolution of performance, we plot the evolution of loss a run of efficientNet. 

![](img/efficient_loss_accuracy_plot.png)

We notice that overfitting is a true concern training with this kind of data, especially considering that the model is of size much larger than the total number of training samples.

## Takeaways of the experiments

Experimenting with different model architectures and validation methods, we tried to account for the imbalance in the training data, with varying degrees of success.

In our training experiments, we considered $\text{m}\in \{0.7, 0.8, 1.0\}$, but we always observed better results with $\text{m}=1$, that is, one-hot encoding.

A simple model like the MelCNN is not able to capture the full image of the data, which is particularly clear when observing the much higher accuracy score of the EfficientNet variation.

Given the limited amounts of data, overfitting is a real concern, which warrants the use of more sophisticated techniques to avoid it, notably Balanced Accuracy and Cross-Fold validation.

# Comparison to EfficientNet B0

In line with the our observations on the exploratory models, we address shortcomings and limitations by applying some key changes and producing a new model, which is more aligned with State-Of-The-Art solutions. 

To address class imbalances, since some classes appear far more frequently than others, we resort to Balanced Accuracy. This metric computes the average of recall (true positive rate) for each class, ensuring that all classes contribute equally to the final score, regardless of their frequency in the dataset.

We use the following key metrics are used to evaluate model performance: Binary Cross Entropy Loss, Balanced Accuracy, and AUC Score. Binary Cross Entropy Loss is a standard loss function for binary classification tasks that measures the distance between predicted probabilities and actual binary labels. It penalizes incorrect predictions with high confidence more heavily, encouraging the model to output calibrated probabilities. AUC Score, or Area Under the Receiver Operating Characteristic Curve, evaluates the model's ability to distinguish between classes across all possible thresholds, offering a threshold-independent view of performance.

Loss quantifies how close the predicted probability vector is to the target distribution, usually a one-hot vector for classification tasks. A lower loss indicates that the model’s predicted probabilities are better aligned with the true labels. Accuracy, on the other hand, measures how often the class with the highest predicted probability matches the actual label.

We use both metrics because loss provides a continuous signal that reflects model confidence and can guide training, even when predictions are incorrect. Accuracy, in contrast, is discrete and only measures final decision correctness.

Finally, we use cross validation to reduce the risk of overfitting on the data during the training phase. This is also relevant in training the final, complete model on the whole dataset, as a 100-0 split would  lack a reliable accuracy metric to decide when to stop the training.

## Results

Observing the evolution of loss throughout training, we observe a similar phenomenon as in the previous implementation of efficientNet: the model is slow to generalize, despite the advantages of the new configuration, together with the availability of the full training dataset.

![](img/efficientb_loss_kfold.png)

On a second note, validation accuracy falls within the previous results, though it is higher as a result of the enlarged training data. This can be attributed to K-fold cross-validation training on the whole dataset, as opposed to only 80% of it.

![](img/efficientb_acc_kfold.png.png)

To account for the limitations of cross-validation, we also trained a model using the same regime for 90% of the dataset, validating at the end on 10% of the dataset. Plotting the confusion matrix, there is no clear 'bias' between taxonomy groups: the model performs uniformly over different labels.

![](img/confusion_matrix_efficientb_train.png)

For reference, the new model performed with 0.60 accuracy when using the whole train dataset (hash 596720). 

## Semi-supervised learning

Given the vast amounts of unlabelled audio recordings that are also present in the dataset, we attempt to use semi-supervised learning in improving the model: we first add labels the soundscape recordings using our best performing model, before continuing to train the model on these labels. We hope that the additional training may provide the model with more information on the distribution of the dataset, thus increasing model effectiveness.

As an additional note, we also ran the "naive" efficientNet implementation, comparing the labelling of the two using a confusion matrix: we observe a high number of vertical and horizontal lines. This is consistent with our expectations: as the old model is biased and somewhat overfitting, we can identify in vertical lines labels that are clumped by the naive implementation but differentiated in the new model, and the opposite in the horizontal lines: uncertain labellings which belong to a single class according to the newer model, with successively poorer performance in later epochs.

![](img/confusion_matrix_efficient_models.png)

Plotting the training performance for the second stage of training, we observed coarser evolution, which could be attributed to overfitting. Moreover, through variations of the model, we noticed that the performance generally peaked in the second fold.

![](img/semisuper_learning_val_acc.png)

# Kaggle Scoreboard

To compare the final performance of the models, we use Kaggle's hidden test feature. Unfortunately, none of the variations on the new model seemed to improve the performance much. We attribute this to the dataset, which has shown itself to be extremely unstable to changes both in training regime and in data augmentation.

Accuracy with Kaggle hidden dataset
- Baseline : 0.781
- Data Augmentation in Misrepresented : 0.500
- Additional Learning, Fold 1: 0.725
- Additional Learning Misrepresented : 0.728
- Additional Learning, Fold 4: 

# Evaluation of Methodology



# Conclusion



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