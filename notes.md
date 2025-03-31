# Global birdsong embeddings
enable superior transfer learning
for bioacoustic classification

`many classes (such as rare and endangered species,
many non-bird taxa, and call-type) lack enough data to train a robust model from scratch.`

It can be useful to use feature embeddings extracted from audio classification models to identify
bioacoustic classes other than the ones these models were originally trained.

Feature embeddings are vectors obtained from some intermediate layer of a trained machine learning model

The results of this study indicate that high-quality feature embeddings from large-scale acoustic bird classifiers can be harnessed for **few-shot transfer learning**

(MAML (Model-Agnostic Meta-Learning) optimize for rapid adaptation by explicitly training on a distribution of tasks.)

songbirds can display local variations (also called dialects) in their song patterns, which may lead to slight differences in note sequences 

Re-using the feature embeddings from
a pre-trained model allows learning the new task efficiently, so long as the embeddings are sufficiently relevant.

(SOTA models: 
- AudioMAE (a self-supervised transformer), 
- YAMNet,
- PSLA (more recent convolutional models))

(They start from different models that each labels the same spectrogram. Labels become a single vector, and a fully connected label + softmax produces the final classification. This technique is called a **linear probe**)

When an example is shorter than the model’s window size, we apply
centered zero-padding to obtain the target length. When a model’s window size is shorter than a target example,we frame the audio according to the model’s window size, create an embedding for each frame, and then average the results.

(k-fold validation is used)

linear probes of pretrained embeddings is a surprisingly successful strategy for few-shot learning (even as k value varies)

# Avoiding Information Bottlenecks
If the hidden layer had fewer units than the embedding size, it might act as a bottleneck, forcing information compression and potential loss of useful features.

A larger hidden layer permits redundancy and allows features to be remapped in a more discriminative way.

(Running an ablation on embedding size means conducting an experiment where the dimensionality of the embedding is systematically varied to analyze its impact on performance.)

Training and inference with small models over fixed embeddings are much faster than training entirely new models: Training a high-quality classifier from scratch can take many days of GPU time, but training small linear classifiers over fixed embeddings, can take less than a minute to train on a modern workstation. This allows fast experimentation with different analysis techniques and quickly iterating with human-in-the-loop active learning techniques.

# ROC-AUC Score with Log-Odds Scaling

The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various classification thresholds.

The ROC curve essentially shows the trade-off between correctly identifying positive instances and incorrectly classifying negative instances.

The AUC score quantifies the overall ability of the classifier to distinguish between the positive and negative classes. An AUC of 0.5 means the model has no discriminatory ability (random guessing), and an AUC of 1.0 indicates perfect classification.

ROC-AUC gives a single performance metric that summarizes the model's ability to discriminate between positive and negative classes.

The AUC score is calculated by finding the area under the ROC curve. The higher the AUC, the better the model is at distinguishing between the two classes.

# Fine Tuning vs Feature Extraction

- Fine-tuning: the model is updated, which means computation cost is high and so it adaptability
- Feature extraction: the backbone is frozen, computation cost is low, hence also adaptability

# Audio Resampling

Sample Rate: The number of samples taken per second in an audio signal. 

Aliasing: The phenomenon where high-frequency components of a signal are misrepresented at lower frequencies when resampling. This is why a low-pass filter is applied to prevent frequencies higher than half the new sample rate from becoming indistinguishable.

# Confusion Rate

Proportion of incorrect predictions: FP + FN / (...) = the rate of error.