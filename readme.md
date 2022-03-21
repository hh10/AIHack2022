# Submission for AI2022 Hackathon project: Predicting microfluidic drop interactions

The script implements all 3 tasks in the challenge (can find details and data [here](https://drive.google.com/drive/folders/17mbPZiTRdUdJeBbb4WjiyEcMCJNqIzBQ)):
* Implements a Sequence-to-Sequence architecture for droplet trajectory predictions based on 100 timestamps of droplet trajectories.
* Adds a classification head in the architecture that uses the learned hidden state of the Sequence-to-Sequence architecture to predict whether the droplets will coalesce or not.
* Feeds the predictions of the decoder back to it (not to the encoder) to help it build longer and smoother trajectories

Sample results, i.e., 472 frames after a 100 frames input (left: actual trajectory, right: predicted trajectory):
![Ground truth trajectory](outputs/test_ground_truth.gif)
![Predicted trajectory](outputs/test_prediction.gif)
