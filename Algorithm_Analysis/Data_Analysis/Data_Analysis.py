# Import required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tools for creating and displaying confusion matrices.
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ---------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------

# Maximum localisation error (metres).
#
# A predicted object is considered a valid match only if
# it is within this distance of the corresponding
# ground-truth object.
#
# Example values:
#   0.25
#   0.50
#   0.75
#   1.00
err = 1


# List of evaluated object classes.
#
# "background" represents:
#   - missed detections
#   - false detections
classes = [
    "chair",
    "box",
    "desk",
    "door_frame",
    "background"
]


# Initialise an empty confusion matrix.
#
# Rows:
#     Ground-truth classes
#
# Columns:
#     Predicted classes
cm_tot = np.zeros(
    (len(classes), len(classes)),
    dtype=int
)



# ---------------------------------------------------------
# Evaluate object configurations
# ---------------------------------------------------------

# Process configurations 1 through 8.
for file_num in range(1, 9):


    filename = (
        f"observations_config_{file_num}.jsonl"
    )


    # Load observations from the JSON Lines file.
    df = pd.read_json(
        filename,
        lines=True
    )


    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    # The first observation is treated as the correct
    # object configuration.
    true_det = df.iloc[0]["detections"]


    # -----------------------------------------------------
    # Compare every observation
    # -----------------------------------------------------

    for j in range(len(df)):

        # Current predicted detections.
        pred_det = df.iloc[j]["detections"]


        # Lists passed into sklearn.
        true_labels = []
        pred_labels = []


        # Keep track of predictions already assigned.
        used_predictions = set()



        # -------------------------------------------------
        # Match each ground-truth object
        # -------------------------------------------------

        for t in true_det:

            best_idx = None
            best_dist = np.inf
            best_class = None


            # Search for the nearest prediction.
            for i, p in enumerate(pred_det):

                # Ignore predictions already matched.
                if i in used_predictions:
                    continue


                # Calculate Euclidean distance between
                # the true object and prediction.
                dist = np.hypot(
                    t["x"] - p["x"],
                    t["y"] - p["y"]
                )


                # Keep the closest prediction.
                if dist < best_dist:

                    best_dist = dist
                    best_idx = i
                    best_class = p["class_name"]


            # Record the true object class.
            true_labels.append(
                t["class_name"]
            )


            # -------------------------------------------------
            # Accept or reject the match
            # -------------------------------------------------

            # Prediction must lie within the localisation
            # threshold.
            if (
                best_idx is not None
                and
                best_dist <= err
            ):

                pred_labels.append(
                    best_class
                )

                used_predictions.add(best_idx)


            # No nearby prediction.
            #
            # Count as a missed detection.
            else:

                pred_labels.append(
                    "background"
                )



        # -------------------------------------------------
        # Remaining predictions become false positives
        # -------------------------------------------------

        for i, p in enumerate(pred_det):

            if i not in used_predictions:

                true_labels.append(
                    "background"
                )

                pred_labels.append(
                    p["class_name"]
                )



        # -------------------------------------------------
        # Compute confusion matrix
        # -------------------------------------------------

        cm = confusion_matrix(
            true_labels,
            pred_labels,
            labels=classes
        )


        # Accumulate into the overall confusion matrix.
        cm_tot += cm



# ---------------------------------------------------------
# Evaluate background-only scenes
# ---------------------------------------------------------

# Configuration 0 contains scenes without objects.
#
# Every detection is therefore a false positive.
df = pd.read_json(
    "observations_config_0.jsonl",
    lines=True
)


for j in range(len(df)):

    pred_det = df.iloc[j]["detections"]


    # Skip empty observations.
    if len(pred_det) == 0:
        continue


    true_labels = []
    pred_labels = []


    # Every detected object is classified as
    # a background false positive.
    for p in pred_det:

        true_labels.append(
            "background"
        )

        pred_labels.append(
            p["class_name"]
        )


    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=classes
    )


    cm_tot += cm



# ---------------------------------------------------------
# Normalise confusion matrix
# ---------------------------------------------------------

# Convert integer counts into floating-point values.
cm_norm = cm_tot.astype(float)


# Divide every row by the total number of samples
# belonging to that true class.
#
# After normalisation:
#
# Every row sums to approximately 1.
#
# This allows the confusion matrix to be interpreted
# as percentages rather than absolute counts.
cm_norm = (
    cm_norm
    /
    cm_tot.sum(axis=1, keepdims=True)
)


# If a row contains no samples, division by zero
# produces NaN values.
#
# Replace NaNs with zeros.
cm_norm = np.nan_to_num(
    cm_norm
)



# ---------------------------------------------------------
# Display normalised confusion matrix
# ---------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=classes
)


# Plot using a red colour map.
disp.plot(cmap="Reds")


# Label axes.
plt.xlabel("Predicted")
plt.ylabel("True")


# Add title.
plt.title("Confusion Matrix")


# Display the figure.
plt.show()
