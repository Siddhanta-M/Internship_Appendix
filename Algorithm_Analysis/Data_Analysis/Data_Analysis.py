# Import required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tools for generating and displaying confusion matrices.
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ---------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------

# Maximum allowable localisation error.
#
# Only detections within this distance (metres) are
# considered valid matches.
#
# Typical values:
#   0.25 m
#   0.50 m
#   0.75 m
#   1.00 m
err = 0.25


# Object classes used during evaluation.
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


# Initialise cumulative confusion matrix.
cm_tot = np.zeros(
    (len(classes), len(classes)),
    dtype=int
)



# ---------------------------------------------------------
# Evaluate object configurations 1–8
# ---------------------------------------------------------

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
    # Compare every observation to the ground truth
    # -----------------------------------------------------

    for j in range(len(df)):

        pred_det = df.iloc[j]["detections"]


        # Lists used to build the confusion matrix.
        true_labels = []
        pred_labels = []


        # Tracks predictions that have already been matched.
        used_predictions = set()



        # -------------------------------------------------
        # Match each ground-truth object
        # -------------------------------------------------

        for t in true_det:

            best_idx = None
            best_dist = np.inf
            best_class = None


            # Search through every predicted object.
            for i, p in enumerate(pred_det):

                # Skip predictions already assigned to another
                # ground-truth object.
                if i in used_predictions:
                    continue


                # Compute Euclidean distance between the
                # ground-truth object and prediction.
                dist = np.hypot(
                    t["x"] - p["x"],
                    t["y"] - p["y"]
                )


                # Keep the closest prediction regardless
                # of object class.
                if dist < best_dist:

                    best_dist = dist
                    best_idx = i

                    # Save the predicted class.
                    best_class = p["class_name"]


            # Store the true object class.
            true_labels.append(
                t["class_name"]
            )


            # -------------------------------------------------
            # Determine whether the match is valid
            # -------------------------------------------------

            # If the closest prediction lies within the
            # localisation threshold, use its predicted class.
            #
            # This allows classification errors to appear
            # naturally in the confusion matrix.
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
            # Record as a missed detection.
            else:

                pred_labels.append(
                    "background"
                )



        # -------------------------------------------------
        # Count false positives
        # -------------------------------------------------

        # Any prediction that was never matched to a
        # ground-truth object is considered a false positive.
        for i, p in enumerate(pred_det):

            if i not in used_predictions:

                true_labels.append(
                    "background"
                )

                pred_labels.append(
                    p["class_name"]
                )



        # -------------------------------------------------
        # Compute confusion matrix for this observation
        # -------------------------------------------------

        cm = confusion_matrix(
            true_labels,
            pred_labels,
            labels=classes
        )


        # Accumulate results.
        cm_tot += cm



# ---------------------------------------------------------
# Evaluate background-only configuration
# ---------------------------------------------------------

# Configuration 0 contains scenes where no objects
# should be detected.
df = pd.read_json(
    "observations_config_0.jsonl",
    lines=True
)


for j in range(len(df)):

    pred_det = df.iloc[j]["detections"]


    # Skip observations without detections.
    if len(pred_det) == 0:
        continue


    true_labels = []
    pred_labels = []


    # Every detected object is a false positive.
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
# Display cumulative confusion matrix
# ---------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_tot,
    display_labels=classes
)


# Plot the confusion matrix.
disp.plot(cmap="Reds")


# Label axes.
plt.xlabel("Predicted")
plt.ylabel("True")


# Add plot title.
plt.title("Confusion Matrix")


# Display the figure.
plt.show()
