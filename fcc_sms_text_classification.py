{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/KonstantinosGkoutis/email-app-/blob/main/fcc_sms_text_classification.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8RZOuS9LWQvv"
      },
      "outputs": [],
      "source": [
        "# Import necessary libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.metrics import accuracy_score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lMHwYXHXCar3"
      },
      "outputs": [],
      "source": [
        "# Load the dataset (assuming CSV format with 'message' and 'label' columns)\n",
        "train_data = pd.read_csv('sms_train.csv')  # Training data\n",
        "test_data = pd.read_csv('sms_test.csv')    # Test data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "g_h508FEClxO"
      },
      "outputs": [],
      "source": [
        "# Convert labels to binary values (ham=0, spam=1)\n",
        "train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})\n",
        "test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zOMKywn4zReN"
      },
      "outputs": [],
      "source": [
        "# Separate features and labels\n",
        "X_train, y_train = train_data['message'], train_data['label']\n",
        "X_test, y_test = test_data['message'], test_data['label']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "J9tD9yACG6M9"
      },
      "outputs": [],
      "source": [
        "# Build a text classification pipeline using TF-IDF and Logistic Regression\n",
        "model = Pipeline([\n",
        "    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, ngram_range=(1, 2))),\n",
        "    ('classifier', LogisticRegression(solver='liblinear'))\n",
        "])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dxotov85SjsC"
      },
      "outputs": [],
      "source": [
        "# Train the model\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Evaluate model accuracy on test data\n",
        "y_pred = model.predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(f'Model Accuracy: {accuracy:.2%}')\n",
        "\n",
        "# Define the predict_message function\n",
        "def predict_message(message):\n",
        "    prediction_prob = model.predict_proba([message])[0][1]  # Probability of being spam\n",
        "    prediction_label = \"spam\" if prediction_prob > 0.5 else \"ham\"\n",
        "    return [float(prediction_prob), prediction_label]\n",
        "\n",
        "# Testing the function\n",
        "test_message = \"Congratulations! You have won a free gift. Call now!\"\n",
        "print(predict_message(test_message))\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "fcc_sms_text_classification.ipynb",
      "private_outputs": true,
      "provenance": [],
      "toc_visible": true,
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {}
  },
  "nbformat": 4,
  "nbformat_minor": 0
}