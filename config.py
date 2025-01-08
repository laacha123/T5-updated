{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "336f7ab9-3670-48db-bc7c-7ed060e54159",
   "metadata": {},
   "outputs": [],
   "source": [
    "class T5Config:\n",
    "    def __init__(self):\n",
    "        self.vocab_size = 32128   # Vocabulary size\n",
    "        self.d_model = 512       # Hidden size\n",
    "        self.n_heads = 8         # Number of attention heads\n",
    "        self.d_ff = 2048         # Feedforward layer size\n",
    "        self.num_layers = 6      # Number of encoder/decoder layers\n",
    "        self.dropout = 0.1       # Dropout probability\n",
    "        self.max_seq_length = 128  # Maximum sequence length\n",
    "        self.learning_rate = 5e-4  # Learning rate\n",
    "        self.batch_size = 16       # Batch size\n",
    "        self.epochs = 10           # Number of epochs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c63fe735-21a4-437a-a5f5-176daa7bd5e0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
