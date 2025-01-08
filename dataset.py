{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7375ff99-adb7-4937-929f-a9fbf3bf5f91",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "class T5Dataset(Dataset):\n",
    "    def __init__(self, source_texts, target_texts, tokenizer, max_length=128):\n",
    "        self.source_texts = source_texts\n",
    "        self.target_texts = target_texts\n",
    "        self.tokenizer = tokenizer\n",
    "        self.max_length = max_length\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.source_texts)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        # Tokenize the source text\n",
    "        source = self.tokenizer(\n",
    "            self.source_texts[idx],\n",
    "            padding=\"max_length\",\n",
    "            truncation=True,\n",
    "            max_length=self.max_length,\n",
    "            return_tensors=\"pt\",\n",
    "        )\n",
    "        # Tokenize the target text\n",
    "        target = self.tokenizer(\n",
    "            self.target_texts[idx],\n",
    "            padding=\"max_length\",\n",
    "            truncation=True,\n",
    "            max_length=self.max_length,\n",
    "            return_tensors=\"pt\",\n",
    "        )\n",
    "        \n",
    "        return {\n",
    "            \"source_ids\": source[\"input_ids\"].squeeze(),\n",
    "            \"source_mask\": source[\"attention_mask\"].squeeze(),\n",
    "            \"target_ids\": target[\"input_ids\"].squeeze(),\n",
    "        }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d9452ad-cc9c-4aed-bce1-df176090bd67",
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
