{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ed8c800a-d0f2-4973-bdf1-1ed4624346ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "def save_model(model, path):\n",
    "    torch.save(model.state_dict(), path)\n",
    "    print(f\"Model saved to {path}\")\n",
    "\n",
    "def load_model(model, path):\n",
    "    model.load_state_dict(torch.load(path))\n",
    "    print(f\"Model loaded from {path}\")\n",
    "    return model\n",
    "\n",
    "def generate_text(model, tokenizer, text, max_length=50):\n",
    "    model.eval()\n",
    "    input_ids = tokenizer.encode(text, return_tensors=\"pt\").cuda() if torch.cuda.is_available() else tokenizer.encode(text, return_tensors=\"pt\")\n",
    "    outputs = model.generate(input_ids=input_ids, max_length=max_length, num_beams=4, early_stopping=True)\n",
    "    return tokenizer.decode(outputs[0], skip_special_tokens=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9efd53dd-f166-482b-8988-eb43fe41b220",
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
