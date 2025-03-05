# Instruct GPT (Training language models to follow instructions with human feedback)
* In this paper, we show an avenue for
aligning language models with user intent on a wide range of tasks by fine-tuning
with human feedback
* Starting with a set of labeler-written prompts and prompts
submitted through the OpenAI API, we collect a dataset of labeler demonstrations
of the desired model behavior, which we use to fine-tune GPT-3 using supervised
learning
* We then collect a dataset of rankings of model outputs, which we use to
further fine-tune this supervised model using reinforcement learning from human
feedback
* Even though InstructGPT still makes simple mistakes, our results
show that fine-tuning with human feedback is a promising direction for aligning
language models with human intent
---
* —predicting the next token on a webpage from the internet—is
different from the objective “follow the user’s instructions helpfully and safely”
the language modeling objective is misaligned
* ![image](https://github.com/user-attachments/assets/99a8c676-89ca-42cb-a5bd-fb42925069f1)
* During RLHF fine-tuning, we observe performance regressions compared
to GPT-3 on certain public NLP datasets, notably SQuAD (Rajpurkar et al., 2018), DROP (Dua et al.,
2019), HellaSwag (Zellers et al., 2019), and WMT 2015 French to English translation (Bojar et al., 2015) -- “alignment tax”
* ![image](https://github.com/user-attachments/assets/baf1865e-c252-4b8c-860e-8d2b8e7bbc16)
* ![image](https://github.com/user-attachments/assets/ca4bb471-a60a-4502-b3e2-f952b0d19360)
* ![image](https://github.com/user-attachments/assets/ffbdb266-696f-4553-a6ad-6f5f4fa2fedf)

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model
* ![image](https://github.com/user-attachments/assets/972f5a0f-26b7-4aad-bffa-1fb0df60b346)
* ![image](https://github.com/user-attachments/assets/0a157919-f2bb-41ba-aa26-f0d54003b50c)
* ![image](https://github.com/user-attachments/assets/3e2739c6-8b5b-46a7-b0d2-9c41bd94a2d1)
