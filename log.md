## General TODOs
- [ ] 2 GPU eval won't return.
- [ ] 2 GPU eval didn't show progress bar at all. Should've appeared on the main process.
- [x] With model_id, can't omit model_dir for llava1.5, should be able to. 

## Discussion
### Eval
- The code is unnecessarily convoluted. On high level, eval is a simple process: 
  - Download and process data. 
  - Choose the prompt. 
  - Download model and do inference. 
  - Process results. 
- [ ] Focus on the process and get `VQA-v2` working as the first step. 
  - [ ] Don't understand the purpose of index dataset. Seems to be important.
  - Even loading model is crazily convoluted just because they treat huggingface as S3. Oh my god. 

## Conclusion
- Adapt to this codebase is waste of time. Learn to do eval from scratch using llava 1.6. 
- [AI2 olmo](https://huggingface.co/allenai/OLMo-7B) UX is way better. Use that as template. Saving versions under different branch and each branch is self-contained model to be one line loaded. 
- The goal is still the same. Master the eval first, then dive into pretraining.
  - I skipped rigorous eval of LLM. There is no escape since VLM would still be compared to LLM on pure language capability. 

## What to learn from this repo
- Choice of eval set. 
- Prompt for each eval set. 
- The purpose of index dataset.
- Learn to build dataset and dataloader from source.  
  - You want to expand to new dataset as soon as it's released and don't rely on the middle man service. 
  - The purpose of huggingface is distribution. Think of it as AI instagram or youtube. It's great to showcase your work and facilitate idea cross pollination. 