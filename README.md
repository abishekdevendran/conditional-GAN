# conditional GAN test:

- Input: main.png (ignored first 5 cols of poses)
- Processed into `sprites/inputs` and `sprites/outputs` (cols 6 and 26) using `process.py`
- Quality of output images during training can be observed in `sprites/epocs` for every 100 epocs
- Observations: High quality output images, but adherence to style is not perfect due to lack of training data.