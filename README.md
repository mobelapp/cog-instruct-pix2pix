# cog-instruct-pix2pix

[![Replicate](https://replicate.com/pwntus/instruct-pix2pix/badge)](https://replicate.com/pwntus/instruct-pix2pix)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="..." -i image=@...
