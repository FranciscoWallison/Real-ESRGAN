## Dockerfile for The Real-ESRGAN

For the original README, refer to [the original repo](https://github.com/xinntao/Real-ESRGAN/blob/master/README.md).

I didn't find a good Dockerfile, so I got one from [adryanfrois](https://github.com/adryanfrois/GFPGAN_docker) and updated upon it.

You can build it with:

    $ ./build.sh

Then run with:

    $ ./docker_run.sh \
    python3 inference_realesrgan.py \
    -i /app/inputs -o /app/results \
    -n RealESRGAN_x4plus_anime_6B

Don't forget to edit [docker_run.sh](docker_run.sh) to map your input and result folders.

