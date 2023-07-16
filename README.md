## Dockerfile for The Real-ESRGAN

For the original README, refer to [the original repo](https://github.com/xinntao/Real-ESRGAN/blob/master/README.md).

I didn't find a good Dockerfile, so I got one from [adryanfrois](https://github.com/adryanfrois/GFPGAN_docker) and updated it. For example, the old version used the CUDA 10 image, but my RTX 3090 requires CUDA 11.

````
python3 inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --fp32
python3 inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs --fp32
python3 inference_realesrgan.py -n realesr-general-x4v3 -i inputs --fp32


python3 inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance  --fp32
python3 inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs --face_enhance --fp32
python3 inference_realesrgan.py -n realesr-general-x4v3 -i inputs --face_enhance  --fp32


python3 inference_realesrgan.py -n 4x_SmolFace_200k.pth -i inputs --fp32
python3 inference_realesrgan.py -n Art_4xFSMangaV2.pth -i inputs --fp32


cd MM-RealSR && python3 inference_mmrealsr.py -opt options/MMRealSRNet_x4.yml --im_path input/ --res_path output/ --model_path experiments/MMRealSRGAN.pth


python3 inference_mmrealsr.py -opt options/MMRealSRNet_x4.yml --im_path input/ --res_path output/ --model_path experiments/4x_ArtStation1337_v2.pth



python3 transer_RRDB_models.py
````

### Exemplos

![image](https://user-images.githubusercontent.com/19413241/226325893-c0a1d3ef-0bde-421c-9ec6-4df5b2547281.png)


![teste1](https://user-images.githubusercontent.com/19413241/223596149-57d5fc83-f88f-487a-b9ae-db0c326a3f5f.png)


![teste2](https://user-images.githubusercontent.com/19413241/223596160-ffbb89b2-e2f9-47c2-9f96-a09e7746e0d3.png)


You can build it with:

    $ ./docker_build.sh

To upscale pictures from ~/Pictures/inputs run with:

    $ ./docker_run.sh \
    python3 inference_realesrgan.py \
    -i /app/inputs -o /app/results \
    -n RealESRGAN_x4plus_anime_6B

To upscale videos from the same folder run with:

    $ ./docker_run.sh \
    python3 inference_realesrgan_video.py \
    -i /app/inputs/anime.mp4 -o /app/results/anime_upscaled.mp4 \

Don't forget to edit [docker_run.sh](docker_run.sh) to map your input and result folders.

