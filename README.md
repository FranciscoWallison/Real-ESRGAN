## Dockerfile for The Real-ESRGAN

For the original README, refer to [the original repo](https://github.com/xinntao/Real-ESRGAN/blob/master/README.md).

I didn't find a good Dockerfile, so I got one from [adryanfrois](https://github.com/adryanfrois/GFPGAN_docker) and updated it. For example, the old version used the CUDA 10 image, but my RTX 3090 requires CUDA 11.

````
python3 inference_realesrgan.py -n RealESRGAN_x4plus.pth -i inputs --fp32
````

![teste1](https://user-images.githubusercontent.com/19413241/223596149-57d5fc83-f88f-487a-b9ae-db0c326a3f5f.png)


![teste2](https://user-images.githubusercontent.com/19413241/223596160-ffbb89b2-e2f9-47c2-9f96-a09e7746e0d3.png)
