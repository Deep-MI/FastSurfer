[WIP]
## List of minimum requirements and recommended flags 

In order to run the FastSurfer segmentation we provide a list of minimum requirements for 1.0, 0.8, 0.7mm resolution.
The following list recommends values for the **--run_viewagg_on** flag depending on resolution, available GPU/CPU memory and are based on our benchmark times found [here]().

To check your memory follow these steps:
- Windows: 
  - CPU: Open cmd.exe and run the following command: ```systeminfo | findstr /C:"Total Physical Memory"```
  - GPU: Follow [these](https://www.thewindowsclub.com/how-to-check-how-much-video-ram-vram-you-have-in-windows-11-10) steps
- Ubuntu: 
  - CPU: ```free -tg```
  - GPU: Follog [these](https://www.cyberciti.biz/faq/howto-find-linux-vga-video-card-ram/) steps


### Minimum Requirements:

|       | --run_viewagg_on | Min GPU (in GB) | Min CPU (in GB) |
|:------|------------------|----------------:|----------------:|
| 1mm   | gpu              |               5 |               5 |
| 1mm   | cpu              |               2 |               7 |
| 0.8mm | gpu              |               8 |               6 |
| 0.8mm | cpu              |               3 |               9 |
| 0.7mm | gpu              |               8 |               6 |
| 0.7mm | cpu              |               3 |               9 |



### Recommended flags

We recommend running the view aggregation on the GPU as it is generally faster, as long as you have enough VRAM. 
You can change the device by using the flag **--run_viewagg_on** _[cpu/gpu/check]_. By default, _check_ ist being used, which checks 
if your device has enough memory to run it on the GPU.
View the following table to see which device you should be using.

#### 1.0mm resolution

|                                                  | GPU: <2GB                                | 2-4GB | 5GB+ |
|:-------------------------------------------------|------------------------------------------|------:|-----:|
| **CPU: <5GB**                                    | <div style="text-align: right"> - </div> |     - |    - |
| <div style="text-align: right"> **5-6GB** </div> | <div style="text-align: right"> - </div> |     - |  gpu |
| <div style="text-align: right"> **7GB+** </div>  | <div style="text-align: right"> - </div> |   cpu |  gpu |


#### 0.8mm / 0.7mm resolution

|                                                  |                                GPU: <3GB | 3-7GB | 8GB+ |
|:-------------------------------------------------|-----------------------------------------:|------:|-----:|
| **CPU: <6GB**                                    | <div style="text-align: right"> - </div> |     - |    - |
| <div style="text-align: right"> **6-8GB** </div> | <div style="text-align: right"> - </div> |     - |  gpu |
| <div style="text-align: right"> **9GB+** </div>  | <div style="text-align: right"> - </div> |   cpu |  gpu |
