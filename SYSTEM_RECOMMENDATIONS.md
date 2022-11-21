[WIP]
## List of recommended flags depending on available GPU and CPU mermory (VRAM/RAM) 

In order to achieve the fastest results for the FastSurfer segmentation we provide a list of recommended flags for the batch size and where to run the view-aggregation.
The following list recommends values depending on resolution, available GPU and CPU memory (RAM) and are based on our benchmark times found [here]().

To check your memory follow these steps:
- Windows: 
  - CPU: Open cmd.exe and run the following command: ```systeminfo | findstr /C:"Total Physical Memory"```
  - GPU: Follow [this](https://www.thewindowsclub.com/how-to-check-how-much-video-ram-vram-you-have-in-windows-11-10) tutorial
- Ubuntu: 
  - CPU: ```free -tg```
  - GPU: ```lspci -v | less```

#### 1.0 resolution
| CPU\GPU |     1 |     2 |     4 |      8 |     16 |    32 |
|:--------|------:|------:|------:|-------:|-------:|------:|
| **2**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **4**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **6**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **8**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **12**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **16**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **24**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **32**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |

#
#### 0.9 resolution
| CPU\GPU |     1 |     2 |     4 |      8 |     16 |    32 |
|:--------|------:|------:|------:|-------:|-------:|------:|
| **2**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **4**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **6**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **8**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **12**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **16**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **24**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **32**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |

#
#### 0.8 resolution
| CPU\GPU |      1 |     2 |     4 |      8 |     16 |    32 |
|:--------|-------:|------:|------:|-------:|-------:|------:|
| **2**       |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **4**       |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **6**       |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **8**       |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **12**      |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **16**      |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **24**      |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **32**      |  1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |

#
#### 0.7 resolution
| CPU\GPU |     1 |     2 |     4 |      8 |     16 |    32 |
|:--------|------:|------:|------:|-------:|-------:|------:|
| **2**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **4**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **6**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **8**       | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **12**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **16**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **24**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |
| **32**      | 1 cpu | 1 cpu | 1 cpu |  1 cpu |  1 cpu | 1 cpu |