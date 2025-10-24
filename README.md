# Efficient-TDP-HeteroSTA
We integrate HeteroSTA into Efficient-TDP ("Timing-Driven Global Placement by Efficient Critical Path Extraction"). It is built upon the popular open-source infrastructure [DREAMPlace](https://github.com/limbo018/DREAMPlace).

## Build with Docker

We highly recommend the use of Docker to enable a smooth environment configuration.

The following steps are borrowed from [DREAMPlace](https://github.com/limbo018/DREAMPlace) repository. We make minor revisions to make it more clear.

1. Get the code and put it in folder `Efficient-TDP-HeteroSTA`.

2. Get the container:

- Option 1: pull from the cloud [limbo018/dreamplace](https://hub.docker.com/r/limbo018/dreamplace).

  ```
  docker pull limbo018/dreamplace:cuda
  ```

- Option 2: build the container.

  ```
  docker build . --file Dockerfile --tag your_name/dreamplace:cuda
  ```

3. Enter bash environment of the container. Replace `limbo018` with your name if option 2 is chosen in the previous step.

- Option 1: Run with GPU on Linux.

  ```
  docker run --gpus 1 -it -v $(pwd):/Efficient-TDP-HeteroSTA limbo018/dreamplace:cuda bash
  ```

- Option 2: Run with CPU on Linux.

  ```
  docker run -it -v $(pwd):/Efficient-TDP-HeteroSTA limbo018/dreamplace:cuda bash
  ```

4. ` cd /Efficient-TDP-HeteroSTA`.

5. Build.

   ```
   mkdir build
   cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=../install -DPython_EXECUTABLE=$(which python)
   make
   make install
   ```

6. Get benchmarks: download the cases here: [Google Drive link for iccad2015.hs ](https://drive.google.com/file/d/1yKABk8yTXU9PCLxjWzW02Ey6Q6IIxJ03/view?usp=drive_link). Unzip the package and put it in the following directory:

   ```
   install/benchmarks/iccad2015.hs
   ```


## Test

Run our method integrated with HeteroSTA on case superblue1 of ICCAD2015 timing-driven placement contest:

```
python dreamplace/Placer.py test/iccad2015.pin2pin/$superblue1.json
```

If you wish to switch to OpenTimer, follow these steps:
- Download the simplified cases here: [Google Drive link for iccad2015.ot ](https://drive.google.com/file/d/1xeauwLR9lOxnYvsK2JGPSY0INQh8VuE4/view?usp=drive_link). Unzip the package and put it in the following directory:

   ```
   install/benchmarks/iccad2015.ot
   ```

- In the JSON file, Change the input path from "iccad2015.hs" to "iccad2015.ot" and set the "timer_engine" option from "heterosta" to "opentimer". 
  
  
- Execute the command as shown above.

## Evaluation
The iccad2015 contest's official evaluation kit can be found at [Google Drive link for evaluation kit](https://drive.google.com/file/d/1VI9S27KQOMoqcHIN29wTYRr-4NxNjtKS/view?usp=sharing).

- for iccad2015 contest's official evaluation kit download the cases here: [Google Drive link for iccad2015 ](https://drive.google.com/file/d/1PnL0w8_Yh5g_xOC5HdkSbjJsTgg85ZH5/view?usp=sharing)

## Non-deterministic bug fixes
The non-deterministic bug in the original Efficient-TDP are caused by two reasons: 
- Applying atomicAdd operations to floating point numbers in "dreamplace/ops/pin2pin_attraction/src/pin2pin_attraction_cuda_kernel.cu"
- Dynamic path insertion by different threads in "thirdparty/OpenTimer/ot/timer/path.cpp"
We have fixed these bugs, and you may refer to the files for the specific implementation details.

## Clarification on the Cases
The differences between the cases stem from the SDC files: 
- The iccad2015 contest's official evaluation kit uses the official datasets 
- OpenTimer utilizes the datasets with simplified SDC commands, due to its limited sdc command support.
- HeteroSTA utilizes datasets derived from the official ones, with double-clock and clock propagation SDC commands added, to eliminate discrepancies arising from the oversight of half-clock periods by both the official kit and OpenTimer.



