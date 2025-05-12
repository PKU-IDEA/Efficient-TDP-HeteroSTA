# DATE'25 "Timing-Driven Global Placement by Efficient Critical Path Extraction"
We provide the implementation of the method proposed in the paper. It is built upon the popular open-source infrastructure [DREAMPlace](https://github.com/limbo018/DREAMPlace).

## Build with Docker

We highly recommend the use of Docker to enable a smooth environment configuration.

The following steps are borrowed from [DREAMPlace](https://github.com/limbo018/DREAMPlace) repository. We make minor revisions to make it more clear.

1. Get the code and put it in folder `DATE25-TDP`.

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
  docker run --gpus 1 -it -v $(pwd):/DATE25-TDP limbo018/dreamplace:cuda bash
  ```

- Option 2: Run with CPU on Linux.

  ```
  docker run -it -v $(pwd):/DREAMPlace limbo018/dreamplace:cuda bash
  ```

4. ` cd /DATE25-TDP`.

5. Build.

   ```
   mkdir build
   cd build
   cmake .. 
   make
   make install
   ```

6. Get benchmarks: download the cases here: https://drive.google.com/file/d/1xeauwLR9lOxnYvsK2JGPSY0INQh8VuE4/view?usp=sharing. Unzip the package and put it in the following directory:

   ```
   install/benchmarks/iccad2015.ot
   ```

## Test

Run our method on case superblue1 of ICCAD2015 timing-driven placement contest:

```
python dreamplace/Placer.py test/iccad2015.pin2pin/$case.json
```

Or you can run all 8 cases by:

```
cd install
./run.sh
```

The iccad2015 contest's official evaluation kit can be found at [Google Drive link](https://drive.google.com/file/d/1BAjEfWxN2dZOtt2-qlgF-qO7D-KHJthX/view?usp=sharing).

## Caution

The default configuration for Critical Path Extraction uses 8 threads to accommodate various CPU cores and RAM capacities, impacting only the execution speed without affecting timing performance. For reproducing the speeds reported in the paper, adjust the thread count to 52 as specified in `DATE25-TDP/thirdparty/OpenTimer/ot/timer/path.cpp` at line 426.
