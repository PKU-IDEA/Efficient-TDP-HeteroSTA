#cd ..
#cd build
#cmake ..
#make -j 8
#make install -j 8
#cd ..
#cd install

cases=("superblue18")
# cases=("superblue1" "superblue3" "superblue4" "superblue5" "superblue7" "superblue10" "superblue16" "superblue18")

for case in "${cases[@]}"; do
  echo "Running experiment with case: $case"
  
  python dreamplace/Placer.py test/iccad2015.hs/$case.json
  
  if [ $? -ne 0 ]; then
    echo "Experiment with case $case failed."
    exit 1
  fi
done