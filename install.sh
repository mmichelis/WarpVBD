cd external/pbrt-v3
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j8
cd ../../..