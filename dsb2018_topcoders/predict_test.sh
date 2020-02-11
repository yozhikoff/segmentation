pushd selim
sh ./predict_test.sh
popd

pushd albu/src

sh ./predict_test.sh
popd

pushd victor
sh ./predict_test.sh
popd
