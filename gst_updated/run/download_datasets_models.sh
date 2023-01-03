cd datasets/shuijing/orca_20humans_fov
cd results/visual
# use curl for big-sized google drive files to redirect
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=17QLGgTcCWatIrF4EE1Gai8BaMbqxz6kk" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > sj_dset_test_batch_trajectories.pt
cd ../../../results
# use wget for small-sized google drive files
wget -O 211203-model.zip "https://drive.google.com/uc?export=download&id=1ukON4cAwM63gKKlJJwgjcdLf7-4-f1KH" > /tmp/intermezzo.html
unzip 211203-model.zip && rm -rf 211203-model.zip